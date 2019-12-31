/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/*
import {backend_util, util} from '@tensorflow/tfjs-core';

import {Conv2DInfo} from '../../ops/conv_util';
import {computeDispatch} from '../webgpu_util';

import {GPGPUProgram} from './gpgpu_math';
import {WebGPUProgram} from './webgpu_program';
*/
import {backend_util} from '@tensorflow/tfjs-core';
import {computeDispatch} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2DSharedProgram implements WebGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;
  localGroupSize: [number, number];

  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  // variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation, inDims;';
  workGroupSize: [number, number, number] = [4, 8, 4];
  workPerThread2: [number, number, number] = [1, 1, 1];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    const xNumRows = convInfo.inHeight;
    const xNumCols = convInfo.inWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const channelMul = convInfo.outChannels / convInfo.inChannels;
    const OUTWIDTH_DIVIDER = 5;

    this.workGroupSize = [8 * channelMul, OUTWIDTH_DIVIDER, 1];
    this.localGroupSize = [8 * channelMul, OUTWIDTH_DIVIDER];

    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.workPerThread2);
    console.log(
        'this.dispatch=' + this.dispatch + ', this.localGroupSize[1] ' +
        this.localGroupSize[1] + ', filterWidth=' + filterWidth);

    const c_h = filterHeight;
    const c_w = (this.localGroupSize[1] - 1) * strideWidth + filterWidth +
        (filterWidth - 1) * (dilationWidth - 1);

    console.log('CACHE_W=' + c_w);

    const c_c = this.localGroupSize[0] < convInfo.outChannels ?
        this.localGroupSize[0] / channelMul :
        convInfo.inChannels;
    console.log('CACHE_C=' + c_c);

    console.log('CACHE_HWC=' + c_h * c_w * c_c);
    // outWidth should be divisible by localGroupSize[1]
    // const getCoords = generateGetCoordsFromFlatIndex(this.outputShape);

    this.userCode = `
    const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
    const ivec2 pads = ivec2(${padTop}, ${padLeft});

    const int CACHE_H = ${filterHeight};
    const int CACHE_W = ${
        (this.localGroupSize[1] - 1) * strideWidth + filterWidth +
        (filterWidth - 1) * (dilationWidth - 1)};
    const int CACHE_C = ${
        this.localGroupSize[0] < convInfo.outChannels ?
            this.localGroupSize[0] / channelMul :
            convInfo.inChannels};
    const int CACHE_WC = CACHE_W * CACHE_C;
    const int CACHE_HWC = CACHE_H * CACHE_W * CACHE_C;
    // Combine CACHE_W and CACHE_C
    shared float cache[CACHE_H][CACHE_W * CACHE_C];
    ivec4 getFirstThreadOutputCoords() {
      ivec2 firstThreadGlobalInvocationID =
          ivec2(gl_WorkGroupID * gl_WorkGroupSize);
      int index = firstThreadGlobalInvocationID.y * ${this.outputShape[1]} +
          firstThreadGlobalInvocationID.x;
      return getCoordsFromFlatIndex(index);//ivec4(r, c, d, d2);
    }

    int getFirstThreadOutputIndex() {
      ivec2 firstThreadGlobalInvocationID =
          ivec2(gl_WorkGroupID * gl_WorkGroupSize);
      int index = firstThreadGlobalInvocationID.y * ${this.outputShape[1]} +
          firstThreadGlobalInvocationID.x;
      return index;
    }

    void writeResult(int batch, int row, int col, int chan, float value) {
      ivec4 coord = ivec4(batch, row, col, chan);
      if (coordsInBounds(coord, outShape)) {
        setOutput(batch, row, col, chan, value);
      }
    }
    void main() {
      ivec4 coords = getFirstThreadOutputCoords();
      int batch = coords.x;
      ivec2 cacheRCCorner = coords.yz * strides - pads;
      int cacheRCorner = cacheRCCorner.x;
      int cacheCCorner = cacheRCCorner.y;
      int cacheDCorner = coords.w / ${channelMul};

      // Fill cache using all threads in a local group
      int index = int(gl_LocalInvocationIndex);
      while (index < CACHE_HWC) {
        int r = index / CACHE_WC;
        int cd = index - r * CACHE_WC;
        int c = cd / CACHE_C;
        int d = cd - c * CACHE_C;

        int xR = cacheRCorner + r * ${dilationHeight};
        int xC = cacheCCorner + c;
        int xD = cacheDCorner + d;
        if (xR >= 0 && xR < ${convInfo.inHeight} &&
            xC >= 0 && xC < ${convInfo.inWidth} &&
            xD < ${convInfo.inChannels}) {
          cache[r][cd] = getX(batch, xR, xC, xD);
        }

        index += ${this.localGroupSize[0] * this.localGroupSize[1]};
      }

      memoryBarrierShared();
      barrier();

      // Discard threads that are out of X bounds
      //if (int(gl_GlobalInvocationID.x) >= ${convInfo.outChannels}) {
      //  return;
      //}

      coords = getOutputCoords();
      ivec2 xRCCorner = coords.yz * strides - pads;
      int xRCorner = xRCCorner.x;
      int xCCorner = xRCCorner.y;
      int d2 = coords.w;
      int d1 = d2 / ${channelMul};
      int q = d2 - d1 * ${channelMul};

      // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
      // ? = to be determined. : = across all values in that axis.
      float dotProd = 0.0;
      // TODO: Flatten the two for loops and vec4 the operations.
      for (int wR = 0; wR < ${filterHeight}; wR++) {
        int xR = xRCorner + wR * ${dilationHeight};
        if (xR < 0 || xR >= ${xNumRows}) {
          continue;
        }
        int sR = wR;

        for (int wC = 0; wC < ${filterWidth}; wC++) {
          int xC = xCCorner + wC * ${dilationWidth};
          if (xC < 0 || xC >= ${xNumCols}) {
            continue;
          }
          int sC = (xC - cacheCCorner) * CACHE_C;

          float xVal = cache[sR][sC + d1 - cacheDCorner];
          float wVal = getW(wR, wC, d1, q);
          dotProd += xVal * wVal;
        }
      }
      //setOutput(dotProd);

      writeResult(batch, coords[1], coords[2], d2, dotProd);
    }


    `;
  }
}
