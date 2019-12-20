/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {computeDispatch} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2DSharedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation, inDims;';
  workGroupSize: [number, number, number] = [4, 8, 4];
  workPerThread2: [number, number, number] = [2, 2, 1];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.workPerThread2);
    const channelMul = convInfo.outChannels / convInfo.inChannels;
    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');

    this.userCode = `
      void writeResult(int batch, int row, int col, int chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, outShape)) {
          setOutput(batch, row, col, chan, value);
        }
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        ivec2 xRCCorner = coords.yz * stride - pad;
        int d2 = coords[3];
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner_orig = xRCCorner.x;
        int xCCorner_orig = xRCCorner.y;
        int xRCorner = 0;
        int xCCorner = 0;

        for (int i = 0; i < 2; i ++) {
          xRCorner = xRCorner_orig + i;
          for (int j = 0; j < 2; j ++) {
            xCCorner = xCCorner_orig + j;

            // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
            // ? = to be determined. : = across all values in that axis.
            float dotProd = 0.0;
            // TODO(xing.xu): Flatten the two for loops and vec4 the operations.
            //for (int wR = 0; wR < filterDims[0]; wR++) {
            for (int wR = 0; wR < filterDims[0]; wR++) {
              int xR = xRCorner + wR * dilation[0];

              if (xR < 0 || xR >= inDims[0]) {
                continue;
              }

              for (int wC = 0; wC < filterDims[1]; wC++) {
                int xC = xCCorner + wC * dilation[1];

                if (xC < 0 || xC >= inDims[1]) {
                  continue;
                }

                float xVal = getX(batch, xR, xC, d1);
                float wVal = getW(wR, wC, d1, q);
                dotProd += xVal * wVal;
                //dotProd  += xVal;// float(i*2+j);
                //dotProd  = wVal;
              }
            }
            //dotProd  = getW(i, j, d1, q);
            writeResult(batch, coords[1]+i, coords[2]+j, d2, dotProd);
          }
        }
      }

    `;

    this.shaderKey = `depthwise${channelMul}`;
  }
}
