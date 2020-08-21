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

// import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, computeWorkGroupSizeForConv2d, tilesFitEvenlyIntoShape} from '../webgpu_util';
// computeWorkPerThreadForConv2d

import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {makeMatMulSource} from './matmul_webgpu';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  // TODO(texture).
  variableNames: string[] = [];
  variableTextureNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number];

  constructor(
      convInfo: backend_util.Conv2DInfo, workPerThread: number, addBias = false,
      activation: string = null, hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.workGroupSize =
        computeWorkGroupSizeForConv2d(this.dispatchLayout, this.outputShape);
    let elementsPerThread: [number, number, number];
    let matMulSource: string;
    if (workPerThread === 0) {
      elementsPerThread = [1, 1, 1];
      matMulSource = makeMatMulSource();
    } else {
      elementsPerThread = [1, 1, 1];
      // computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape);
      matMulSource = makeMatMulPackedSource(elementsPerThread);
    }
    console.log(convInfo);
    console.log(' this.outputShape = ' + this.outputShape);
    console.log(' this.workGroupSize = ' + this.workGroupSize);

    const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () =>
            // tslint:disable-next-line: max-line-length
        'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);

    // var sampleA = `imageLoad(x, ivec2(col,row)).r`;

    {
      const {newShape, keptDims} = util.squeezeShape(convInfo.inShape);
      console.log(' newInShape = ' + newShape);
      console.log(' keptInDims = ' + keptDims);
      // if this shape is 2d:
      // if this shape is 3d:
      // if this shape is 4d:
      // if (shape.length == 4) {
    }
    /*
if (texNumC === stride2 && flatOffset == null) {
  // texR is used directly as physical (no risk of float16 overflow).
  return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      float texR = dot(vec3(row, col, depth),
                       vec3(${shape[1] * shape[2]}, ${shape[2]}, 1));
      float texC = float(depth2);
      vec2 uv = (vec2(texC, texR) + halfCR) /
                vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
}
        */

    {
      const {newShape, keptDims} = util.squeezeShape(convInfo.filterShape);
      console.log(' newFilterShape = ' + newShape);
      console.log(' keptFilterDims = ' + keptDims);
      // if this shape is 2d:
      // if this shape is 3d:
      // if this shape is 4d:
    }

    // const sampleA = `imageLoad(x, ivec2(col,row)).r`;

    /*
    const sampleA = `imageLoad(x, ivec2(col,row)).r`: `coordsInBounds(coord, ${
       getShapeCoords(
           convInfo.inShape)}) ? imageLoad(x, ivec2(col,row)).r : 0`;
           */

    const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
    const sampleB = `imageLoad(W, ivec2(col,row)).r`;
    /*
        const sampleB = fitB ?
            `imageLoad(W, ivec2(col,row)).r` :
            `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            imageLoad(W, ivec2(col,row)).r : 0`;
    */

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec4 outCoord) {
              float b = getPreluActivationWeightsAtOutCoords(outCoord);
              ${activation}
            }`;
      } else {
        activationSnippet = `
              float activation(float a, ivec4 outCoord) {
                ${activation}
              }
            `;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet = addBias ? 'ivec4 coords = getOutputCoords(); ' +
            'value += getBiasAtOutCoords(outCoord);' :
                                     '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.userCode = `
        ${activationSnippet}
        ${matMulSource}

        int batch;
        int dimAOuter = ${this.outputShape[1]} * ${this.outputShape[2]};
        int dimBOuter = ${this.outputShape[3]};
        int dimInner = filterDims[0] * filterDims[1] * ${convInfo.inShape[3]};
        float mm_readA(int row, int col) {
          int r = int(row), c = int(col);
          int outRow = r / ${this.outputShape[2]};
          int outCol = r % ${this.outputShape[2]};
		  
          // filter_W = WROW, filter_H = WCol
          int WRow = c / (filterDims[1] * ${convInfo.inShape[3]});
          int WCol = (c / ${convInfo.inShape[3]}) % filterDims[1];
          ivec4 coord = ivec4(
              batch,
              outRow * stride[0] + dilation[0] * WRow - pad[0],
              outCol * stride[1] + dilation[1] * WCol - pad[1],
              c % ${convInfo.inShape[3]});

          // TODO(texture): For 4D(if (texNumC === stride2) only:
          int texR = int(dot(vec3(coord[0], coord[1], coord[2]), vec3(${
        convInfo.inShape[1]} * ${convInfo.inShape[2]}, ${
        convInfo.inShape[2]}, 1)) );
          int texC = coord[3];
          //return getX(coord[0],coord[1], coord[2], coord[3]);
          return imageLoad(x, ivec2(texC,texR)).r;
        }

        // dispatch 1,1,1
        // workgroup size: 4, 16, 1
    
        float mm_readB(int row, int col) {
          // return ${sampleB};
          /*
          int col_ = row/${convInfo.filterShape[0]};
          int row_ = row%${convInfo.filterShape[0]};
          ivec2 uv = uvFromFlat(18,2,row);
          */
          //return imageLoad(W, ivec2(row,col)).r;
          // TODO(texture): use getW instead.
          // ivec2 uv = ivec2(row/18, row%18);
          ivec2 uv = ivec2(col, row);
          return imageLoad(W, ivec2(uv.x,uv.y)).r;
        }

        void mm_write(int row, int col, float value) {
          ivec4 outCoord = ivec4(
              batch,
              row / ${this.outputShape[2]},
              row % ${this.outputShape[2]},
              col);
          ${addBiasSnippet}
          ${applyActivationSnippet}
        //imageStore(result, ivec2(outCoord[3], outCoord[2]), vec4(value, 0.0, 0.0, 0.0));
        int texR = int(dot(vec3(outCoord[0], outCoord[1], outCoord[2]), vec3(${
          convInfo.outShape[1]} * ${convInfo.outShape[2]}, ${
          convInfo.outShape[2]}, 1)) );
            int texC = outCoord[3];
            imageStore(result, ivec2(texC,texR), vec4(value, 0.0, 0.0, 0.0));
    }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
      this.shaderKey = `conv2dmm'${elementsPerThread.join('')}${fitA}${fitB}${
        addBiasSnippet}${activationSnippet}${convInfo.inShape}${convInfo.outShape}${convInfo.filterShape}`;
      console.log(this.shaderKey);
      
  }
}
