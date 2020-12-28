/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

function makeMatMulPackedVec4Source(workPerThread: number[]): string {
  return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${workPerThread[0]};
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileBOuter;

    shared vec4 mm_Asub[TileAOuter][TileInner / ColPerThread];
    shared vec4 mm_Bsub[TileInner][TileBOuter / ColPerThread];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x);

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x);

      int numTiles = (dimInner - 1) / TileInner + 1;

      vec4 acc[RowPerThread];
      vec4 ACached;
      vec4 BCached[4];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          acc[innerRow] = vec4(0.0, 0.0, 0.0, 0.0);
      }

      // Loop over shared dimension.
      int globalColA = tileCol;
      int tileRowB = int(gl_LocalInvocationID.y) * ColPerThread;
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                globalColA);
        }
        globalColA += TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < ColPerThread; innerRow++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol);
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner / ColPerThread; k++) {
          BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
          BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
          BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

          for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[tileRow + i][k];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
          }
        }
        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow,
          globalCol,
          acc[innerRow]);
      }
    }
  `;
}

export class Conv2DMMVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number];
  isVec4 = true;

  constructor(
      convInfo: backend_util.Conv2DInfo, usePackedTexture = false,
      addBias = false, activation: string = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.workGroupSize = [16, 16, 1];
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    const matMulSource = makeMatMulPackedVec4Source(elementsPerThread);

    const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
    const tileInner = tileBOuter;

    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);
    const batchSize =
        this.outputShape[1] * this.outputShape[2] * this.outputShape[3] / 4;

    const glslZERO = 'vec4(0.0, 0.0, 0.0, 0.0)';
    const vecSize = 4;
    let sampleA;
    if (usePackedTexture) {
      sampleA = fitA ?
          `getX(coord[0],coord[1], coord[2], coord[3])` :
          `coordsInBounds(coord, ${
              getShapeCoords(
                  convInfo
                      .inShape)}) ? getX(coord[0],coord[1], coord[2], coord[3]) : ${
              glslZERO}`;
    } else {
      sampleA = fitA ?
          `x[getFlatIndex(coord, ${getShapeCoords(convInfo.inShape)}) / 4]` :
          `coordsInBounds(coord, ${
              getShapeCoords(convInfo.inShape)}) ? x[getFlatIndex(coord, ${
              getShapeCoords(convInfo.inShape)}) / 4] : ${glslZERO}`;
    }
    const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
    let sampleB;
    if (usePackedTexture) {
      sampleB = fitB ?
          `getW(index)` :
          `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
          getW(index) : ${glslZERO}`;
    } else {
      sampleB = fitB ?
          `W[row * dimBOuter + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner * 4, dimBOuter)) ?
          W[row * dimBOuter + col] : ${glslZERO}`;
    }
    let sampleResult;
    if (usePackedTexture) {
      sampleResult = `setOutput(batch, row, col * ${vecSize}, value)`;
    } else {
      sampleResult =
          `result[batch * ${batchSize} + row * dimBOuter + col] = value`;
    }

    // TODO(jiajia.qin@intel.com): Add the fused conv2d vec4 support.
    this.userCode = `
        ${matMulSource}

        int batch;
        int dimAOuter = ${this.outputShape[1]} * ${this.outputShape[2]};
        int dimBOuter = ${this.outputShape[3]};
        int dimInner = filterDims[0] * filterDims[1] * ${
        convInfo.inShape[3]};
        vec4 mm_readA(int row, int col) {
          int r = int(row), c = int(col * 4);
          if (r < dimAOuter && c < dimInner)
          {
          int outRow = r / ${this.outputShape[2]};
          int outCol = r % ${this.outputShape[2]};

          int WRow = c / (filterDims[1] * ${convInfo.inShape[3]});
          int WCol = (c / ${convInfo.inShape[3]}) % filterDims[1];

          int inChCoord = c % ${convInfo.inShape[3]};
          ivec4 coord = ivec4(
              batch,
              outRow * stride[0] + dilation[0] * WRow - pad[0],
              outCol * stride[1] + dilation[1] * WCol - pad[1],
              inChCoord);
          vec4 resData = ${sampleA};
          if (inChCoord < (${convInfo.inShape[3]} - 3))
          {
             // do nothing
          } else if (inChCoord < (${convInfo.inShape[3]} - 2))
          {
            if (WCol < (filterDims[1] - 1))
            {
              coord = ivec4(
                coord.x, coord.y, coord.z + 1, 0);
              WCol = WCol + 1;
            } else {
              coord = ivec4(
                coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
              WCol = 0;
            }

            vec4 temp = ${sampleA};
            resData = vec4(resData.xyz, temp.x);
          } else if (inChCoord < (${convInfo.inShape[3]} - 1))
          {
            if (WCol < (filterDims[1] - 1))
            {
              coord = ivec4(
                coord.x, coord.y, coord.z + 1, 0);
              WCol = WCol + 1;
            } else {
              coord = ivec4(
                coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
              WCol = 0;
            }
            vec4 temp = ${sampleA};
            resData = vec4(resData.xy, temp.xy);
            if (${convInfo.inShape[3]} < 2)
            {
              if (WCol < (filterDims[1] - 1))
              {
                coord = ivec4(
                  coord.x, coord.y, coord.z + 1, 0);
                WCol = WCol + 1;
              } else {
                coord = ivec4(
                  coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
                WCol = 0;
              }
              temp = ${sampleA};
              resData = vec4(resData.xyz, temp.x);
            }
          } else if (inChCoord < ${convInfo.inShape[3]})
          {
            if (WCol < (filterDims[1] - 1))
            {
              coord = ivec4(
                coord.x, coord.y, coord.z + 1, 0);
              WCol = WCol + 1;
            } else {
              coord = ivec4(
                coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
              WCol = 0;
            }
            vec4 temp = ${sampleA};
            resData = vec4(resData.x, temp.xyz);
            if (${convInfo.inShape[3]} < 3)
            {
              if (${convInfo.inShape[3]} == 2)
              {
                if (WCol < (filterDims[1] - 1))
                {
                  coord = ivec4(
                    coord.x, coord.y, coord.z + 1, 0);
                  WCol = WCol + 1;
                } else {
                  coord = ivec4(
                    coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
                  WCol = 0;
                }
                temp = ${sampleA};
                resData = vec4(resData.xyz, temp.x);
              } else if (${convInfo.inShape[3]} == 1)
              {
                if (WCol < (filterDims[1] - 1))
                {
                  coord = ivec4(
                    coord.x, coord.y, coord.z + 1, 0);
                  WCol = WCol + 1;
                } else {
                  coord = ivec4(
                    coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
                  WCol = 0;
                }
                temp = ${sampleA};
                resData = vec4(resData.xy, temp.x, 0);

                if (WCol < (filterDims[1] - 1))
                {
                  coord = ivec4(
                    coord.x, coord.y, coord.z + 1, 0);
                  WCol = WCol + 1;
                } else {
                  coord = ivec4(
                    coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
                  WCol = 0;
                }
                temp = ${sampleA};
                resData = vec4(resData.xyz, temp.x);
              }
            }
          }

            if (c < (dimInner - 3))
            {
            }
            else if (c < (dimInner - 2))
            {
              resData = vec4(resData.xyz, 0);
            } else if (c < (dimInner - 1))
            {
              resData = vec4(resData.xy, 0, 0);
            } else if (c < dimInner)
            {
              resData = vec4(resData.x, 0, 0, 0);
            }
            return resData;
          } else {
            return vec4(0, 0, 0, 0);
          }
        }

        vec4 mm_readB(int row, int col) {
          int index = row * dimBOuter + col * 4;
          return ${sampleB};
        }

        void mm_write(int row, int col, vec4 value) {
          if (row < dimAOuter && col * 4 < dimBOuter)
          {
            ${sampleResult};
          }
        }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
  }
}
