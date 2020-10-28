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
import {getCoordsDataType} from '../shader_preprocessor';

// computeDispatch
import {computeDispatch, getTextureShapeFromLogicalShape} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  // dispatchLayout: {x: number[]};
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  // TODO:
  variableNames: string[] = [];
  variableTextureNames = ['A', 'B'];
  workPerThread: number;
  workGroupSize: [number, number, number];

  constructor(op: string, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    // const outputShapeLogical =
    // backend_util.assertAndGetBroadcastShape(aShape, bShape); this.outputShape
    // = getTextureShapeFromLogicalShape(outputShapeLogical);
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    const outputShapeLogical =
        getTextureShapeFromLogicalShape(this.outputShape);
    this.dispatchLayout = {
      x: [0],
      y: [1]
    };  // flatDispatchLayout(this.outputShape);
    const size = util.sizeFromShape(this.outputShape);
    const sizeFit = size % workGroupSizeX === 0;
    const shapesFit = util.arraysEqual(aShape, bShape) && sizeFit;
    this.workPerThread = 1;  // shapesFit || sizeFit ? 1 : 2;

    this.dispatch = computeDispatch(
        this.dispatchLayout, outputShapeLogical, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    // this.dispatch = [10, 3, 1];


    if (shapesFit) {
      console.error("TODO(texture): not tried");
      this.userCode = `
          float binaryOperation(float a, float b) {
            ${op}
          }

          void main() {
            //int index = int(gl_GlobalInvocationID.x);

            // float a = A[index];
            // float b = B[index];
            // float a = imageLoad(A, ivec2(index, 0)).r+imageLoad(A, ivec2(index, 0)).g;//+imageLoad(A, ivec2(index, 0)).b+imageLoad(A, ivec2(index, 0)).a;
            // float b = imageLoad(B, ivec2(index, 0)).r+imageLoad(B, ivec2(index, 0)).g;//+imageLoad(B, ivec2(index, 0)).b+imageLoad(B, ivec2(index, 0)).a;
            float a = imageLoad(A, ivec2(gl_GlobalInvocationID.xy)).r;
            float b = imageLoad(B, ivec2(gl_GlobalInvocationID.xy)).r;
            imageStore(result, ivec2(gl_GlobalInvocationID.xy), vec4(binaryOperation(a, b), 100.0, 101.0, 102.0));
            // imageStore(result, ivec2(gl_GlobalInvocationID.xy), vec4(binaryOperation(a, b)+99.0, 100.0, 101.0, 102.0));
            // imageStore(result, ivec2(gl_GlobalInvocationID.xy), vec4(float(index)+10.0, 100.0, 101.0, 102.0));
          }
        `;
      this.shaderKey = `binary2${op}`;
    } else if (sizeFit) {
      console.error("TODO(texture): not tried");
      const type = getCoordsDataType(this.outputShape.length);
      this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        ${type} coords = getCoordsFromFlatIndex(index);
        float a = imageLoad(A, ivec2(index, 0)).r;
        float b = imageLoad(B, ivec2(index, 0)).r;

        //float a = getAAtOutCoords(coords);
        //float b = getBAtOutCoords(coords);
        imageStore(result, ivec2(gl_GlobalInvocationID.xy), vec4(binaryOperation(a, b), 0.0, 0.0, 0.0));
      }
      `;
    } else {
      const type = getCoordsDataType(this.outputShape.length);

      this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);
            float a = getAAtOutCoords();
            float b = getBAtOutCoords();
            setOutput(binaryOperation(a, b));
          }
        }
      }
      `;
      this.shaderKey = `binary${op}${type}${size}`;
    }
  }
}
