/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export const RELU = 'return max(a, 0.0)';
export const RELU6 = 'return (a < 0.0) ? 0.0 : min(6.0, a);';
export const LINEAR = `return a;`;
export const ELU = `return (a >= 0.0) ? a : (exp(a) - 1.0);`;

export const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
export const ABS = `return abs(a);`;
export const SQUARE = `return a * a;`;
export const NEG = `return -a;`;
export const TANH = `
  float e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
export const EXP = `return exp(a);`;
export const LOG = `if (a < 0.0) return 1.0/0.0;
  return log(a);`;

export class UnaryOpVec4Program implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workPerThread = 4;
  workGroupSize: [number, number, number];
  isVec4 = true;

  constructor(outputShape: number[], op: string) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    const size = util.sizeFromShape(this.outputShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    const fit = size % workGroupSizeX === 0;
    this.workPerThread = fit ? 1 : 2;
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    if (fit) {
      this.userCode = `
      vec4 unaryOperation(vec4 a) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        vec4 a = A[index];
        setOutput(index, unaryOperation(a));;
      }
      `;
      this.shaderKey = `unary2${op}`;
    } else {
      const type = getCoordsDataType(this.outputShape.length);
      this.userCode = `
      vec4 unaryOperation(vec4 a) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if(index < ${size}) {
          vec4 a = A[index];
          setOutput(index, unaryOperation(a));
        }
      }
      `;
      this.shaderKey = `unaryvec4${op}${type}${size}`;
    }
  }
}
