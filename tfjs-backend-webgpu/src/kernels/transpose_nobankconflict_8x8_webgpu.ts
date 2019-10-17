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

import {computeDispatch} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class TransposeNoBankConflict8x8Program implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  rank: number;
  workPerThread = 32/8;
  workGroupSize: [number, number, number] = [32,8,1];

  constructor(aShape: number[], newDim: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[newDim[i]];
    }
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    this.dispatchLayout = {x: [0], y: [1]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape,
        this.workGroupSize, [this.workPerThread, 1 ,1]);
    this.userCode = `
    const int TILE_DIM = ${this.workGroupSize[0]};
    const int BLOCK_ROWS =  ${this.workGroupSize[1]};

    shared float tile[TILE_DIM][TILE_DIM+1];
    void main() {
        int index = int(gl_GlobalInvocationID.x);
        int x = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.x);
        int y = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.y);
        int width = int(gl_NumWorkGroups.x) * TILE_DIM;
        for (int j = 0; j < TILE_DIM; j += TILE_DIM/${this.workPerThread}) {
          tile[gl_LocalInvocationID.y + j][gl_LocalInvocationID.x] =
              A[(y + j) * width + x];
        }
        barrier();

        x = int(gl_WorkGroupID.y) * TILE_DIM + int(gl_LocalInvocationID.x);
        y = int(gl_WorkGroupID.x) * TILE_DIM + int(gl_LocalInvocationID.y);
        for (int j = 0; j < TILE_DIM; j += TILE_DIM/${this.workPerThread}) {
          int flatIndex = (y+j) * width + x;
          setOutput(flatIndex, tile[gl_LocalInvocationID.x]
            [gl_LocalInvocationID.y+j]);
        }
      }
    `;
  }
}