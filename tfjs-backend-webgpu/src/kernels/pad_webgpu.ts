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

import {util} from '@tensorflow/tfjs-core';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class PadProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  // TODO(texture).
  variableNames: string[] = [];
  variableTextureNames = ['x'];
  workPerThread = 8;
  workGroupSize: [number, number, number] = [16, 1, 1];

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      constantValue: number) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    console.log('this.outputShape = ' + this.outputShape);
    const size = util.sizeFromShape(this.outputShape);
    const type = getCoordsDataType(rank);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
    const endValue = rank > 1 ? `${type}(${end})` : `${end}`;

    const leftPadCondition =
        rank > 1 ? `any(lessThan(outCoords, start))` : `outCoords < start`;
    const rightPadCondition =
        rank > 1 ? `any(greaterThanEqual(outCoords, end))` : `outCoords >= end`;

    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';
    console.log(unpackedCoords);
    const dims = ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(
        0, this.outputShape.length);
    dims.map(d => `${d}`).join(', ');

    const dims2 =
        ['outCoords[0]', 'outCoords[1]', 'outCoords[2]', 'outCoords[3]'].slice(
            0, this.outputShape.length);
    dims2.map(d => `${d}`).join(', ');
    this.userCode = `
      ${type} start = ${startValue};
      ${type} end = ${endValue};

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        for (int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < ${size}) {
            ${type} outCoords = getCoordsFromFlatIndex(flatIndex);

            if (${leftPadCondition} || ${rightPadCondition}) {
			  ${type} coords = outCoords;
              setOutput(${dims}, ${constantValue});
            } else {
              ${type} coords = outCoords - start;
              setOutput(${dims2},  getX(${unpackedCoords}));
            }
          }
        }
      }
    `;
    this.shaderKey =
        `pad${startValue}${endValue}${rank}${size}${type}${constantValue}`;
  }
}
