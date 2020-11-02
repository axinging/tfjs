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
  // TODO(texture): make this works under WPT = 8.
  workPerThread = 1;
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
    const inShape = xShape;

    const leftPadCondition =
        rank > 1 ? `any(lessThan(outC, start))` : `outC < start`;
    const rightPadCondition =
        rank > 1 ? `any(greaterThanEqual(outC, end))` : `outC >= end`;

    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';
    console.log(unpackedCoords);
    const dims = ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(
        0, this.outputShape.length);
    dims.map(d => `${d}`).join(', ');
    this.userCode = `
      ${type} start = ${startValue};
      ${type} end = ${endValue};

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for (int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if (flatIndex < ${size}) {
            ${type} outC = getCoordsFromFlatIndex(flatIndex);

            if (${leftPadCondition} || ${rightPadCondition}) {
			  ${type} coords = outC;
              setOutput(${dims}, ${constantValue});
            } else {
              ${type} coords = outC - start;
              //setOutput(flatIndex, getX(${unpackedCoords}));
              // setOutput(${dims}, getX(${unpackedCoords}));
              setOutput(getX(${unpackedCoords}));
              int texR, texC;
			  /*

              texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        inShape[1]} * ${inShape[2]}, ${inShape[2]}, 1)) );
              texC = coords[3];
    
              float value = imageLoad(x, ivec2(texC,texR)).r;
              */
              /*
              int texR2 = int(dot(vec3(outCoord[0], outCoord[1], outCoord[2]), vec3(${
        this.outputShape[1]} * ${this.outputShape[2]}, ${
        this.outputShape[2]}, 1)) );
              int texC2 = outCoord[3];
              imageStore(result, ivec2(texC2,texR2), vec4(value, 0.0, 0.0, 0.0));
              */
             /*
              ivec2 uv = uvFromFlat(9, 5, flatIndex); // padtexture0d1 works.
              //ivec2 uv = getUVFromFlat(flatIndex);
              //ivec2 uv = uvFromFlat(121, 3, flatIndex); // padtexture0d2 works?.
              imageStore(result, ivec2(uv.y,uv.x), vec4(value, 0.0, 0.0, 0.0));
              */
              //
            }
          }
        }
      }
    `;
    this.shaderKey =
        `pad${startValue}${endValue}${rank}${size}${type}${constantValue}`;
  }
}
