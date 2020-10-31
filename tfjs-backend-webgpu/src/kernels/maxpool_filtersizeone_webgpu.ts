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

import {backend_util} from '@tensorflow/tfjs-core';

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class MaxPoolWithFilterSizeEqualsOneProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  // variableNames = ['x'];
  // TODO(texture).
  variableNames: string[] = [];
  variableTextureNames = ['x'];
  uniforms = 'ivec2 pad, stride, dilation, convDims, filterDims;';
  workGroupSize: [number, number, number] = [4, 4, 4];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [0, 1], y: [2], z: [3]};

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        if (all(lessThan(coords, ${getShapeCoords(this.outputShape)}))) {
          ivec2 xRCCorner = coords.yz * stride;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          /* this works.
          int texR, texC;

          texR = int(dot(vec3(batch, xRCorner, xCCorner), vec3(${
        convInfo.inShape[1]} * ${convInfo.inShape[2]}, ${
        convInfo.inShape[2]}, 1)) );
          texC = d;

          float value = imageLoad(x, ivec2(texC,texR)).r;
          */
          float value = getX(batch, xRCorner, xCCorner, d);

          setOutput(batch, coords[1], coords[2], d, value);
          /*
          ivec4 outCoord = ivec4(batch, coords[1], coords[2], d);
          int texR2 = int(dot(vec3(outCoord[0], outCoord[1], outCoord[2]), vec3(${
        convInfo.outShape[1]} * ${convInfo.outShape[2]}, ${
        convInfo.outShape[2]}, 1)) );
          int texC2 = outCoord[3];
          imageStore(result, ivec2(texC2,texR2), vec4(value, 0.0, 0.0, 0.0));
          */
        }
      }
    `;
    this.shaderKey = 'maxpoolv2';
  }
}
