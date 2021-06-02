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

import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {makeMatMulPackedVec4WGSLSource} from './matmul_packed_vec4_webgpu';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DMMVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number];
  useWGSL = true;
  isVec4 = true;
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;
  hasLeakyreluAlpha: boolean;
  fitA: boolean;
  fitB: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivationWeights = false,
      hasLeakyreluAlpha = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.workGroupSize = [16, 16, 1];
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);
    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.hasLeakyreluAlpha = hasLeakyreluAlpha;
    if (this.addBias) {
      this.variableNames.push('bias');
    }

    if (this.hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    if (this.hasLeakyreluAlpha) {
      this.variableNames.push('leakyreluAlpha');
    }

    [this.fitA, this.fitB] = this.getShapeFit(elementsPerThread);
    this.shaderKey =
        `conv2DMMVec4_${this.activation}_${this.fitA}_${this.fitB}`;
  }

  getShapeFit(elementsPerThread: [number, number, number]): boolean[] {
    const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
    const tileInner = tileBOuter;

    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
        this.convInfo.inChannels;
    return [
      tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]),
      tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter])
    ];
  }

  getUserCode(): string {
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    const matMulSource = makeMatMulPackedVec4WGSLSource("", "", elementsPerThread);

    // Below code only applys to valid padding type.
    const sampleAWithRemainder = `int flatIndex = getFlatIndex(coord, xShape);
        int divBy4Remainder = flatIndex % 4;
        int divBy4Index = flatIndex / 4;
        vec4 curData = x[divBy4Index];
        if (divBy4Remainder == 0) {
          temp = curData;
        } else {
          // TODO: This could end up being a redundant load with another one in
          // the same shader invocation. Perhaps there's an opportunity for
          // optimization
          vec4 nextData = x[divBy4Index + 1];
          if (divBy4Remainder == 1) {
            temp = vec4(curData.yzw, nextData.x);
          } else if (divBy4Remainder == 2) {
            temp = vec4(curData.zw, nextData.xy);
          } else if (divBy4Remainder == 3) {
            temp = vec4(curData.w, nextData.xyz);
          }
        }
        `;

    const remainder = this.convInfo.inChannels % 4;
    const remainderSnippet = remainder === 0 ?
        `// The bounds checking is always needed since we use it to pad zero for
        // the 'same' padding type.
        resData = coordsInBounds(coord, xShape) ?
        x[getFlatIndex(coord, xShape) / 4] : vec4(0.0, 0.0, 0.0, 0.0);` :
        `vec4 temp = vec4(0, 0, 0, 0);
        ${sampleAWithRemainder}
        resData = temp;
        if (WCol == (filterDims[1] - 1)) {
          coord = ivec4(
            coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
          ${sampleAWithRemainder}
          if (inChCoord == 0) {
            resData = vec4(resData.xyz, temp.x);
          } else if (inChCoord == 1) {
            resData = vec4(resData.xy, temp.xy);
          } else {
            resData = vec4(resData.x, temp.xyz);
          }
        }
        `;

    const readASnippet = `int outRow = r / outShape[2];
        int outCol = r % outShape[2];
        int WRow = c / (filterDims[1] * xShape[3]);
        int WCol = (c / xShape[3]) % filterDims[1];
        int inChCoord = c % xShape[3];
        ivec4 coord = ivec4(
            batch,
            outRow * stride[0] + dilation[0] * WRow - pad[0],
            outCol * stride[1] + dilation[1] * WCol - pad[1],
            inChCoord);
        vec4 resData = vec4(0, 0, 0, 0);
        ${remainderSnippet}
        return resData;`;

    const sampleA =
        this.fitA ? `${readASnippet}` : `if (r < dimAOuter && c < dimInner) {
          ${readASnippet}
        } else {
          return vec4(0.0, 0.0, 0.0, 0.0);
        }`;

    const sampleB = this.fitB ?
        `W[row * dimBOuter / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            W[row * dimBOuter / 4 + col] : vec4(0.0, 0.0, 0.0, 0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `fn activation(a : vec4<f32>, outCoord : vec4<u32>) -> vec4<f32>{
          let b: vec4<f32> = getPreluActivationWeightsAtOutCoords(outCoord);
          ${this.activation}
        }`;
      } else if (this.hasLeakyreluAlpha) {
        activationSnippet = `fn activation(a: vec4<f32>) ->vec4<f32> {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${this.activation}
        }`;
        throw new Error('Leakyrelu is not supported.');
      } else {
        activationSnippet = `
        vec4 activation(vec4 a, ivec4 outCoord) {
          ${this.activation}
        }`;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet = this.addBias ? 'let coords : vec4<u32>= getOutputCoords(); ' +
            'value += getBiasAtOutCoords(outCoord);' :
                                          '';

    /*
    fn mm_readA(row : u32, col : u32) -> vec4<f32>  {
    if (row < uniforms.dimAOuter && col < uniforms.dimInner)
    {
        let result : vec4<f32> = firstMatrix.numbers[row * uniforms.dimInner / 4u + col];
        return result;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn mm_readB(row : u32, col : u32) -> vec4<f32> {
    if (row < uniforms.dimInner && col < uniforms.dimBOuter)
    {
        let result : vec4<f32> = secondMatrix.numbers[row * uniforms.dimBOuter / 4u + col];
        return result;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn mm_write(row : u32, col : u32, value : vec4<f32>) {
    if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
    {
        ${addBiasSnippet}
        ${applyActivationSnippet}
        let index : u32 = col + row * uniforms.dimBOuter / 4u;
        resultMatrix.numbers[index] = value;
    }
}
    
    */

    const userCode = `
        ${activationSnippet}
        ${matMulSource}

        var batch : u32;
        var dimAOuter : u32 = outShape[1] * outShape[2];
        var dimBOuter : u32 = outShape[3];
        var dimInner : u32 = filterDims[0] * filterDims[1] * xShape[3];
        fn mm_readA(row : u32, col : u32) -> vec4<f32> {
          let r = u32(row), c = u32(col * 4);
          ${sampleA};
        }

        fn mm_readB(row : u32, col : u32) -> vec4<f32> {
          return ${sampleB};
        }

        fn mm_write(row : u32, col : u32, value : vec4<f32>) {
          if (row < dimAOuter && col * 4 < dimBOuter)
          {
            let outCoord : vec4<u32> = vec4<u32>(
              batch,
              row / outShape[2],
              row % outShape[2],
              col * 4);
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }

        [[stage(compute), workgroup_size(16, 16, 1)]]
fn main([[builtin(local_invocation_id)]] local_id : vec3<u32>,
        [[builtin(global_invocation_id)]] global_id  : vec3<u32>) {
          batch = int(global_id.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
    return userCode;
  }
}
