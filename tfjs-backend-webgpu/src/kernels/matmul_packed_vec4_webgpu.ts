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

import {TensorInfo} from '@tensorflow/tfjs-core';
import {computeDispatch, computeWorkGroupSizeForMatMul, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedVec4Source(workPerThread: number[]): string {
  console.log("makeMatMulPackedVec4Source");
  return `
    //vec4 mm_readA(int row, int col);
    //vec4 mm_readB(int row, int col);
    //void mm_write(int row, int col, vec4 value);
    [[block]] struct Uniforms {
      dimAOuter : u32;
      dimInner : u32;
      dimBOuter : u32;
  };
  [[block]] struct Matrix {
      numbers: array<f32>;
  };

  [[group(0), binding(0)]] var<storage> firstMatrix : [[access(read)]] Matrix;
  [[group(0), binding(1)]] var<storage> secondMatrix : [[access(read)]] Matrix;
  [[group(0), binding(2)]] var<storage> resultMatrix : [[access(write)]] Matrix;
  [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

    let RowPerThread : u32 = ${workPerThread[1]};
    let ColPerThread : u32 = ${
      workPerThread[0]}; // only support ColPerThread = 4
    let TileAOuter : u32 = int(gl_WorkGroupSize.y) * RowPerThread;
    let TileBOuter : u32 = int(gl_WorkGroupSize.x) * ColPerThread;
    let TileInner : u32 = TileBOuter;

    // var<workgroup> mm_Asub : array<f32, 4096>;
    shared vec4 mm_Asub[TileAOuter][TileInner / ColPerThread];
    shared vec4 mm_Bsub[TileInner][TileBOuter / ColPerThread];

    fn mm_matMul(dimAOuter : i32, dimInner : i32, dimBOuter : i32) {
      let tileRow : i32 = int(gl_LocalInvocationID.y) * RowPerThread;
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
      var globalColA : u32 = tileCol;
      let RowPerThreadB : u32 = TileInner / int(gl_WorkGroupSize.y);
      let tileRowB  : u32 = int(gl_LocalInvocationID.y) * RowPerThreadB;
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < RowPerThread; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                globalColA);
        }
        globalColA += TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol);
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileInner / ColPerThread; k++) {
          BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
          BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
          BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

          for (var i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[tileRow + i][k];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
          }
        }
        workgroupBarrier();
      }

      for (var innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow,
          globalCol,
          acc[innerRow]);
      }
    }
  `;
}

export function makeMatMulVectorVec4Source(): string {
  console.log("makeMatMulVectorVec4Source");
  return `
    //vec4 mm_readA(int row, int col);
    //vec4 mm_readB(int row, int col);
    //void mm_write(int row, int col, vec4 value);

    let TileSize : u32 = int(gl_WorkGroupSize.x) * 4;


    //shared vec4 mm_Asub[TileSize / 4];
    var<workgroup> mm_Asub : array<f32, TileSize / 4>;

    mm_matMul(dimAOuter : u32, dimInner : u32, dimBOuter : u32) {
      let tileCol = int(gl_LocalInvocationID.x);
      let globalCol = int(gl_GlobalInvocationID.x);
      let globalRow = int(gl_GlobalInvocationID.y);

      let numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      var acc : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        let colA = t * TileSize / 4 + tileCol;
        mm_Asub[tileCol] = mm_readA(globalRow, colA);
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileSize / 4; k++) {
          let rowB = t * TileSize + k * 4;
          vec4 BCached0 = mm_readB(rowB, globalCol);
          vec4 BCached1 = mm_readB(rowB + 1, globalCol);
          vec4 BCached2 = mm_readB(rowB + 2, globalCol);
          vec4 BCached3 = mm_readB(rowB + 3, globalCol);

          vec4 ACached = mm_Asub[k];
          acc += BCached0 * ACached.x;
          acc += BCached1 * ACached.y;
          acc += BCached2 * ACached.z;
          acc += BCached3 * ACached.w;
        }

        workgroupBarrier();
      }

      if (globalRow < dimAOuter && globalCol < dimBOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }
  `;
}

export class MatMulPackedVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];
  useWGSL = true;
  isVec4 = true;
  aShape: [number, number, number];
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;
  vecSize = 4;
  fitA: boolean;
  fitB: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      rowPerThread: number, bias: TensorInfo = null, activation: string = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.workGroupSize = computeWorkGroupSizeForMatMul(
        outputShape[1], aShape[2], outputShape[2]);
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    if (outputShape[1] === 1) {
      rowPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.vecSize, rowPerThread, 1]);

    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = rowPerThread;
    this.aShape = aShape;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    [this.fitA, this.fitB] = this.getShapeFit();

    this.shaderKey = `matMulPackedVec4_${rowPerThread}_${activation}_${
        this.fitA}_${this.fitB}_${this.outputShape[1] > 1}`;
  }

  getShapeFit(): boolean[] {
    const dimInner = this.aShape[2];
    const dimBOuter = this.outputShape[2];
    const bShape = [this.outputShape[0], dimInner, dimBOuter];
    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * this.vecSize;
    const tileInner = tileBOuter;  // Make sure tileInner is divisible by 4.

    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    return [
      tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
      tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
    ];
  }

  getUserCode(): string {
    console.log("MatMulPackedVec4Program ddd");
    const sampleA = this.fitA ?
        `A[batch * batchASize + row * dimInner / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner / 4 + col] :
            vec4(0.0, 0.0, 0.0, 0.0)`;

    const sampleB = this.fitB ?
        `B[batch * batchBSize + row * dimBOuter / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter / 4 + col] :
            vec4(0.0, 0.0, 0.0, 0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `vec4 activation(vec4 a, ivec3 outCoord) {
                  vec4 b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${this.activation}
                }`;
      } else {
        activationSnippet = `
                vec4 activation(vec4 a, ivec3 outCoord) {
                  ${this.activation}
                }`;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}
      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];
      int batch;

      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedVec4Source([this.vecSize, this.workPerThread, 1]) :
            makeMatMulVectorVec4Source()}

      fn mm_readA(row : u32, col : u32) => vec4<f32> {
        let batchASize = aShape[1] * aShape[2] / ${this.vecSize};
        return ${sampleA};
      }

      fn mm_readB(row : u32, col : u32) => vec4<f32> {
        // TODO: This is not covered in unit tests.
        int batchBSize = bShape[1] * bShape[2] / ${this.vecSize};
        return ${sampleB};
      }

      fn mm_write(row : u32, col : u32, value : vec4<f32>){
        if (row < dimAOuter && col * 4 < dimBOuter)
        {
          let outCoord :vec3<i32> = vec3<i32>(batch, row, col * 4);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }

      [[stage(compute), workgroup_size(16, 16, 1)]]
      fn main([[builtin(local_invocation_id)]] local_id : vec3<u32>,
        [[builtin(global_invocation_id)]] global_id  : vec3<u32>) {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    return userCode;
  }
}
