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
  return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${
      workPerThread[0]}; // only support ColPerThread = 4
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
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
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
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
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

const kMatMulVec4Header =`
[[block]] struct Uniforms {
    dimAOuter : u32;
    dimInner : u32;
    dimBOuter : u32;
};
[[block]] struct Matrix {
    numbers: array<vec4<f32>>;
};

[[group(0), binding(0)]] var<storage> firstMatrix : [[access(read)]] Matrix;
[[group(0), binding(1)]] var<storage> secondMatrix : [[access(read)]] Matrix;
[[group(0), binding(2)]] var<storage> resultMatrix : [[access(write)]] Matrix;
[[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

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
        let index : u32 = col + row * uniforms.dimBOuter / 4u;
        resultMatrix.numbers[index] = value;
    }
}

let RowPerThread : u32 = 4u;
let ColPerThread : u32 = 4u;
let TileAOuter : u32 = 64u;
let TileBOuter : u32 = 64u;
let TileInner : u32 = 64u;`;
const kMatMulVec4SharedArray1D = `
var<workgroup> mm_Asub : array<vec4<f32>, 1024>;
var<workgroup> mm_Bsub : array<vec4<f32>, 1024>;`;
const kMatMulVec4SharedArray2D = `
var<workgroup> mm_Asub : array<array<vec4<f32>, 16>, 64>;
var<workgroup> mm_Bsub : array<array<vec4<f32>, 16>, 64>;`;
const kMatMulVec4BodyPart1 = `
[[stage(compute), workgroup_size(16, 16, 1)]]
fn main([[builtin(local_invocation_id)]] local_id : vec3<u32>,
        [[builtin(global_invocation_id)]] global_id  : vec3<u32>) {
    let tileRow : u32 = local_id.y * RowPerThread;
    let tileCol : u32 = local_id.x;

    let globalRow : u32 = global_id.y * RowPerThread;
    let globalCol : u32 = global_id.x;

    let numTiles : u32 = (uniforms.dimInner - 1u) / TileInner + 1u;

    var acc: array<vec4<f32>, 4>;
    var ACached : vec4<f32>;
    var BCached : array<vec4<f32>, 4>;

    // Without this initialization strange values show up in acc.
    // TODO: Remove it once the following bug is fixed.
    // https://bugs.chromium.org/p/tint/issues/detail?id=759
    for (var index : u32 = 0u; index < RowPerThread; index = index + 1u) {
        acc[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var globalColA : u32 = tileCol;
    let RowPerThreadB : u32 = TileInner / 16u;
    let tileRowB : u32 = local_id.y * RowPerThreadB;

    // Loop over shared dimension.
    for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
        // Load one tile of A into local memory.
        for (var innerRow : u32 = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
            let inputRow : u32 = tileRow + innerRow;
            let inputCol : u32 = tileCol;`;
const kMatMulVec4BodyPart2Array1D = `
            let index : u32 = inputRow * TileInner / ColPerThread + inputCol;
            mm_Asub[index] = mm_readA(globalRow + innerRow, globalColA);
        }
        globalColA = globalColA + TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow : u32 = 0u; innerRow < RowPerThreadB; innerRow = innerRow + 1u) {
            let inputRow : u32 = tileRowB + innerRow;
            let inputCol : u32 = tileCol;
            let index : u32 = inputRow * TileBOuter / ColPerThread + inputCol;
            mm_Bsub[index] = mm_readB(t * TileInner + inputRow, globalCol);;
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k : u32 = 0u; k < TileInner / ColPerThread; k = k + 1u) {
            BCached[0] = mm_Bsub[(k * ColPerThread) * (TileBOuter / ColPerThread) + tileCol];
            BCached[1] = mm_Bsub[(k * ColPerThread + 1u) * (TileBOuter / ColPerThread) + tileCol];
            BCached[2] = mm_Bsub[(k * ColPerThread + 2u) * (TileBOuter / ColPerThread) + tileCol];
            BCached[3] = mm_Bsub[(k * ColPerThread + 3u) * (TileBOuter / ColPerThread) + tileCol];

            for (var i : u32 = 0u; i < RowPerThread; i = i + 1u) {
                ACached = mm_Asub[(tileRow + i) * (TileInner / ColPerThread) + k];`;
const kMatMulVec4BodyPart2Array2D = `
            mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA);
        }
        globalColA = globalColA + TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow : u32 = 0u; innerRow < RowPerThreadB; innerRow = innerRow + 1u) {
            let inputRow : u32 = tileRowB + innerRow;
            let inputCol : u32 = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol);;
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k : u32 = 0u; k < TileInner / ColPerThread; k = k + 1u) {
            BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
            BCached[1] = mm_Bsub[k * ColPerThread + 1u][tileCol];
            BCached[2] = mm_Bsub[k * ColPerThread + 2u][tileCol];
            BCached[3] = mm_Bsub[k * ColPerThread + 3u][tileCol];

            for (var i : u32 = 0u; i < RowPerThread; i = i + 1u) {
                ACached = mm_Asub[tileRow + i][k];`;
const kMatMulVec4BodyPart3 = `
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
                acc[i] = BCached[2] * ACached.z + acc[i];
                acc[i] = BCached[3] * ACached.w + acc[i];
            }
        }

        workgroupBarrier();
    }

    for (var innerRow : u32 = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
        mm_write(globalRow + innerRow,
                 globalCol,
                 acc[innerRow]);
    }
}`;

    export function makeMatMulPackedVec4WGSLSource(workPerThread: number[]): string {
      console.log("makeMatMulPackedVec4WGSLSource");

    /*
const kMatMulVec4OneDimensionalSharedArray =
    kMatMulVec4Header + kMatMulVec4SharedArray1D + kMatMulVec4BodyPart1 +
    kMatMulVec4BodyPart2Array1D + kMatMulVec4BodyPart3;
    */
    console.log(kMatMulVec4SharedArray1D + kMatMulVec4BodyPart2Array1D);
const kMatMulVec4TwoDimensionalSharedArray =
    kMatMulVec4Header + kMatMulVec4SharedArray2D + kMatMulVec4BodyPart1 +
    kMatMulVec4BodyPart2Array2D + kMatMulVec4BodyPart3;
    return kMatMulVec4TwoDimensionalSharedArray;
}

export function makeMatMulVectorVec4WGSLSource(workPerThread: number[]): string {
  console.log("makeMatMulPackedVec4WGSLSource");


const kMatMulVec4OneDimensionalSharedArray =
kMatMulVec4Header + kMatMulVec4SharedArray1D + kMatMulVec4BodyPart1 +
kMatMulVec4BodyPart2Array1D + kMatMulVec4BodyPart3;

//console.log(kMatMulVec4SharedArray1D + kMatMulVec4BodyPart2Array1D);

return kMatMulVec4OneDimensionalSharedArray;
}

export function makeMatMulVectorVec4Source(): string {
  return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int TileSize = int(gl_WorkGroupSize.x) * 4;

    shared vec4 mm_Asub[TileSize / 4];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileCol = int(gl_LocalInvocationID.x);
      int globalCol = int(gl_GlobalInvocationID.x);
      int globalRow = int(gl_GlobalInvocationID.y);

      int numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      vec4 acc = vec4(0.0);

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        int colA = t * TileSize / 4 + tileCol;
        mm_Asub[tileCol] = mm_readA(globalRow, colA);
        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileSize / 4; k++) {
          int rowB = t * TileSize + k * 4;
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

        barrier();
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
    /*
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
    */

    /*
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
    */

    /*
    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';
    */

    const userCode = `
      //{activationSnippet}
      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];
      int batch;

      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedVec4WGSLSource([this.vecSize, this.workPerThread, 1]) :
            makeMatMulVectorVec4Source()}
    `;
    return userCode;
  }
}
