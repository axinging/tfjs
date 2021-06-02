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

import {TensorInfo, util} from '@tensorflow/tfjs-core';

import {computeDispatch, computeWorkGroupSizeForMatMul, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedSource(workPerThread: number[]): string {
  console.log("makeMatMulPackedSource");
  return `
    //float mm_readA(int row, int col);
    //float mm_readB(int row, int col);
    //void mm_write(int row, int col, float value);
    //void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);
    // float NAN; ivec3 aShape; ivec3 bShape; ivec3 outShape; ivec2 outShapeStrides; 


    let RowPerThread : u32 = ${workPerThread[1]};
    let ColPerThread : u32 = ${workPerThread[0]};
    //let TileAOuter : u32 = int(gl_WorkGroupSize.y) * RowPerThread;
    //let TileBOuter : u32 = int(gl_WorkGroupSize.x) * ColPerThread;
    //let TileInner : u32 = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

    //var<workgroup> mm_Asub : array<array<f32, TileInner>, TileAOuter>;
    //var<workgroup> mm_Bsub : array<array<f32, TileBOuter>, TileInner>;

    var<workgroup> mm_Asub : array<array<vec4<f32>, 16>, 64>;
    var<workgroup> mm_Bsub : array<array<vec4<f32>, 16>, 64>;
    // shared float mm_Asub[TileAOuter][TileInner];
    // shared float mm_Bsub[TileInner][TileBOuter];

    fn mm_matMul(dimAOuter : u32, dimInner : u32, dimBOuter : u32, localId: vec3<u32>, globalId: vec3<u32>, ) {
      let tileRow = int(localId.y) * RowPerThread;
      let tileCol = int(localId.x) * ColPerThread;

      let globalRow = int(globalId.y) * RowPerThread;
      let globalCol = int(globalId.x) * ColPerThread;

      let numTiles = (dimInner - 1) / TileInner + 1;

      let acc[RowPerThread][ColPerThread] : f32;
      let ACached : f32;
      let BCached[ColPerThread] : f32;

      let TileAOuter : u32 = int(gl_WorkGroupSize.y) * RowPerThread;
      let TileBOuter : u32 = int(gl_WorkGroupSize.x) * ColPerThread;
      let TileInner : u32 = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (var innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      let ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      let tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      let RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      let tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (var innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (var innerCol = 0; innerCol < ColPerThread; innerCol++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol + innerCol);;
          }
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileInner; k++) {
          for (var inner = 0; inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (var innerRow = 0; innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (var innerCol = 0; innerCol < ColPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        workgroupBarrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

          if ((globalCol + innerCol) < dimBOuter &&
              (globalRow + innerRow) < dimAOuter) {
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol]);
          }
        }
      }
    }
  `;
}

export function makeMatMulVectorSource(): string {
  console.log("makeMatMulVectorSource");
  return `
    //float mm_readA(int row, int col);
    //float mm_readB(int row, int col);
    //void mm_write(int row, int col, float value);
    //void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);
/*
    [[block]] struct Uniforms {
      NAN : f32;
      aShape : vec3<u32>;
      bShape : vec3<u32>;
      outShape : vec3<u32>;
      outShapeStrides : vec2<u32>;
  };
  [[block]] struct Matrix {
      numbers: array<f32>;
  };
    [[group(0), binding(0)]] var<storage> A : [[access(read)]] Matrix;
    [[group(0), binding(1)]] var<storage> B : [[access(read)]] Matrix;
    [[group(0), binding(2)]] var<storage> resultMatrix : [[access(write)]] Matrix;
    [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;
    */

    let TileSize = int(gl_WorkGroupSize.x) * 4;

    var<workgroup> mm_Asub : array<f32, TileSize>;
    // shared vec4 mm_Asub[TileSize / 4];

    fn mm_matMul(dimAOuter : u32, int dimInner : u32, int dimBOuter : u32) {
      let tileCol = int(gl_LocalInvocationID.x);
      let globalCol = int(gl_GlobalInvocationID.x);
      let globalRow = int(gl_GlobalInvocationID.y);

      let numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      let acc : f32 = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        let colA = t * TileSize + tileCol * 4;
        mm_Asub[tileCol] = vec4(mm_readA(globalRow, colA),
                                mm_readA(globalRow, colA + 1),
                                mm_readA(globalRow, colA + 2),
                                mm_readA(globalRow, colA + 3));
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileSize / 4; k++) {
          let rowB = t * TileSize + k * 4;
          let BCached : vec4<f32> = vec4<f32>(mm_readB(rowB, globalCol),
                              mm_readB(rowB + 1, globalCol),
                              mm_readB(rowB + 2, globalCol),
                              mm_readB(rowB + 3, globalCol));

          let ACached : vec4<f32> = mm_Asub[k];
          acc += dot(ACached, BCached);
        }

        workgroupBarrier();
      }

      if (globalRow < dimAOuter && globalCol < dimBOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }
  `;
}

export class MatMulPackedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];
  aShape: [number, number, number];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;
  fitA: boolean;
  fitB: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, transposeA = false, transposeB = false,
      bias: TensorInfo = null, activation: string = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.workGroupSize =
        computeWorkGroupSizeForMatMul(outputShape[1], dimInner, outputShape[2]);
    if (outputShape[1] === 1 || outputShape[2] === 1) {
      workPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);
    // If dispaching number is one, it means only one work group is running.
    // For modern GPUs, it supports multiple work groups running in parallel.
    // So there may be some idle hardware threads.
    // In this case, we prefer to reduce the work per thread and improve the
    // thread utilization
    if (util.arraysEqual(this.dispatch, [1, 1, 1])) {
      workPerThread = 1;
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [workPerThread, workPerThread, 1]);
    }
    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = workPerThread;
    this.aShape = aShape;
    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    const dimBOuter = this.outputShape[2];
    const bShape = this.transposeB ?
        [this.outputShape[0], dimBOuter, dimInner] :
        [this.outputShape[0], dimInner, dimBOuter];

    [this.fitA, this.fitB] = this.getShapeFit(bShape);
    this.shaderKey =
        `matMulPacked_${this.workPerThread}_${transposeA}_${transposeB}_${
            activation}_${this.fitA}_${this.fitB}_${this.outputShape[1] > 1}`;
  }

  getShapeFit(bShape: number[]): boolean[] {
    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * this.workPerThread;
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () => `tileInner must be multiple of workgroupsize.x ` +
            `and workgroupsize.y`);
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];

    return [
      tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
      tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
    ];
  }

  getUserCode(): string {
    let sampleA;
    console.log("MatMulPackedProgram ddd");

    if (this.transposeA === false) {
      sampleA = this.fitA ?
          `A.numbers[batch * batchASize + row * dimInner + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
          A.numbers[batch * batchASize + row * dimInner + col] : 0`;
    } else {
      sampleA = this.fitA ?
          `A.numbers[batch * batchASize + col * dimAOuter + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
          A.numbers[batch* batchASize + col * dimAOuter + row] : 0`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB = this.fitB ?
          `B.numbers[batch * batchBSize + row * dimBOuter + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B.numbers[batch * batchBSize + row * dimBOuter + col] : 0`;
    } else {
      sampleB = this.fitB ?
          `B.numbers[batch * batchBSize + col * dimInner + row]` :
          `coordsInBounds(vec2<u32>(row, col), vec2<u32>(dimInner, dimBOuter)) ?
            B.numbers[batch * batchBSize + col * dimInner + row] : 0`;
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec3 outCoord) {
              float b = getPreluActivationWeightsAtOutCoords(outCoord);
              ${this.activation}
            }`;
      } else {
        activationSnippet = `
              float activation(float a, ivec3 outCoord) {
                ${this.activation}
              }
            `;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
    // Below test for packed
    [[block]] struct Uniforms {
      NAN : f32;
      aShape : vec3<u32>;
      bShape : vec3<u32>;
      outShape : vec3<u32>;
      outShapeStrides : vec2<u32>;
  };
  [[block]] struct Matrix {
      numbers: array<f32>;
  };
    [[group(0), binding(0)]] var<storage> A : [[access(read)]] Matrix;
    [[group(0), binding(1)]] var<storage> B : [[access(read)]] Matrix;
    [[group(0), binding(2)]] var<storage> result : [[access(write)]] Matrix;
    [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

      ${activationSnippet}

      //let dimAOuter : u32 = ${this.transposeA === true ? `uniforms.aShape[2]` : `uniforms.aShape[1]`};
      //let dimInner : u32 = ${this.transposeA === true ? `uniforms.aShape[1]` : `uniforms.aShape[2]`};
      //let dimBOuter : u32 = ${this.transposeB === true ? `uniforms.bShape[1]` : `uniforms.bShape[2]`};

      var batch : u32;

      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedSource(
                [this.workPerThread, this.workPerThread, 1]) :
            makeMatMulVectorSource()}
      fn mm_readA(row: u32, col : u32) -> f32 {
        let dimAOuter : u32 = ${this.transposeA === true ? `uniforms.aShape[2]` : `uniforms.aShape[1]`};
        let dimInner : u32 = ${this.transposeA === true ? `uniforms.aShape[1]` : `uniforms.aShape[2]`};
        let dimBOuter : u32 = ${this.transposeB === true ? `uniforms.bShape[1]` : `uniforms.bShape[2]`};
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        return ${sampleA};
      }
      fn mm_readB(row : u32, col : u32) -> f32 {
        int batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        return ${sampleB};
      }
      fn mm_write(row : u32, col : u32, value : f32) {
        ivec3 outCoord = ivec3(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(batch, row, col, value);
      }
      fn main([[builtin(local_invocation_id)]] local_id : vec3<u32>,
      [[builtin(global_invocation_id)]] global_id  : vec3<u32>) {
        batch = u32(global_id.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    return userCode;
  }
}
