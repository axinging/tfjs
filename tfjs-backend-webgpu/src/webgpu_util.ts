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
import {DataType, env, util} from '@tensorflow/tfjs-core';

const arrayProduct = (arr: number[]) => {
  let product = 1;
  for (let i = 0; i < arr.length; i++) {
    product *= arr[i];
  }
  return product;
};

export function tilesFitEvenlyIntoShape(
    tileSize: number[], shape: number[]): boolean {
  if (tileSize.length !== shape.length) {
    throw new Error(
        `Cannot compute whether rank ${tileSize.length}` +
        ` tiles fit evenly into rank ${shape.length} shape` +
        ` - ranks must match.`);
  }
  return shape.every(
      (dim: number, dimIdx: number) => dim % tileSize[dimIdx] === 0);
}

// Computes dispatch geometry based on layout of output dimensions and
// workGroupSize.
export function computeDispatch(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    workGroupSize: [number, number, number] = [1, 1, 1],
    elementsPerThread: [number, number, number] =
        [1, 1, 1]): [number, number, number] {
  return [
    Math.ceil(
        arrayProduct(layout.x.map(d => outputShape[d])) /
        (workGroupSize[0] * elementsPerThread[0])),
    layout.y ? Math.ceil(
                   arrayProduct(layout.y.map(d => outputShape[d])) /
                   (workGroupSize[1] * elementsPerThread[1])) :
               1,
    layout.z ? Math.ceil(
                   arrayProduct(layout.z.map(d => outputShape[d])) /
                   (workGroupSize[2] * elementsPerThread[2])) :
               1
  ];
}

export function computeWorkGroupSizeForConv2d(
    layout: {x: number[], y?: number[], z?: number[]},
    outputShape: number[]): [number, number, number] {
  const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
  const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // These are experimental values. Usually, we need to adjust the work group
  // size based on the output shape. For example, when one dimension is smaller
  // than 4, it will be wasteful if we assign a larger size for this dimension,
  // which results lots of threads doing useless work and reduces parallelism
  // of hardware threads. But it is always a balance between work group size
  // and shared memory. If one dimension is too small, such as 1, shared memory
  // will won't be fully utilized.
  if (dim0 <= 4) {
    return [4, 16, 1];
  }
  if (dim1 <= 4) {
    return [16, 4, 1];
  }

  return [16, 16, 1];
}

export function computeWorkPerThreadForConv2d(
    layout: {x: number[], y?: number[], z?: number[]},
    outputShape: number[]): [number, number, number] {
  const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
  const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // The following conditions correspond to the values set in
  // computeWorkGroupSizeForConv2d.
  if (dim0 <= 4) {
    return [1, 2, 1];
  }
  if (dim1 <= 4) {
    return [2, 1, 1];
  }

  if ((dim1 > dim0) && (dim1 / dim0 >= 2)) {
    return [2, 4, 1];
  }
  if ((dim0 > dim1) && (dim0 / dim1 >= 2)) {
    return [4, 2, 1];
  }

  return [2, 2, 1];
}

export function flatDispatchLayout(shape: number[]) {
  return {x: shape.map((d, i) => i)};
}

export function GPUBytesPerElement(dtype: DataType): number {
  if (dtype === 'float32' || dtype === 'int32' || dtype === 'bool') {
    return 4;
  } else if (dtype === 'complex64') {
    return 8;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function ArrayBufferToTypedArray(data: ArrayBuffer, dtype: DataType) {
  if (dtype === 'float32') {
    return new Float32Array(data);
  } else if (dtype === 'int32') {
    return new Int32Array(data);
  } else if (dtype === 'bool') {
    const dataAsInt32Array = new Int32Array(data);
    const boolData = new ArrayBuffer(dataAsInt32Array.length);
    const dataAsTypedArray = new Uint8Array(boolData);
    for (let i = 0; i < dataAsInt32Array.length; i++) {
      dataAsTypedArray[i] = dataAsInt32Array[i];
    }
    return dataAsTypedArray;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function getBatchDim(shape: number[], dimsToSkip = 2): number {
  return util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
}

export function getRowsCols(shape: number[]): [number, number] {
  if (shape.length === 0) {
    throw Error('Cannot get rows and columns of an empty shape array.');
  }

  return [
    shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
  ];
}

export function getShapeAs3D(shape: number[]): [number, number, number] {
  let shapeAs3D: [number, number, number] = [1, 1, 1];
  const isScalar = shape.length === 0 || (shape.length === 1 && shape[0] === 1);
  if (!isScalar) {
    shapeAs3D =
        [getBatchDim(shape), ...getRowsCols(shape)] as [number, number, number];
  }
  return shapeAs3D;
}


export function getTextureShapeFromLogicalShape(
    logShape: number[], isPacked = false): [number, number] {
  let maxTexSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
  if (isPacked) {
    maxTexSize = maxTexSize * 2;

    // This logic ensures we accurately count the number of packed texels needed
    // to accommodate the tensor. We can only pack values in the same texel if
    // they are from adjacent pairs of rows/cols within the same batch. So if a
    // tensor has 3 rows, we pretend it has 4 rows in order to account for the
    // fact that the texels containing the third row are half empty.
    logShape = logShape.map(
        (d, i) => i >= logShape.length - 2 ?
            util.nearestLargerEven(logShape[i]) :
            logShape[i]);

    // Packed texture height is at least 2 (the channel height of a single
    // texel).
    if (logShape.length === 1) {
      logShape = [2, logShape[0]];
    }
  }

  // If logical shape is 2, we don't squeeze, since we want to match physical.
  if (logShape.length !== 2) {
    const squeezeResult = util.squeezeShape(logShape);
    logShape = squeezeResult.newShape;
  }

  let size = util.sizeFromShape(logShape);
  if (logShape.length <= 1 && size <= maxTexSize) {
    return [1, size];
  } else if (
      logShape.length === 2 && logShape[0] <= maxTexSize &&
      logShape[1] <= maxTexSize) {
    return logShape as [number, number];
  } else if (
      logShape.length === 3 && logShape[0] * logShape[1] <= maxTexSize &&
      logShape[2] <= maxTexSize) {
    return [logShape[0] * logShape[1], logShape[2]];
  } else if (
      logShape.length === 3 && logShape[0] <= maxTexSize &&
      logShape[1] * logShape[2] <= maxTexSize) {
    return [logShape[0], logShape[1] * logShape[2]];
  } else if (
      logShape.length === 4 &&
      logShape[0] * logShape[1] * logShape[2] <= maxTexSize &&
      logShape[3] <= maxTexSize) {
    return [logShape[0] * logShape[1] * logShape[2], logShape[3]];
  } else if (
      logShape.length === 4 && logShape[0] <= maxTexSize &&
      logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
    return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
  } else {
    if (isPacked) {
      // For packed textures size equals the number of channels required to
      // accommodate the texture data. However in order to squarify such that
      // inner dimensions stay even, we rewrite size to equal the number of
      // texels. Then in the return statement we rehydrate the squarified
      // dimensions to channel units.

      const batchDim = getBatchDim(logShape);
      let rows = 2, cols = 2;
      if (logShape.length) {
        [rows, cols] = getRowsCols(logShape);
      }
      size = batchDim * (rows / 2) * (cols / 2);
      return util.sizeToSquarishShape(size).map(d => d * 2) as [number, number];
    }
    return util.sizeToSquarishShape(size);
  }
}
