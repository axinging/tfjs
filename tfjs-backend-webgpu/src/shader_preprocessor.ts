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

import {backend_util, DataType, util} from '@tensorflow/tfjs-core';

import {symbolicallyComputeStrides} from './shader_util';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'int';
  } else if (rank === 2) {
    return 'ivec2';
  } else if (rank === 3) {
    return 'ivec3';
  } else if (rank === 4) {
    return 'ivec4';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

export function getShapeCoords(dataShape: number[]): string {
  const rank = dataShape.length;
  if (rank <= 1) {
    return `int(${dataShape[0]})`;
  } else if (rank === 2) {
    return `ivec2(${dataShape[0]}, ${dataShape[1]})`;
  } else if (rank === 3) {
    return `ivec3(${dataShape[0]}, ${dataShape[1]}, ${dataShape[2]})`;
  } else if (rank === 4) {
    return `ivec4(${dataShape[0]}, ${dataShape[1]}, ${dataShape[2]}, ${
        dataShape[3]})`;
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

type GLSLDataType = 'float'|'int';
function mapToGlslTypes(type: DataType): GLSLDataType|DataType {
  if (type === 'float32') {
    return 'float';
  }
  if (type === 'int32') {
    return 'int';
  }
  return type;
}

interface ProgramParams {
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  workGroupSize?: [number, number, number];
  variableNames: string[];
  variableTextureNames?: string[];
  uniforms?: string;
  userCode: string;
}

export type ShapeInfo = {
  logicalShape: number[],
  texShape: [number, number],
  isUniform: boolean,
  isPacked: boolean,
  flatOffset: number
};

export interface InputInfo {
  dtype: DataType;
  shape: number[];
  name: string;
  shapeInfo: ShapeInfo;
  useTexture: boolean;
}

export function makeShader(
    inputInfo: InputInfo[], outputData: {dtype: DataType, shape: number[]},
    program: ProgramParams): string {
  const prefixSnippets: string[] = [];

  if (program.workGroupSize != null) {
    prefixSnippets.push(`
      layout (local_size_x = ${program.workGroupSize[0]},
              local_size_y = ${program.workGroupSize[1]},
              local_size_z = ${program.workGroupSize[2]}) in;
    `);
  }

  // Output buffer.
  /*
  prefixSnippets.push(`
    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      ${mapToGlslTypes(outputData.dtype)} result[];
    };
  `);
  */

  prefixSnippets.push(`
    layout(set = 0, binding = 0, r32f) uniform writeonly image2D result;
  `);

  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, set = 0, binding = ${1 + i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputInfo[i].dtype)} ${x}[];
      };
    `);
  });

  if (program.variableTextureNames) {
    program.variableTextureNames.forEach((x, i) => {
      prefixSnippets.push(`
        layout(set = 0, binding = ${
          1 + program.variableNames.length +
          i}, r32f) uniform readonly image2D ${x};
      `);
    });
  }

  let uniformDeclaration = '';

  if (program.uniforms) {
    uniformDeclaration += program.uniforms;
    const bindingOffset = program.variableTextureNames ?
        1 + program.variableNames.length + program.variableTextureNames.length :
        1 + program.variableNames.length;
    prefixSnippets.push(`
    layout(std140, set = 0, binding = ${bindingOffset}) uniform Uniforms {
      ${uniformDeclaration}
    };
  `);
  }
  const [getOutputCoords, dispatchLayoutRank] =
      generateGetOutputCoords(outputData.shape, program.dispatchLayout);
  const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    getOutputCoords, getCoords,
    getSetOutputSnippet(outputData.shape, outputData.dtype)
  ];

  if (dispatchLayoutRank === outputData.shape.length) {
    // Input sampling snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSamplingSnippet =
        inputInfo.map(x => getInputSamplingSnippet(x, outputData.shape))
            .join('\n');
    sources.push(inputSamplingSnippet);
  }

  sources.push(program.userCode);
  const source = sources.join('\n');
  return source;
}

const SHADER_PREFIX = `#version 450

  int idiv(int a, int b, float sign) {
    int res = a / b;
    int mod = a % b;
    if (sign < 0. && mod != 0) {
      res -= 1;
    }
    return res;
  }

  // Checks whether coordinates lie within the bounds of the shape.
  bool coordsInBounds(ivec4 coord, ivec4 shape) {
    return all(greaterThanEqual(coord, ivec4(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec3 coord, ivec3 shape) {
    return all(greaterThanEqual(coord, ivec3(0))) &&
        all(lessThan(coord, shape));
  }

  bool coordsInBounds(ivec2 coord, ivec2 shape) {
    return all(greaterThanEqual(coord, ivec2(0))) && 
        all(lessThan(coord, shape));
  }

  ivec2 uvFromFlat(int texNumR, int texNumC, int index) {
    int texR = index / texNumC;
    int texC = index - texR * texNumC;
    return ivec2(texC, texR);
  }
`;

const SAMPLING_SNIPPETS = `
  int getFlatIndex(int coord, int shape) {
    return coord;
  }

  int getFlatIndex(ivec2 coords, ivec2 shape) {
    return int(dot(coords, ivec2(shape.y, 1.)));
  }

  int getFlatIndex(ivec3 coords, ivec3 shape) {
    return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
  }

  int getFlatIndex(ivec4 coords, ivec4 shape) {
    return int(dot(coords, ivec4(
      shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
  }
`;

function getSetOutputSnippet(
    outShape: number[], outBufferType: DataType): string {
  const outRank = outShape.length;
  const glslType = mapToGlslTypes(outBufferType);
  const useTexture = true;
  let snippet;
  if (useTexture) {
    snippet = `void setOutput(int flatIndex, float value) {
    }
    void setOutput(int flatIndex, int value) {
    }`;
  } else {
    snippet = `void setOutput(int flatIndex, float value) {
      result[flatIndex] = ${
        glslType === 'int' ? 'int(value)' :
                             (glslType === 'bool' ? 'bool(value)' : 'value')};
    }
    void setOutput(int flatIndex, int value) {
      result[flatIndex] = ${
        glslType === 'float' ?
            'float(value)' :
            (glslType === 'bool' ? 'bool(value)' : 'value')};
    }`;
  }


  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
        getShapeCoords(outShape)});
        setOutput(flatIndex, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
        getShapeCoords(outShape)});
        setOutput(flatIndex, value);
      }
    `;
  }

  return snippet;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShape: number[]): string {
  let res = getSamplerFromInInfo2(inInfo);
  // TODO(texture): this is for mm_read
  // res += getSamplerFromInInfo2(inInfo);

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getSamplerAtOutputCoords(inInfo, outShape);
  }

  return res;
}

function getSamplerFromInInfo(inInfo: InputInfo): string {
  const texName = inInfo.name;
  const rank = inInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
  const inputs = dims.map(d => `int ${d}`).join(', ');

  if (rank < 1) {
    return `
      float ${funcName}() {
        return ${texName}[0];
      }
    `;
  }

  if (inInfo.useTexture) {
    return `
    float ${funcName}(${inputs}) {
      int index = getFlatIndex(${type}(${dims.join(',')}),
      ${getShapeCoords(inInfo.shape)});
      return imageLoad(${texName}, ivec2(index, 0)).r;
    }
  `;
  }

  return `
    float ${funcName}(${inputs}) {
      int index = getFlatIndex(${type}(${dims.join(',')}),
      ${getShapeCoords(inInfo.shape)});
      return float(${texName}[index]);
    }
  `;
}

function getSamplerAtOutputCoords(
    inInfo: InputInfo, outShape: number[]): string {
  const texName = inInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);

  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const inRank = inInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
  const rankDiff = outRank - inRank;

  let coordsSnippet = '';

  if (inRank === 0) {
    return `
      float ${funcName}() {
        return get${texFuncSnippet}();
      }

      float ${funcName}(${type} coords) {
        return get${texFuncSnippet}();
      }
    `;
  } else {
    if (outRank < 2 && broadcastDims.length >= 1) {
      coordsSnippet = 'coords = 0;';
    } else {
      coordsSnippet =
          broadcastDims.map(d => `coords[${d + rankDiff}] = 0;`).join('\n');
    }
  }

  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    if (outRank > 1) {
      const coordsType = getCoordsDataType(inRank);
      const coordsValues =
          inInfo.shape.map((s, i) => `coords[${i + rankDiff}]`).join(', ');
      unpackedCoordsSnippet = `${coordsType}(${coordsValues})`;
    } else {
      unpackedCoordsSnippet = 'coords';
    }
  }

  if (inInfo.useTexture) {
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      int index = getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)});
      return imageLoad(${texName}, ivec2(index, 0)).r;
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      int index = getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)});
      return imageLoad(${texName}, ivec2(index, 0)).r;
    }
  `;
  }

  return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      getShapeCoords(inInfo.shape)})]);
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      getShapeCoords(inInfo.shape)})]);
    }
  `;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
function generateGetOutputCoords(
    outShape: number[],
    dispatchLayout: {x: number[], y?: number[], z?: number[]}):
    [string, number] {
  const {x, y = [], z = []} = dispatchLayout;
  let gatherDimensionsStr = '';
  const dims = [x, y, z];

  let rank = 0;

  for (let i = 0; i < dims.length; i++) {
    const arr = dims[i];

    if (arr.length === 0) {
      continue;
    }

    rank += arr.length;

    if (arr.length === 1) {
      gatherDimensionsStr += `int d${arr[0]} =
        int(gl_GlobalInvocationID[${i}]);`;
    } else {
      const strides =
          symbolicallyComputeStrides(arr, `${getShapeCoords(outShape)}`);
      gatherDimensionsStr += `int index${i} =
        int(gl_GlobalInvocationID[${i}]);`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr += `int d${arr[j]} = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `int d${arr[j + 1]} = ` +
              `index${i} - d${arr[j]} * ${strides[j]};`;
        } else {
          gatherDimensionsStr += `index${i} -= d${arr[j]} * ${strides[j]};`;
        }
      }
    }
  }

  const dimensions = [];
  for (let i = 0; i < rank; i++) {
    dimensions.push(`d${i}`);
  }

  const dtype = getCoordsDataType(rank);
  let snippet = `${dtype} getOutputCoords() {
    ${gatherDimensionsStr}
  `;
  if (dimensions.length === 0) {
    snippet += `return ${dtype}(0);}`;
  } else {
    snippet += `return ${dtype}(${dimensions.join(',')});}`;
  }

  return [snippet, rank];
}

/**
 * Derives logical coordinates from a flat index. Performs integer division with
 * each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function generateGetCoordsFromFlatIndex(shape: number[]): string {
  const rank = shape.length;

  if (rank <= 1) {
    return `int getCoordsFromFlatIndex(int index) {return index; }`;
  }

  const strides = util.computeStrides(shape);
  const dtype = getCoordsDataType(rank);
  const coords: string[] = [];
  for (let i = 0; i < rank; i++) {
    coords.push(`d${i}`);
  }

  const snippet =
      strides
          .map((stride, i) => {
            const line1 = `int ${coords[i]} = index / ${stride}`;
            const line2 = i === strides.length - 1 ?
                `int ${coords[i + 1]} = index - ${coords[i]} * ${stride}` :
                `index -= ${coords[i]} * ${stride}`;
            return `${line1}; ${line2};`;
          })
          .join('');

  return `
    ${dtype} getCoordsFromFlatIndex(int index) {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}

export function getSamplerFromInInfo2(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    /*
    case 0:
      return getSamplerScalar(inInfo);
    case 1:
      return getSampler1D(inInfo);
    */
    case 2:
      return getSampler2D(inInfo);
    case 3:
      return getSampler3D(inInfo);
    case 4:
      return getSampler4D(inInfo);
    /*
    case 5:
      return getSampler5D(inInfo);
    case 6:
      return getSampler6D(inInfo);
    */
    default:
      throw new Error(
          `${shape.length}-D input sampling` +
          ` is not yet supported`);
  }
}

export function getSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      float ${funcName}(int row, int col) {
        return imageLoad(${texName}, ivec2(row,col)).r; 
      }
    `;
  }

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  // TODO(texture): this requires get1D.
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col'];
    return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  return `
    float ${funcName}(int row, int col) {
      imageLoad(W, ivec2(row,col)).r; 
    }
  `;
}

export function getSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  console.log('3D texName ' + texName + ', ' + shape);

  const {newShape, keptDims} = util.squeezeShape(shape);
  const squeezedShape = newShape;
  if (squeezedShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['row', 'col', 'depth'];
    return `
            ${getSamplerFromInInfo2(newInputInfo)}
            float ${funcName}(int row, int col, int depth) {
              return ${funcName}(${getSqueezedParams(params, keptDims)});
            }
          `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];

  console.log(' texNumR = ' + texNumR);
  console.log(' stride0 = ' + stride0);
  console.log(' texNumC = ' + texNumC);
  console.log(' stride1 = ' + stride1);
  if (texNumC === stride0)
    // texC is used directly as physical (no risk of float16 overflow).
    return `
        float ${funcName}(int row, int col, int depth) {
          int texR = float(row);
          int texC = dot(vec2(col, depth), vec2(${stride1}, 1));
          return imageLoad(${texName}, ivec2(texR,texC)).r;
        }
      `;

  // case [logShape[0] * logShape[1], logShape[2]];
  if (texNumC === stride1) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col, int depth) {
      int texR = int(dot(vec2(row, col), vec2(${shape[1]}, 1)));
      int texC = int(depth);
      // int texC = col%2;
      return imageLoad(${texName}, ivec2(texC,texR)).r;
    }
  `;
  }

  return `
    float ${funcName}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth;
      ivec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return imageLoad(${texName}, ivec2(uv.y,uv.x)).r;
    } `;
}

export function getSampler4D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride2 = shape[3];
  const stride1 = shape[2] * stride2;
  const stride0 = shape[1] * stride1;

  const {newShape, keptDims} = util.squeezeShape(shape);

  if (newShape.length < shape.length) {
    const newInputInfo = squeezeInputInfo(inputInfo, newShape);
    const params = ['row', 'col', 'depth', 'depth2'];
    return `
      ${
        getSamplerFromInInfo2(
            newInputInfo)} // this may call into getSampler2D or getSampler3D;
      float ${funcName}(int row, int col, int depth, int depth2) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
  }

  const texShape = inputInfo.shapeInfo.texShape;
  const texNumR = texShape[0];
  const texNumC = texShape[1];

  if (texNumC === stride2)
    // texR is used directly as physical (no risk of float16 overflow).
    return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int texR = int(dot(vec3(row, col, depth),
                         vec3(${shape[1] * shape[2]}, ${shape[2]}, 1)));
        int texC = int(depth2);
        return imageLoad(${texName}, ivec2(texR,texC)).r;
      }
    `;

  return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} +
          depth * ${stride2} + depth2;
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return imageLoad(${texName}, ivec2(uv.y,uv.x)).r;
    }
  `;
}

/** Returns a new input info (a copy) that has a squeezed logical shape. */
export function squeezeInputInfo(
    inInfo: InputInfo, squeezedShape: number[]): InputInfo {
  // Deep copy.
  const newInputInfo: InputInfo = JSON.parse(JSON.stringify(inInfo));
  newInputInfo.shapeInfo.logicalShape = squeezedShape;
  return newInputInfo;
}

export function getSqueezedParams(
    params: string[], keptDims: number[]): string {
  return keptDims.map(d => params[d]).join(', ');
}
