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
// import {getGlslDifferences} from './glsl_version';
import {symbolicallyComputeStrides} from './shader_util';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'u32';
  } else if (rank === 2) {
    return 'vec2<u32>';
  } else if (rank === 3) {
    return 'vec3<u32>';
  } else if (rank === 4) {
    return 'vec4<u32>';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

type GLSLDataType = 'f32'|'i32'|'vec4<f32>'|'vec4<i32>'|'vec4<bool>';
function mapToGlslTypes(type: DataType, isVec4: boolean): GLSLDataType|
    DataType {
  if (type === 'float32') {
    return isVec4 ? 'vec4<f32>' : 'f32';
  } else if (type === 'int32') {
    return isVec4 ? 'vec4<i32>' : 'i32';
  } else if (type === 'bool') {
    return isVec4 ? 'vec4<bool>' : 'bool';
  }

  return type;
}

interface ProgramParams {
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  workGroupSize?: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  isVec4?: boolean;
  size?: number;
  getUserCode: () => string;
  getUserHeaderCode?: () => string;
}

export interface InputInfo {
  dtype: DataType;
  shape: number[];
  name: string;
}

export function makeShader(
    inputInfo: InputInfo[], outputData: {dtype: DataType, shape: number[]},
    program: ProgramParams, isFromPixel = false): string {
  const outputBufferStr =
      `    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      ${mapToGlslTypes(outputData.dtype, program.isVec4)} result[];
    };`;
  if (isFromPixel === true) {
    const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
    return [
      SHADER_PREFIX, outputBufferStr, program.getUserCode(), getCoords
    ].join('\n');
  }
  /*
  const prefixSnippets: string[] = [];


  if (program.workGroupSize != null) {
    prefixSnippets.push(`
      layout (local_size_x = ${program.workGroupSize[0]},
              local_size_y = ${program.workGroupSize[1]},
              local_size_z = ${program.workGroupSize[2]}) in;
    `);
  }

  // Output buffer.
  prefixSnippets.push(`
    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      ${mapToGlslTypes(outputData.dtype, program.isVec4)} result[];
    };
  `);

  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, set = 0, binding = ${1 + i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputInfo[i].dtype, program.isVec4)} ${x}[];
      };
    `);
  });

  let uniformDeclaration = 'float NAN; ';
  program.variableNames.forEach((x, i) => {
    uniformDeclaration += `${getCoordsDataType(inputInfo[i].shape.length)} ${
        x.charAt(0).toLowerCase() + x.slice(1)}Shape; `;
  });
  uniformDeclaration +=
      `${getCoordsDataType(outputData.shape.length)} outShape; `;
  const stridesLength = outputData.shape.length - 1;
  uniformDeclaration += `${getCoordsDataType(stridesLength)} outShapeStrides; `;

  if (program.size != null) {
    uniformDeclaration += 'int size; ';
  }

  if (program.uniforms) {
    uniformDeclaration += program.uniforms;
  }


  if (uniformDeclaration !== '') {
    prefixSnippets.push(`
        layout(std140, set = 0, binding = ${
        1 + program.variableNames.length}) uniform Uniforms {
            ${uniformDeclaration}
        };
    `);
  }
  */


  // prefixSnippets.push(getGlslDifferences().defineSpecialNaN);

  const [getOutputCoords, dispatchLayoutRank] =
      generateGetOutputCoords(outputData.shape, program.dispatchLayout);
  const getCoords = generateGetCoordsFromFlatIndex(outputData.shape);
  /*
  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS, getCoords,
    getOutputCoords,
    getSetOutputSnippet(outputData.shape, outputData.dtype, program.isVec4)
  ];
  */

  const sources = [
    program.getUserHeaderCode(), SAMPLING_SNIPPETS, getCoords, getOutputCoords,
    getSetOutputSnippet(outputData.shape, outputData.dtype, program.isVec4)
  ];
  if (dispatchLayoutRank === outputData.shape.length) {
    // Input sampling snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSamplingSnippet =
        inputInfo
            .map(
                x => getInputSamplingSnippet(
                    x, outputData.shape, program.isVec4,
                    program.dispatchLayout.x.length ===
                        outputData.shape.length))
            .join('\n');
    sources.push(inputSamplingSnippet);
  }

  sources.push(program.getUserCode());
  // const source = program.getUserCode();
  const source = sources.join('\n');
  return source;
}

/*
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
`;
*/
const SAMPLING_SNIPPETS = `
  fn getFlatIndex(coord : u32, shape : u32) -> u32 {
    return coord;
  }

  fn getFlatIndex2(coords : vec2<u32>, shape : vec2<u32>) -> u32 {
    return u32(dot(vec2<f32>(coords), vec2<f32>(f32(shape.y), 1.0)));
  }

  fn getFlatIndex3(coords : vec3<u32>, shape : vec3<u32>) -> u32 {
    return u32(dot(vec3<f32>(coords), vec3<f32>(f32(shape.y) * f32(shape.z), f32(shape.z), 1.0)));
  }

  fn getFlatIndex4(coords : vec4<u32>, shape : vec4<u32>) -> u32 {
    return u32(dot(vec4<f32>(coords), vec4<f32>(
        f32(shape.y) * f32(shape.z) * f32(shape.w), f32(shape.z) * f32(shape.w), f32(shape.w), 1.0)));
  }
`;

const SHADER_PREFIX = ` `;

// const SAMPLING_SNIPPETS = ` `;

function getSetOutputSnippet(
    outShape: number[], outBufferType: DataType, isVec4: boolean): string {
  const outRank = outShape.length;
  const glslType = mapToGlslTypes(outBufferType, isVec4);
  let snippet;
  if (isVec4) {
    snippet = `fn setOutput(flatIndex : u32, value : vec4<f32>) {
      result.numbers[flatIndex] = ${
        glslType === 'vec4<i32>' ?
            'vec4<i32>(value)' :
            (glslType === 'vec4<bool>' ? 'vec4<bool>(value)' : 'value')};
    }
    fn setOutputI32(flatIndex : u32, value : vec4<i32>) {
      result.numbers[flatIndex] = ${
        glslType === 'vec4<f32>' ?
            'vec4<f32>(value)' :
            (glslType === 'vec4<bool>' ? 'vec4<bool>(value)' : 'value')};
    }`;
  } else {
    snippet = `fn setOutput(flatIndex : u32, value : f32) {
      result.numbers[flatIndex] = ${
        glslType === 'i32' ? 'i32(value)' :
                             (glslType === 'bool' ? 'bool(value)' : 'value')};
    }
    fn setOutputI32(flatIndex : u32, value : i32) {
      result.numbers[flatIndex] = ${
        glslType === 'f32' ? 'f32(value)' :
                             (glslType === 'bool' ? 'bool(value)' : 'value')};
    }`;
  }

  if (outRank >= 2) {
    switch (outRank) {
      case 2:
        snippet += `
        int getOutputFlatIndex(ivec2 coords) {
          return int(dot(coords, ivec2(outShapeStrides, 1)));
        }
        `;
        break;
      case 3:
        snippet += `
        int getOutputFlatIndex(ivec3 coords) {
          return int(dot(coords, ivec3(outShapeStrides.x, outShapeStrides.y, 1)));
        }
        `;
        break;
      case 4:
        snippet += `
        int getOutputFlatIndex(ivec4 coords) {
          return int(dot(coords, ivec4(
            outShapeStrides.x, outShapeStrides.y, outShapeStrides.z, 1)));
        }
        `;
        break;
      default:
        util.assert(false, () => `Unsupported ${outRank}D shape`);
        break;
    }
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    if (isVec4) {
      snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, vec4 value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex / 4, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, ivec4 value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex / 4, value);
      }
    `;
    } else {
      snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        int flatIndex = getOutputFlatIndex(${type}(${dims.join(', ')}));
        setOutput(flatIndex, value);
      }
    `;
    }
  }

  return snippet;
}


function getInputSamplingSnippet(
    inInfo: InputInfo, outShape: number[], isVec4: boolean,
    isFlatDispatchLayout: boolean): string {
  let res = getSamplerFromInInfo(inInfo, isVec4);

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getSamplerAtOutputCoords(
        inInfo, outShape, isVec4, isFlatDispatchLayout);
  }

  return res;
}

function getSamplerFromInInfo(inInfo: InputInfo, isVec4: boolean): string {
  const texName = inInfo.name;
  const rank = inInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
  const inputs = dims.map(d => `${d} : u32`).join(', ');

  if (rank < 1) {
    if (isVec4) {
      return `
        fn ${funcName}() -> vec4<f32>{
          return ${texName}.numbers[0];
        }
      `;
    }

    return `
      fn ${funcName}() ->f32 {
        return ${texName}.numbers[0];
      }
    `;
  }

  const shapeStr =
      `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;

  if (isVec4) {
    return `
      fn ${funcName}(${inputs}) -> vec4<f32>{
        return ${texName}.numbers[getFlatIndex(${type}(${dims.join(',')}),
          ${shapeStr}) / 4];
      }
      `;
  }

  return `
    fn ${funcName}(${inputs}) -> f32 {
      return f32(${texName}.numbers[getFlatIndex(${type}(${dims.join(',')}),
        ${shapeStr})]);
    }
   `;
}


export function getSamplerAtOutputCoords(
    inInfo: InputInfo, outShape: number[], isVec4: boolean,
    isFlatDispatchLayout: boolean): string {
  const texName = inInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);

  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const inRank = inInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  // If the inShape equals the outShape and the dispatch layout is flat, we can
  // directly use |gl_GlobalInvocationID.x| as the index and don't need coords
  // conversion between these two shapes.
  if (util.arraysEqual(inInfo.shape, outShape) && isFlatDispatchLayout) {
    if (isVec4) {
      return `
        fn ${funcName}(global_id : vec3<u32>) -> vec4<f32> {
          return ${texName}.numbers[global_id.x];
        }

        fn ${funcName}2(coords : ${type}) -> vec4<f32> {
          return ${texName}.numbers[${
          outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'} / 4];
        }
        `;
    } else {
      return `
      fn ${funcName}(global_id : vec3<u32>) -> f32 {
        return f32(${texName}.numbers[global_id.x]);
      }

      fn ${funcName}2(coords : ${type}) -> f32 {
        return f32(${texName}.numbers[${
          outRank > 1 ? 'getOutputFlatIndex(coords)' : 'coords'}]);
      }
      `;
    }
  }

  const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
  const rankDiff = outRank - inRank;

  let coordsSnippet = '';

  if (inRank === 0) {
    if (isVec4) {
      return `
      fn ${funcName}() -> vec4<f32> {
        return get${texFuncSnippet}();
      }

      fn ${funcName}(coords : ${type}) -> vec4<f32> {
        return get${texFuncSnippet}();
      }
    `;
    }
    return `
      fn ${funcName}() -> f32{
        return get${texFuncSnippet}();
      }

      fn ${funcName}( coords : ${type}) -> f32{
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

  const shapeStr = `${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;

  if (isVec4) {
    return `
      fn ${funcName}(global_id : vec3<u32>) -> vec4<f32> {
        let coords : ${type} = getOutputCoords(global_id);
        ${coordsSnippet}
        return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        shapeStr}) / 4];
      }

      fn ${funcName}(coords : ${type}) -> vec4<f32> {
        ${coordsSnippet}
        return ${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        shapeStr}) / 4];
      }
    `;
  }

  return `
    fn ${funcName}() ->f32 {
      let coords :${type} = getOutputCoords();
      ${coordsSnippet}
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      shapeStr})]);
    }

    fn ${funcName}(coords : ${type}) ->f32 {
      ${coordsSnippet}
      return f32(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      shapeStr})]);
    }
  `;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */
export function generateGetOutputCoords(
    outShape: number[],
    dispatchLayout: {x: number[], y?: number[], z?: number[]}):
    [string, number] {
  const {x, y = [], z = []} = dispatchLayout;

  const outRank = outShape.length;
  if (x.length === outRank) {
    const dtype = getCoordsDataType(outRank);
    const snippet = `fn getOutputCoords(global_id : vec3<u32>) -> ${dtype}{
      return getCoordsFromFlatIndex(u32(global_id.x));
    }
    `;
    return [snippet, outRank];
  }

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
      gatherDimensionsStr += `let d${arr[0]} : u32 =
        u32(gl_GlobalInvocationID[${i}]);`;
    } else {
      const strides = symbolicallyComputeStrides(arr, 'outShape');
      gatherDimensionsStr += `let index${i} : u32 =
          u32(gl_GlobalInvocationID[${i}]);`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr +=
            `let d${arr[j]} : u32 = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `let d${arr[j + 1]} : u32 = ` +
              `index${i} - d${arr[j]} * ${strides[j]};`;
        } else {
          gatherDimensionsStr +=
              `index${i} = index${i} - d${arr[j]} * ${strides[j]};`;
        }
      }
    }
  }

  const dimensions = [];
  for (let i = 0; i < rank; i++) {
    dimensions.push(`d${i}`);
  }

  const dtype = getCoordsDataType(rank);
  let snippet = `fn getOutputCoords(global_id : vec3<u32>) -> ${dtype}{
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
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
function generateGetCoordsFromFlatIndex(shape: number[]): string {
  const rank = shape.length;

  if (rank <= 1) {
    return `fn getCoordsFromFlatIndex(index : u32) ->u32 {return index; }`;
  }

  const strides = util.computeStrides(shape);
  let dtype = getCoordsDataType(rank);
  if (dtype == 'ivec4') {
    dtype = 'vec4<u32>';
  }
  const coords: string[] = [];
  for (let i = 0; i < rank; i++) {
    coords.push(`d${i}`);
  }

  if (strides.length === 1) {
    return `    fn getCoordsFromFlatIndex(index : u32) -> vec2<u32>{
      let d0 = index / outShapeStrides; let d1 = index - d0 * outShapeStrides;
      return vec2<u32>(d0,d1);
    }`;
  }
  const snippet =
      strides
          .map((_, i) => {
            const line1 =
                `let ${coords[i]} :u32 = index / outShapeStrides[${i}]`;
            const line2 = i === strides.length - 1 ?
                `let ${coords[i + 1]}  : u32 = index - ${
                    coords[i]} * outShapeStrides[${i}]` :
                `index = index - ${coords[i]} * outShapeStrides[${i}]`;
            return `${line1}; ${line2};`;
          })
          .join('');

  return `
    fn getCoordsFromFlatIndex(index : u32) -> ${dtype} {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}
