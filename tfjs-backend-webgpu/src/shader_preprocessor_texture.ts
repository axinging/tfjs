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
// const {getBroadcastDims} = backend_util;

import * as shader_util from './shader_compiler_util';
import {symbolicallyComputeStrides} from './shader_util';
import {PACKED_RGBA_HEIGHT, PACKED_RGBA_WIDTH} from './webgpu_texture_util';

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
  dtype: DataType,
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

function getGlslFormat(format: GPUTextureFormat) {
  if (format == 'rgba32float') {
    return 'rgba32f';
  } else
    return 'r32f';
}

export function makeShader(
    inputInfo: InputInfo[], outputData: ShapeInfo, program: ProgramParams,
    format: GPUTextureFormat): string {
  const prefixSnippets: string[] = [];
  const glslFormat = getGlslFormat(format);

  if (program.workGroupSize != null) {
    prefixSnippets.push(`
      layout (local_size_x = ${program.workGroupSize[0]},
              local_size_y = ${program.workGroupSize[1]},
              local_size_z = ${program.workGroupSize[2]}) in;
    `);
  }

  prefixSnippets.push(`
    layout(set = 0, binding = 0, ${
      glslFormat}) uniform writeonly image2D result;
  `);

  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
        layout(set = 0, binding = ${1 + i}, ${
        glslFormat}) uniform readonly image2D ${x};
      `);
  });

  let uniformDeclaration = '';

  if (program.uniforms) {
    uniformDeclaration += program.uniforms;
    const bindingOffset = 1 + program.variableNames.length;
    prefixSnippets.push(`
    layout(std140, set = 0, binding = ${bindingOffset}) uniform Uniforms {
      ${uniformDeclaration}
    };
  `);
  }
  // const [getOutputCoords, dispatchLayoutRank] =
  //    generateGetOutputCoords(outputData.shape, program.dispatchLayout);
  const getCoords = generateGetCoordsFromFlatIndex(outputData.logicalShape);
  // const outputSamplingSnippet =
  // getOutputSamplingSnippet(outputData.logicalShape, outputData.texShape);

  let outputSamplingSnippet = '';
  if (glslFormat == 'rgba32f') {
    outputSamplingSnippet = getPackedOutputSamplingSnippet(
        outputData.logicalShape, outputData.texShape);
  } else {
    outputSamplingSnippet =
        getOutputSamplingSnippet(outputData.logicalShape, outputData.texShape);
  }

  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    outputSamplingSnippet, getCoords  // ,
    // getSetOutputSnippet(outputData.logicalShape, outputData.dtype)
  ];
  let packedMode = false;
  if (glslFormat == 'rgba32f') {
    packedMode = true;
    sources.push(SAMPLE_1D_SNIPPET);
    sources.push(SAMPLE_2D_SNIPPET);
    sources.push(SAMPLE_3D_SNIPPET);
    sources.push(getPackedSetOutputSnippet(
        outputData.logicalShape, outputData.texShape, outputData.dtype));

  } else {
    packedMode = false;
    sources.push(SHADER_PREFIX_R32F);
    sources.push(
        getSetOutputSnippet(outputData.logicalShape, outputData.dtype));
  }

  const inputSamplingSnippet =
      inputInfo.map(x => getInputSamplingSnippet(x, outputData, packedMode))
          .join('\n');
  sources.push(inputSamplingSnippet);

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

  int imod(int x, int y) {
    return x - y * (x / y);
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

const SHADER_PREFIX_R32F =
    `ivec2 uvFromFlat(int texNumR, int texNumC, int index) {
      int texR = index / texNumC;
      int texC = index - texR * texNumC;
      return ivec2(texC, texR);
   }`;

const SAMPLE_1D_SNIPPET = `
  ivec2 uvFromFlat(int texNumR, int texNumC, int index) {
    int texR = index / texNumC;
    int texC = index - texR * texNumC;
    return ivec2(texC, texR);
  }

  ivec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
    int texelIndex = index /${PACKED_RGBA_WIDTH};
    int texR = texelIndex / texNumC;
    int texC = texelIndex - texR * texNumC;
    return ivec2(texC, texR);
  }
  `;

const SAMPLE_2D_SNIPPET = `
  ivec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
    int texNumC, int row, int col) {
    int texelIndex = (row / ${
    PACKED_RGBA_HEIGHT}) * texelsInLogicalRow + (col / ${PACKED_RGBA_WIDTH});
    int texR = texelIndex / texNumC;
    int texC = texelIndex - texR * texNumC;
    return ivec2(texC, texR);
  }
  `;

const SAMPLE_3D_SNIPPET = `
  ivec2 packedUVfrom3D(int texNumR, int texNumC,
      int texelsInBatch, int texelsInLogicalRow, int b,
      int row, int col) {
    int index = b * texelsInBatch + (row / ${
    PACKED_RGBA_HEIGHT}) * texelsInLogicalRow + (col /  ${PACKED_RGBA_WIDTH});
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
  if (useTexture && outRank === 2) {
    snippet = `void setOutput(float value) {
      // TODO(texture): This only works under 2D.
		  imageStore(result, getOutputCoords(), vec4(value, 0.0, 0.0, 0.0));
    }
    void setOutput(int flatIndex, float value) {
    }
    void setOutput(int flatIndex, int value) {
    }`;
  } else if (useTexture && outRank === 3) {
    snippet = `void setOutput(float value) {
      // TODO(texture): This only works under 3D.
      ivec3 coords = getOutputCoords();
      int texR = int(dot(vec2(coords[0], coords[1]), vec2(${outShape[1]}, 1)));
      int texC = coords[2];
      imageStore(result, ivec2(texC, texR), vec4(value, 0.0, 0.0, 0.0));
    }
    void setOutput(int flatIndex, float value) {
    }
    void setOutput(int flatIndex, int value) {
    }`;
  } else if (useTexture && outRank === 4) {
    snippet = `void setOutput(float value) {
      // TODO(texture): This only works under 4D.
      ivec4 coords = getOutputCoords();
      int texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        outShape[1]} * ${outShape[2]}, ${outShape[2]}, 1)));
      int texC = coords[3];
      imageStore(result, ivec2(texC, texR), vec4(value, 0.0, 0.0, 0.0));
    }
    void setOutput(int flatIndex, float value) {
      ivec4 coords = getCoordsFromFlatIndex(flatIndex);
      int texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        outShape[1]} * ${outShape[2]}, ${outShape[2]}, 1)));
      int texC = coords[3];
      imageStore(result, ivec2(texC, texR), vec4(value, 0.0, 0.0, 0.0));
    }
    void setOutput(int flatIndex, int value) {
    }`;
  } else {
    snippet = `void setOutput(int flatIndex, float value) {
      /*
      result[flatIndex] = ${
        glslType === 'int' ? 'int(value)' :
                             (glslType === 'bool' ? 'bool(value)' : 'value')};
                          */
    }
    void setOutput(int flatIndex, int value) {
      /*
      result[flatIndex] = ${
        glslType === 'float' ?
            'float(value)' :
            (glslType === 'bool' ? 'bool(value)' : 'value')};
      */
    }`;
  }

  if (outRank == 2) {
    snippet += `ivec2 getUVfromCoords(int d0, int d1) {
      return ivec2(d0, d1);
    }`
  } else if (outRank == 3) {
    snippet += `ivec2 getUVfromCoords(int d0, int d1, int d2) {
      int texR = int(dot(ivec2(d0, d1), ivec2(${outShape[1]}, 1)));
      int texC = d2;
      return ivec2(texR, texC);
    }`
  } else if (outRank == 4) {
    snippet += `ivec2 getUVfromCoords(int d0, int d1, int d2, int d3) {
      int texR = int(
          dot(ivec3(d0, d1, d2),
              ivec3(${outShape[1]} * ${outShape[2]}, ${outShape[2]}, 1)));
      int texC = d3;
      return ivec2(texR, texC);
    }`
  }

  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
        getShapeCoords(outShape)});
        //setOutput(flatIndex, value);
        // TODO(texture): make this work for 3D, 2D.
        /*
        ivec4 coords = ivec4(${dims.map(d => `${d}`).join(', ')});
        int texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        outShape[1]} * ${outShape[2]}, ${outShape[2]}, 1)));
        int texC = coords[3];
        */
        ivec2 uv = getUVfromCoords(${dims.map(d => `${d}`).join(', ')});
        int texR = uv.x;
        int texC = uv.y;
        imageStore(result, ivec2(texC, texR), vec4(value, 0.0, 0.0, 0.0));
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), ${
        getShapeCoords(outShape)});
        /*
        // setOutput(flatIndex, value);
        // TODO(texture): not tested.
        // TODO(texture): make this work for 3D, 2D.
        ivec4 coords = ivec4(${dims.map(d => `${d}`).join(', ')});
        int texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        outShape[1]} * ${outShape[2]}, ${outShape[2]}, 1)));
        int texC = coords[3];
        */
        ivec2 uv = getUVfromCoords(${dims.map(d => `${d}`).join(', ')});
        int texR = uv.x;
        int texC = uv.y;
        imageStore(result, ivec2(texC, texR), vec4(value, 0.0, 0.0, 0.0));
      }
    `;
  }

  return snippet;
}

function setPackedSampler2D(
    shape: number[], texShape: [number, number]): string {
  const outRank = shape.length;
  const type = getCoordsDataType(outRank);
  // TODO(texture): clean this, outRank - inRank;
  const rankDiff = 0;
  let coordsSnippet = '';
  const fields = ['x', 'y', 'z', 'w', 'u', 'v'];
  let unpackedCoordsSnippet =
      shape.map((s, i) => `coords.${fields[i + rankDiff]}`).join(', ');

  // TODO(texture): make sure below path will be used for the right case.
  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      void setOutput(int row, int col, vec4 value) {
        ivec2 uv = ivec2(col, row);
        imageStore(result, ivec2(uv.y, uv.x), value);
      }
    `;
  }

  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];
  const valuesPerRow = Math.ceil(shape[1] / PACKED_RGBA_WIDTH);

  return `
    void setOutput(int row, int col, vec4 value) {
      ivec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${
      packedTexShape[1]}, row, col);
      imageStore(result, ivec2(uv.x, uv.y), value);
    }
    void setOutput(vec4 value) {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      setOutput(${unpackedCoordsSnippet}, value);
    }
  `;
}

export function getPackedSetOutputSnippet(
    outShape: number[], texShape: [number, number],
    outBufferType: DataType): string {
  const outRank = outShape.length;
  if (outRank <= 1)
    // console.error('<=1D is not supported in getPackedSetOutputSnippet!');
    return setPackedSampler1D(outShape, texShape);
  if (outRank == 2) {
    return setPackedSampler2D(outShape, texShape);
  } else if (outRank == 3) {
    return setPackedSampler3D(outShape, texShape);
  } else {
    return setPackedSamplerND(outShape, texShape);
  }
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShapeInfo: ShapeInfo,
    usesPackedTextures = false): string {
  // let res = getSamplerFromInInfo2(inInfo);

  let res = '';
  if (usesPackedTextures) {
    res += getPackedSamplerFromInInfo(inInfo);
  } else {
    res += getSamplerFromInInfo2(inInfo);
  }
  // TODO(texture): this is for mm_read
  // res += getSamplerFromInInfo2(inInfo);
  // TODO(texture): move getSamplerCommon to a proper place.
  // res += getSamplerCommon(inInfo);
  const outShape = outShapeInfo.logicalShape;
  const outTexShape = outShapeInfo.texShape;

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    if (usesPackedTextures) {
      res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo);
    } else {
      res += getSamplerAtOutputCoords(inInfo, outShape, outTexShape);
    }
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
    inInfo: InputInfo, outShape: number[], outTexShape: number[]): string {
  const texName = inInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);

  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const inRank = inInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  const broadcastDims = backend_util.getBroadcastDims(inInfo.shape, outShape);
  const rankDiff = outRank - inRank;
  // const inTexShape = inInfo.shapeInfo.texShape;

  /*
  if (!inInfo.shapeInfo.isUniform && inRank === outRank &&
      inInfo.shapeInfo.flatOffset == null &&
      util.arraysEqual(inTexShape, outTexShape)) {
    return `
  float ${funcName}() {
    // return sampleTexture(${texName}, resultUV);
    // TODO(texture): this maybe wrong when gl_GlobalInvocationID has z index.
    ivec2 resultUV = ivec2(gl_GlobalInvocationID.yx);
    return imageLoad(${texName}, ivec2(resultUV.x, resultUV.y)).r;
  }
`;
  }
  */

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

  if (outRank == 4)
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      // TODO(texture): handle this for squeezed and 2D, and 3D.
      int texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        inInfo.shape[1]} * ${inInfo.shape[2]}, ${inInfo.shape[2]}, 1)));
      int texC = coords[3];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      int texR = int(dot(vec3(coords[0], coords[1], coords[2]), vec3(${
        inInfo.shape[1]} * ${inInfo.shape[2]}, ${inInfo.shape[2]}, 1)));
      int texC = coords[3];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }
  `;
  else if (outRank == 3)
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      // TODO(texture): handle this for squeezed and 2D, and 3D.
      int texR = int(dot(vec2(coords[0], coords[1]), vec2(${
        inInfo.shape[1]}, 1)));
      int texC = coords[2];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      int texR = int(dot(vec2(coords[0], coords[1]), vec2(${
        inInfo.shape[1]}, 1)));
      int texC = coords[2];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }
  `;
  else if (outRank == 2)
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      // TODO(texture): this shift x and y is bad.
      int texR = coords[1];
      int texC = coords[0];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      int texR = coords[0];
      int texC = coords[1];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }
  `;
  else  // TODO(texture). This not tested.
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      // TODO(texture): handle this for squeezed and 2D, and 3D.
      int texR = int(dot(vec2(coords[0], coords[1]), vec2(${
        inInfo.shape[1]}, 1)));
      int texC = coords[2];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
    }

    float ${funcName}(${type} coords) {
      ${coordsSnippet}
      int texR = int(dot(vec2(coords[0], coords[1]), vec2(${
        inInfo.shape[1]}, 1)));
      int texC = coords[2];
      return imageLoad(${texName}, ivec2(texC,texR)).r;
      /*
      return float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
        getShapeCoords(inInfo.shape)})]);
      */
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
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
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

function getPackedSamplerFromInInfo(inInfo: InputInfo): string {
  const shape = inInfo.shapeInfo.logicalShape;
  switch (shape.length) {
    case 0:
      return getPackedSamplerScalar(inInfo);
    case 1:
      return getPackedSampler1D(inInfo);
    case 2:
      return getPackedSampler2D(inInfo);
    case 3:
      // return getPackedSampler3D(inInfo);
      return getPackedSamplerND(inInfo);
    default:
      return getPackedSamplerND(inInfo);
  }
}

function getPackedSampler2D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;

  // const texNumR = texShape[0];
  // const texNumC = texShape[1];
  console.log('getPackedSampler2D shape=' + shape + ', texShape =' + texShape);
  if (texShape != null && util.arraysEqual(shape, texShape)) {
    return `
      vec4 ${funcName}(int row, int col) {
        ivec2 uv = ivec2(col, row);

        return imageLoad(${texName}, uv);
      }
    `;
  }

  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];
  const valuesPerRow = Math.ceil(shape[1] / PACKED_RGBA_WIDTH);

  return `
    vec4 ${funcName}(int row, int col) {
      ivec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${
      packedTexShape[1]}, row, col);
      return imageLoad(${texName}, uv);
    }
  `;
}

function getPackedSamplerScalar(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  return `
    vec4 ${funcName}() {
      return imageLoad(${texName}, ivec2(0, 0));
    }
  `;
}

function getPackedSampler1D(inputInfo: InputInfo): string {
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_HEIGHT),
    Math.ceil(texShape[1] / PACKED_RGBA_WIDTH)
  ];

  return `
    vec4 ${funcName}(int index) {
      ivec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return imageLoad(${texName}, ivec2(uv.y,uv.x));
    }
  `;
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
      return imageLoad(${texName}, ivec2(row,col)).r; 
    }
  `;
}

export function getPackedSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];

  if (shape[0] === 1) {
    const squeezedShape = shape.slice(1);
    const keptDims = [1, 2];
    const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
    const params = ['b', 'row', 'col'];
    return `
        ${getPackedSamplerFromInInfo(newInputInfo)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
  }

  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[2] / PACKED_RGBA_WIDTH);
  const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / PACKED_RGBA_HEIGHT);
  // const glsl = getGlslDifferences();

  return `
    vec4 ${funcName}(int b, int row, int col) {
      ivec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${
      valuesPerRow}, b, row, col);
      return imageLoad(${texName}, uv);
    }
  `;
}

export function getSampler3D(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const stride0 = shape[1] * shape[2];
  const stride1 = shape[2];
  console.warn('3D texName ' + texName + ', ' + shape);

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

  console.warn(' texNumR = ' + texNumR);
  console.warn(' stride0 = ' + stride0);
  console.warn(' texNumC = ' + texNumC);
  console.warn(' stride1 = ' + stride1);
  if (texNumR === stride0)
    // texC is used directly as physical (no risk of float16 overflow).
    // TODO(texture): not tested.
    return `
        float ${funcName}(int row, int col, int depth) {
          int texR = row;
          int texC = int(dot(vec2(col, depth), vec2(${stride1}, 1)));
          return imageLoad(${texName}, ivec2(texR,texC)).r;
        }
      `;

  // case [logShape[0] * logShape[1], logShape[2]];
  // TODO(texture): this seems R and C were misused.
  if (texNumR === stride1) {
    // texR is used directly as physical (no risk of float16 overflow).
    return `
    float ${funcName}(int row, int col, int depth) {
      int texR = int(dot(vec2(row, col), vec2(${inputInfo.shape[1]}, 1)));
      int texC = depth;
      return imageLoad(${texName}, ivec2(texC,texR)).r;
    }
  `;
  }
  // TODO(texture): not tested.
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
      ${getSamplerFromInInfo2(newInputInfo)}
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

function getPackedSamplerND(inputInfo: InputInfo): string {
  const shape = inputInfo.shapeInfo.logicalShape;
  const rank = shape.length;
  const texName = inputInfo.name;
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const texShape = inputInfo.shapeInfo.texShape;
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];
  //[Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
  // console.log();
  const texNumC = packedTexShape[0];
  // const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[rank - 1] / PACKED_RGBA_WIDTH);
  let texelsInBatch =
      valuesPerRow * Math.ceil(shape[rank - 2] / PACKED_RGBA_HEIGHT);
  let params = `int b, int row, int col`;
  let index = `b * ${texelsInBatch} + (row / ${PACKED_RGBA_HEIGHT}) * ${
      valuesPerRow} + (col / ${PACKED_RGBA_WIDTH})`;
  for (let b = 2; b < rank - 1; b++) {
    params = `int b${b}, ` + params;
    texelsInBatch *= shape[rank - b - 1];
    index = `b${b} * ${texelsInBatch} + ` + index;
  }
  return `
    vec4 ${funcName}(${params}) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      return imageLoad(${texName}, ivec2(texC,texR));
    }
  `;
}

function setPackedSampler1D(
    shape: number[], texShape: [number, number]): string {
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_HEIGHT),
    Math.ceil(texShape[1] / PACKED_RGBA_WIDTH)
  ];

  return `
    void setOutput(vec4 value) {
      int coords = getOutputCoords();
      vec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, coords);
      imageStore(result, ivec2(uv.y,uv.x), value);
    }
  `;
}

// From getPackedSampler3D
export function setPackedSampler3D(
    shape: number[], texShape: [number, number]): string {
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];

  const texNumR = packedTexShape[0];
  const texNumC = packedTexShape[1];

  const valuesPerRow = Math.ceil(shape[2] / PACKED_RGBA_WIDTH);
  const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / PACKED_RGBA_HEIGHT);


  return `
    void setOutput(vec4 value) {
      ivec3 coords = getOutputCoords();
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${
      valuesPerRow}, coords[0], coords[1], coords[2]);
      imageStore(result, ivec2(uv.y,uv.x), value);
    }

    void setOutput(int b, int row, int col, vec4 value) {
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${
      valuesPerRow}, b, row, col);
      imageStore(result, ivec2(uv.y,uv.x), value);
    }
  `;
}

// From: getPackedSamplerND
export function setPackedSamplerND(
    shape: number[], texShape: [number, number]): string {
  const rank = shape.length;
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];

  const texNumC = packedTexShape[0];

  const valuesPerRow = Math.ceil(shape[rank - 1] / PACKED_RGBA_WIDTH);
  let texelsInBatch =
      valuesPerRow * Math.ceil(shape[rank - 2] / PACKED_RGBA_HEIGHT);
  let params = `int b, int row, int col`;
  let index = `b * ${texelsInBatch} + (row / ${PACKED_RGBA_HEIGHT}) * ${
      valuesPerRow} + (col / ${PACKED_RGBA_WIDTH})`;
  for (let b = 2; b < rank - 1; b++) {
    params = `int b${b}, ` + params;
    texelsInBatch *= shape[rank - b - 1];
    index = `b${b} * ${texelsInBatch} + ` + index;
  }
  return `
    void setOutput(${params}, vec4 value) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      imageStore(result, ivec2(texC, texR), value);
    }
    // TODO(texture): remove this.
    void setOutput(vec4 value) {
      ivec4 coords = getOutputCoords();
	    setOutput(coords[0],coords[1],coords[2],coords[3], value);
      // imageStore(result, ivec2(col, row), value);
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

function getOutput2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  if (util.arraysEqual(shape, texShape)) {
    console.error('TODO(texture), not tried');
    return `
      ivec2 getOutputCoords() {
        ivec2 resultUV = ivec2(gl_GlobalInvocationID.xy);
        return ivec2(resultUV.yx * vec2(${texShape[0]}, ${texShape[1]}));
      }
    `;
  }
  if (shape[1] === 1) {
    console.error('TODO(texture), not tried');
    return `
      ivec2 getOutputCoords() {
        ivec2 resultUV = ivec2(gl_GlobalInvocationID.xy);
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
  }
  if (shape[0] === 1) {
    console.error('TODO(texture), not tried');
    return `
      ivec2 getOutputCoords() {
        ivec2 resultUV = ivec2(gl_GlobalInvocationID.xy);
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `;
  }
  return `
    ivec2 getOutputCoords() {
      ivec2 resultUV = ivec2(gl_GlobalInvocationID.yx);
      /*
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${shape[1]};
      int c = index - r * ${shape[1]};
      return ivec2(r, c);
      */
      return resultUV;
    }
  `;
}

function getOutputPacked1DCoords(
    shape: [number], texShape: [number, number]): string {
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_HEIGHT),
    Math.ceil(texShape[1] / PACKED_RGBA_WIDTH)
  ];
  if (packedTexShape[0] === 1) {
    return `
      int getOutputCoords() {
        ivec2 resultUV = ivec2(gl_GlobalInvocationID.xy);
        return ${PACKED_RGBA_HEIGHT} * int(resultUV.y);
      }
    `;
  }

  if (packedTexShape[1] === 1) {
    return `
      int getOutputCoords() {
        ivec2 resultUV = ivec2(gl_GlobalInvocationID.xy);
        return ${PACKED_RGBA_WIDTH} * int(resultUV.x);
      }
    `;
  }

  return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_GlobalInvocationID.xy);
      return ${PACKED_RGBA_WIDTH} * (resTexRC.x * ${
      packedTexShape[1]} + resTexRC.y);
    }
  `;
}

function getOutputPacked2DCoords(
    shape: [number, number], texShape: [number, number]): string {
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];
  if (util.arraysEqual(shape, texShape)) {
    return `
      ivec2 getOutputCoords() {
        ivec2 resultUV = ivec2(gl_GlobalInvocationID.xy);
        return ivec2(${PACKED_RGBA_WIDTH}*resultUV.y, ${
        PACKED_RGBA_HEIGHT}*resultUV.x);
      }
    `;
  }

  // texels needed to accommodate a logical row
  const texelsInLogicalRow = Math.ceil(shape[1] / PACKED_RGBA_HEIGHT);

  /**
   * getOutputCoords
   *
   * resTexRC: The rows and columns of the texels. If you move over one
   * texel to the right in the packed texture, you are moving over one column
   * (not two).
   *
   * index: The texel index
   */
  return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_GlobalInvocationID.xy);

      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
      int r = ${PACKED_RGBA_HEIGHT} * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * ${PACKED_RGBA_WIDTH};

      return ivec2(r, c);
    }
  `;
}

function getOutputPacked3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];
  const texelsInLogicalRow = Math.ceil(shape[2] / PACKED_RGBA_WIDTH);
  const texelsInBatch =
      texelsInLogicalRow * Math.ceil(shape[1] / PACKED_RGBA_HEIGHT);

  return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = ${PACKED_RGBA_HEIGHT} * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * ${PACKED_RGBA_WIDTH};

      return ivec3(b, r, c);
    }
  `;
}

export function getOutput3DCoords(
    shape: [number, number, number], texShape: [number, number]): string {
  const coordsFromIndexSnippet =
      shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);

  return `
    ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);
    int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
    ${coordsFromIndexSnippet}
    return ivec3(r, c, d);
    }
  `;
}

function getOutputPackedNDCoords(
    shape: number[], texShape: [number, number]): string {
  const packedTexShape = [
    Math.ceil(texShape[0] / PACKED_RGBA_WIDTH),
    Math.ceil(texShape[1] / PACKED_RGBA_HEIGHT)
  ];
  // 4D NHWC:[batch, height, width, channels]
  console.log('wgs: shape =' + shape);
  const texelsInLogicalRow =
      Math.ceil(shape[shape.length - 1] / PACKED_RGBA_WIDTH);
  const texelsInBatch = texelsInLogicalRow *
      Math.ceil(shape[shape.length - 2] / PACKED_RGBA_HEIGHT);
  let texelsInBatchN = texelsInBatch;
  let batches = ``;
  let coords = 'b, r, c';

  for (let b = 2; b < shape.length - 1; b++) {
    texelsInBatchN *= shape[shape.length - b - 1];
    batches = `
      int b${b} = index / ${texelsInBatchN};
      index -= b${b} * ${texelsInBatchN};
    ` + batches;
    coords = `b${b}, ` + coords;
  }

  return `
    ivec${shape.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      ${batches}

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = ${PACKED_RGBA_HEIGHT} * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * ${PACKED_RGBA_WIDTH};

      return ivec${shape.length}(${coords});
    }
  `;
}

function getOutput4DCoords(
    shape: [number, number, number, number],
    texShape: [number, number]): string {
  const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(
      ['r', 'c', 'd', 'd2'], shape);

  return `
  ivec4 getOutputCoords() {
    ivec2 resTexRC = ivec2(gl_GlobalInvocationID.yx);
    int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
    ${coordsFromIndexSnippet}
    return ivec4(r, c, d, d2);
  }
`;
}

function getPackedOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    case 0:
      // return getOutputScalarCoords();
      console.error('0D not supported');
      return '';
    case 1:
      return getOutputPacked1DCoords(outShape as [number], outTexShape);
    case 2:
      return getOutputPacked2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutputPacked3DCoords(
          outShape as [number, number, number], outTexShape);
    default:
      return getOutputPackedNDCoords(outShape, outTexShape);
  }
}


function getOutputSamplingSnippet(
    outShape: number[], outTexShape: [number, number]): string {
  switch (outShape.length) {
    /*
    case 0:
      return getOutputScalarCoords();
    case 1:
      return getOutput1DCoords(outShape as [number], outTexShape);
    */
    case 2:
      return getOutput2DCoords(outShape as [number, number], outTexShape);
    case 3:
      return getOutput3DCoords(
          outShape as [number, number, number], outTexShape);
    case 4:
      return getOutput4DCoords(
          outShape as [number, number, number, number], outTexShape);
    /*
    case 5:
      return getOutput5DCoords(
          outShape as [number, number, number, number, number], outTexShape);
    case 6:
      return getOutput6DCoords(
          outShape as [number, number, number, number, number, number],
          outTexShape);
    */
    default:
      throw new Error(
          `${outShape.length}-D output sampling is not yet supported`);
  }
}


export function getPackedSamplerAtOutputCoords(
    inputInfo: InputInfo, outShapeInfo: ShapeInfo) {
  const texName = inputInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
  const inRank = inputInfo.shapeInfo.logicalShape.length;
  const outRank = outShapeInfo.logicalShape.length;

  const broadcastDims = backend_util.getBroadcastDims(
      inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);

  const type = getCoordsDataType(outRank);
  const rankDiff = outRank - inRank;
  let coordsSnippet: string;
  const fields = ['x', 'y', 'z', 'w', 'u', 'v'];

  if (inRank === 0) {
    coordsSnippet = '';
  } else if (outRank < 2 && broadcastDims.length >= 1) {
    coordsSnippet = 'coords = 0;';
  } else {
    coordsSnippet =
        broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
            .join('\n');
  }
  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
                                .map((s, i) => `coords.${fields[i + rankDiff]}`)
                                .join(', ');
  }

  let output = `return outputValue;`;
  const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
  const isInputScalar = inSize === 1;
  const outSize = util.sizeFromShape(outShapeInfo.logicalShape);
  const isOutputScalar = outSize === 1;

  if (inRank === 1 && !isInputScalar && !isOutputScalar) {
    output = `
    return outputValue;// vec4(outputValue.xy, outputValue.xy);
  `;
  } else if (isInputScalar && !isOutputScalar) {
    if (outRank === 1) {
      output = `
      return vec4(outputValue.x, outputValue.x, 0., 0.);
    `;
    } else {
      output = `
      return vec4(outputValue.x);
    `;
    }
  } else if (broadcastDims.length) {
    const rows = inRank - 2;
    const cols = inRank - 1;

    if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
      output = `return vec4(outputValue.x);`;
    } else if (broadcastDims.indexOf(rows) > -1) {
      output = `return vec4(outputValue.x, outputValue.y, ` +
          `outputValue.x, outputValue.y);`;
    } else if (broadcastDims.indexOf(cols) > -1) {
      output = `return vec4(outputValue.xx, outputValue.zz);`;
    }
  }

  return `
  vec4 ${funcName}() {
    ${type} coords = getOutputCoords();
    ${coordsSnippet}
    vec4 outputValue = get${texFuncSnippet}(${unpackedCoordsSnippet});
    ${output}
  }
`;
}
