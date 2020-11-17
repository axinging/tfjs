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
import {getPackedMatrixTextureShapeWidthHeight} from './webgpu_texture_util';

export function GetFormatSize(texFormat: GPUTextureFormat) {
  switch (texFormat) {
    case 'r32float':
      return 4;
    case 'rgba8unorm':
      return 4;
    default:
      break;
  }
  console.error('Unsupported format!');
  return 4;
}

export function getBytesPerTexel(format: GPUTextureFormat): number {
  if (format == 'rgba32float' || format == 'rgba32uint')
    return 16;
  else if (format == 'r32float' || format == 'r32uint')
    return 4;
  else {
    console.error('Unsupported format ' + format);
    return 4;
  }
}

export type BackendValues = Float32Array|Int32Array|Uint8Array|Uint8Array[];

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: Map<string, GPUTexture[]> = new Map();
  private usedTextures: Map<string, GPUTexture[]> = new Map();
  private format: GPUTextureFormat;
  private kBytesPerTexel: number;
  private kBytesPerFloat = 4;
  public numBytesUsed = 0;
  public numBytesAllocated = 0;
  constructor(
      private device: GPUDevice, format: GPUTextureFormat = 'rgba32float') {
    this.format = format;
    this.kBytesPerTexel = getBytesPerTexel(this.format);
  }

  private addTexturePadding(
      textureData: Float32Array|Uint32Array,
      width: number,
      height: number,
      bytesPerRow: number,
  ) {
    let textureDataWithPadding =
        new Float32Array(bytesPerRow / this.kBytesPerFloat * height);

    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const dst = x + y * bytesPerRow / this.kBytesPerFloat;
        const src = x + y * width;
        textureDataWithPadding[dst] = textureData[src];
      }
    }
    return textureDataWithPadding;
  }

  // This will remove padding for data downloading from GPU texture.
  public removeTexturePadding(
      textureDataWithPadding: Float32Array, width: number, height: number) {
    const [widthTex, heightTex] =
        getPackedMatrixTextureShapeWidthHeight(height, width, this.format);
    const bytesPerRow = this.getBytesPerRow(widthTex);
    console.warn(
        'in remove: widthTex =' + widthTex + ',heightTex = ' + heightTex);

    let textureData = new Float32Array(width * height);

    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const src = x + y * bytesPerRow / this.kBytesPerFloat;
        const dst = x + y * width;
        textureData[dst] = textureDataWithPadding[src];
      }
    }
    console.warn('in removeTexturePadding textureData=' + textureData);
    return textureData;
  }

  public getBufferSize(width: number, height: number) {
    const [widthTex, heightTex] =
        getPackedMatrixTextureShapeWidthHeight(height, width, this.format);

    const bytesPerRow = this.getBytesPerRow(widthTex);
    return bytesPerRow * heightTex;
  }

  public writeTexture(
      queue: GPUQueue, texture: GPUTexture, data: BackendValues, width: number,
      height: number) {
    const [widthTex, heightTex] =
        getPackedMatrixTextureShapeWidthHeight(height, width, this.format);

    const bytesPerRow = this.getBytesPerRow(widthTex);

    /* Alignment is not required for writeTexture.
    const dataWithPadding = this.addTexturePadding(
        data as Float32Array, widthTex, heightTex, bytesPerRow);
    console.log('writeTexture dataWithPadding ' + dataWithPadding);
    */

    queue.writeTexture(
        {texture: texture}, data as ArrayBuffer,
        {bytesPerRow: bytesPerRow},  // heightTex
        {width: widthTex, height: heightTex, depth: 1});
    return texture;
  }


  public writeTextureWithCopy(
      device: GPUDevice, texture: GPUTexture, matrixData: BackendValues,
      width: number, height: number) {
    console.warn(
        ' write wxh=' + width + ', ' + height +
        ', getBufferSize = ' + this.getBufferSize(width, height));
    const src = this.device.createBuffer({
      mappedAtCreation: true,
      size: this.getBufferSize(width, height),  // 640 * 4,  //
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST
    });

    const [widthTex, heightTex] =
        getPackedMatrixTextureShapeWidthHeight(height, width, this.format);


    console.log(
        'copyBufferToTexture width =' + width + ', height=' + height + '; ' +
        ' widthTex =' + widthTex + ', heightTex=' + heightTex);

    const bytesPerRow = this.getBytesPerRow(widthTex);

    const matrixDataWithAlignment = this.addTexturePadding(
        matrixData as Float32Array, width, height, bytesPerRow);

    new Float32Array(src.getMappedRange()).set(matrixDataWithAlignment);
    src.unmap();

    const encoder = this.device.createCommandEncoder();

    encoder.copyBufferToTexture(
        {buffer: src, bytesPerRow: bytesPerRow},
        {texture: texture, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
        {width: widthTex, height: heightTex, depth: 1});
    this.device.defaultQueue.submit([encoder.finish()]);
    return texture;
  }

  acquireTexture(
      width: number, height: number, texFormat: GPUTextureFormat,
      usages: GPUTextureUsageFlags) {
    const key = getTextureKey(width, height, texFormat, usages);
    console.log('acquireTexture keytex: ' + key);
    if (!this.freeTextures.has(key)) {
      this.freeTextures.set(key, []);
    }

    if (!this.usedTextures.has(key)) {
      this.usedTextures.set(key, []);
    }
    this.numBytesUsed += width * height * this.kBytesPerTexel;
    this.numUsedTextures++;

    if (this.freeTextures.get(key).length > 0) {
      this.numFreeTextures--;

      const newTexture = this.freeTextures.get(key).shift();
      this.usedTextures.get(key).push(newTexture);
      return newTexture;
    }
    const [widthTex, heightTex] =
        getPackedMatrixTextureShapeWidthHeight(height, width, texFormat);

    this.numBytesAllocated += width * height * this.kBytesPerTexel;
    const newTexture = this.device.createTexture({
      size: {width: widthTex, height: heightTex, depth: 1},
      format: texFormat,
      dimension: '2d',
      usage: usages,
    });
    this.usedTextures.get(key).push(newTexture);

    return newTexture;
  }

  releaseTexture(
      texture: GPUTexture, width: number, height: number,
      texFormat: GPUTextureFormat, usage: GPUTextureUsageFlags) {
    if (this.freeTextures == null) {
      return;
    }

    const key = getTextureKey(width, height, texFormat, usage);
    if (!this.freeTextures.has(key)) {
      this.freeTextures.set(key, []);
    }

    this.freeTextures.get(key).push(texture);
    this.numFreeTextures++;
    this.numUsedTextures--;

    const textureList = this.usedTextures.get(key);
    const textureIndex = textureList.indexOf(texture);
    if (textureIndex < 0) {
      throw new Error(
          'Cannot release a Texture that was never provided by this ' +
          'Texture manager');
    }
    textureList.splice(textureIndex, 1);
    this.numBytesUsed -= width * height * this.kBytesPerFloat;
  }

  getBytesPerRow(width: number) {
    const kTextureBytesPerRowAlignment = 256;
    const alignment = kTextureBytesPerRowAlignment;
    const value = this.kBytesPerTexel * width;
    const bytesPerRow =
        ((value + (alignment - 1)) & ((~(alignment - 1)) >>> 0)) >>> 0;
    return bytesPerRow;
  }

  getBytesPerTexel(format: GPUTextureFormat): number {
    if (format == 'rgba32float' || format == 'rgba32uint')
      return 16;
    else if (format == 'r32float' || format == 'r32uint')
      return 4;
    else {
      console.error('Unsupported format ' + format);
      return 4;
    }
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  getNumFreeTextures(): number {
    return this.numFreeTextures;
  }

  reset() {
    this.freeTextures = new Map();
    this.usedTextures = new Map();
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }

  dispose() {
    if (this.freeTextures == null && this.usedTextures == null) {
      return;
    }

    this.freeTextures.forEach((textures) => {
      textures.forEach(tex => {
        tex.destroy();
      });
    });

    this.usedTextures.forEach((textures) => {
      textures.forEach(tex => {
        tex.destroy();
      });
    });

    this.freeTextures = null;
    this.usedTextures = null;
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }
}

function getTextureKey(
    width: number, height: number, format: GPUTextureFormat,
    usage: GPUTextureUsageFlags) {
  return `${width}_${height}_${format}_${usage}`;
}
