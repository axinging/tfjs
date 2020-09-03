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

export function GetFormatSize(texFormat: GPUTextureFormat) {
  switch (texFormat) {
    case 'r32float':
      return 4;
    case 'rgba8unorm':
      return 4;
    default:
      break;
  }
  return 4;
}

export type BackendValues = Float32Array|Int32Array|Uint8Array|Uint8Array[];

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: Map<string, GPUTexture[]> = new Map();
  private usedTextures: Map<string, GPUTexture[]> = new Map();
  private format: GPUTextureFormat = 'r32float';
  private kBytesPerTexel = 4;

  public numBytesUsed = 0;
  public numBytesAllocated = 0;
  // private key = 0;

  constructor(private device: GPUDevice) {}

  getPackedMatrixTextureShapeWidthHeight(
      rows: number, columns: number,
      format: GPUTextureFormat): [number, number] {
    // kBytesPerTexel = 4;
    if (format == 'rgba32float' || format == 'rgba32uint')
      // return [Math.max(1, Math.ceil(rows)), Math.max(1, Math.ceil(columns /
      // 4))];
      return [
        Math.max(1, Math.ceil(rows / 4)), Math.max(1, Math.ceil(columns))
      ];
    else if (format == 'rgba8uint')
      return [rows, columns];
    else
      return [rows, columns];
  }

  //
  private addTexturePadding(
      textureData: Float32Array|Uint32Array,
      width: number,
      height: number,
      bytesPerRow: number,
  ) {
    let textureDataWithPadding =
        new Float32Array(bytesPerRow / this.kBytesPerTexel * height);
    console.log(textureData);
    console.log('width =' + width + ', height =' + height);

    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const dst = x + y * bytesPerRow / this.kBytesPerTexel;
        const src = x + y * width;
        textureDataWithPadding[dst] = textureData[src];
        // console.log('x =' + x + ', y =' + y);
        // console.log(   ' src = ' + src + ',' + textureData[src] + '; dst = '
        // + dst + ',' +      textureDataWithPadding[dst]);
      }
    }
    console.log(textureDataWithPadding);
    return textureDataWithPadding;
  }
  //

  // This will remove padding for data downloading from GPU texture.
  public removeTexturePadding(
      textureDataWithPadding: Float32Array, width: number, height: number) {
    const kBytesPerTexel = 4;
    const [widthTex, heightTex] =
        this.getPackedMatrixTextureShapeWidthHeight(width, height, this.format);
    const bytesPerRow = this.getBytesPerRow(widthTex);
    console.log(heightTex);

    let textureData = new Float32Array(width * height);
    console.log(
        'in removeTexturePadding textureDataWithPadding=' +
        (textureDataWithPadding as Float32Array));
    console.log(textureDataWithPadding.length);
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const src = x + y * bytesPerRow / kBytesPerTexel;
        const dst = x + y * width;
        textureData[dst] = textureDataWithPadding[src];
      }
    }
    console.log('in removeTexturePadding textureData=' + textureData);
    return textureData;
  }

  public getBufferSize(width: number, height: number) {
    /*
    const blockHeight = 1;
    const blockWidth = 1;

    const [widthTex, heightTex] =
        tex_util.getPackedMatrixTextureShapeWidthHeight(
            this.shape[0], this.shape[1], this.format);

    const bytesPerRow = tex_util.getBytesPerRow(widthTex, this.kBytesPerTexel);

    const sliceSize = bytesPerRow * (heightTex / blockHeight - 1) +
        (widthTex / blockWidth) * this.kBytesPerTexel;
    */
    /*
    this.shape = 4096, 128, this.kBytesPerTexel=16
    texture.ts:56  bytesPerRow = 16384, heightTex =128
    texture.ts:125  this.getBufferSize() =2097152

    */

    const [widthTex, heightTex] =
        this.getPackedMatrixTextureShapeWidthHeight(width, height, this.format);
    /*
    console.log(
        ' this.shape = ' + this.shape[0] + ', ' + this.shape[1] +
        ', this.kBytesPerTexel=' + this.kBytesPerTexel);
    */

    const bytesPerRow = this.getBytesPerRow(widthTex);
    /*
    console.log(
        ' bytesPerRow = ' + bytesPerRow + ' widthTex, heightTex =' + widthTex +
        ', ' + heightTex);
    */
    const sliceSize = bytesPerRow * heightTex;
    return sliceSize;
  }

  public writeTexture(
      queue: GPUQueue, texture: GPUTexture, data: BackendValues, width: number,
      height: number) {
    const [widthTex, heightTex] =
        this.getPackedMatrixTextureShapeWidthHeight(width, height, this.format);

    /*
    const texture = this.device.createTexture({
      size: {width: widthTex, height: heightTex, depth: 1},
      format: this.format,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.STORAGE
    });
    */

    const bytesPerRow = this.getBytesPerRow(widthTex);
    console.log(
        'xx writeTexture width tex,  heightTex =' + widthTex + ', ' +
        heightTex + ',   start ' + this.format +
        ', bytesPerRow=' + bytesPerRow);
    /*
            this.queue.writeTexture(
            {texture: info.bufferInfo.texture}, info.values as ArrayBuffer,
            {bytesPerRow: bytesPerRow, rowsPerImage: 1},
            {width: widthTex, height: heightTex, depth: 1});
    */

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
    const src = this.device.createBuffer({
      mappedAtCreation: true,
      size: this.getBufferSize(width, height),  // 640 * 4,  //
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST
    });

    const [widthTex, heightTex] =
        this.getPackedMatrixTextureShapeWidthHeight(width, height, this.format);

    const bytesPerRow = this.getBytesPerRow(widthTex);
    console.log(
        'xx writeTextureWithCopy: widthTex = ' + widthTex +
        '; heightTex = ' + heightTex + ', bytesPerRow=' + bytesPerRow);

    console.log(
        'xx writeTextureWithCopy:  this.getBufferSize() =' +
        this.getBufferSize(width, height));
    // TODO: turn this into type of secondMatrix.
    const matrixDataWithAlignment = this.addTexturePadding(
        matrixData as Float32Array, width, height, bytesPerRow);

    new Float32Array(src.getMappedRange()).set(matrixDataWithAlignment);
    src.unmap();

    const encoder = this.device.createCommandEncoder();
    // TODO: fix the width height.
    // copyBufferToTexture(source, destination, copySize).
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
    const key =
        getTextureKey(width * height * this.kBytesPerTexel, texFormat, usages);
    console.log(key);
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
        this.getPackedMatrixTextureShapeWidthHeight(width, height, texFormat);
    console.log(
        'xx createTexture widthTex, heightTex =' + widthTex + ', ' + heightTex);

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

    // TODO: this is buggy.
    const key =
        getTextureKey(width * height * this.kBytesPerTexel, texFormat, usage);
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
    const kBytesPerTexel = 4;
    this.numBytesUsed -= width * height * kBytesPerTexel;
  }

  getBytesPerRow(width: number) {
    const kTextureBytesPerRowAlignment = 256;
    const alignment = kTextureBytesPerRowAlignment;
    // const kBytesPerTexel = 16;
    const value = this.kBytesPerTexel * width;
    // const bytesPerRow = (value + (alignment - 1)) & ~(alignment - 1);
    const bytesPerRow =
        ((value + (alignment - 1)) & ((~(alignment - 1)) >>> 0)) >>> 0;
    return bytesPerRow;
  }

  getBytesPerTexel(format: GPUTextureFormat): number {
    // kBytesPerTexel = 4;
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
    byteSize: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags) {
  return `${byteSize}_${format}_${usage}`;
}
