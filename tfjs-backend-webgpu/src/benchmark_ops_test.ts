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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';
const expectArraysClose = test_util.expectArraysClose;

/*
function createFloat32Array(w: number, h: number) {
  const matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] =
        i % 5;  // Math.random();  // tf.randomUniform(shape, 0, 2.5);//0.01*i;
  }
  return matrix;
}

async function doTest(
    xShape: [number, number, number, number],
    fShape: [number, number, number, number], stride: [number, number],
    pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') {
  const x = tf.tensor4d(
      createFloat32Array(xShape[0] * xShape[1] * xShape[2], xShape[3]), xShape);
  const f = tf.tensor4d(
      createFloat32Array(fShape[0] * fShape[1] * fShape[2], fShape[3]), fShape);
  const result = tf.conv2d(x, f, stride, pad, format);
  console.log(await result.data());
  return await result.data();
}
*/

describeWebGPU('Ops conv2dbenchmarks', () => {
  // SUCCESS
  it('conv2d conv2d x=[2,2,2,2] f=[1,1,2,2] s=1 d=1 p=0', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    // row is 0,1,2,3,4,5,6,7.
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-5, 2, -11, 5, -17, 8, -23, 11, -29, 14, -35, 17, -41, 20, -47, 23];

    expectArraysClose(await result.data(), expected);
  });
  // FAILED
  it('x=[3,3,2] f=[2,2,2,1] s=1 d=1 p=valid', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [3, 3, 2]);
    const w = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8], [2, 2, 2, 1]);

    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(resultData, new Float32Array([25.6, 53.5, 157.0, 220.9]));
  });

  it('conmm x=[2,2,2,1] f=[1,1,1,1] s=1 d=1 p=0', async () => {
    const inputDepth = 1;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], inShape);
    const w = tf.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    expect(result.shape).toEqual([2, 2, 2, 1]);
    const expected = [2, 4, 6, 8, 10, 12, 14, 16];

    expectArraysClose(await result.data(), expected);
  });

  it('x=[2,1,2,2] f=[1,1,1,1] s=1 d=1 p=0 NCHW', async () => {
    const inputDepth = 1;
    const inShape: [number, number, number, number] = [2, inputDepth, 2, 2];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NCHW';

    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], inShape);
    const w = tf.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat);
    expect(result.shape).toEqual([2, 1, 2, 2]);
    const expected = [2, 4, 6, 8, 10, 12, 14, 16];

    expectArraysClose(await result.data(), expected);
  });

  it('x=[4,2,1] f=[4,2,1,1] s=1 d=1 p=same', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2, inputDepth]);
    const w =
        tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);

    const resultData = await result.data();
    expect(result.shape).toEqual([4, 2, 1]);
    expectArraysClose(resultData, [133, 66, 200, 102, 108, 58, 56, 58]);
  });

  it('x=[4,2,1] f=[4,2,1,1] s=1 d=1 p=explicit', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const pad =
        [[0, 0], [1, 2], [0, 1], [0, 0]] as tf.backend_util.ExplicitPadding;
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2, inputDepth]);
    const w =
        tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);

    const resultData = await result.data();
    expect(result.shape).toEqual([4, 2, 1]);
    expectArraysClose(resultData, [133, 66, 200, 102, 108, 58, 56, 58]);
  });

  it('x=[2,2,1] f=[2,2,1,1] s=1 d=1 p=same', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        tf.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(resultData, new Float32Array([20, 26, 13, 12]));
  });

  it('x=[2,2,1] f=[1,1,1,2] s=1 d=1 p=0', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);

    expectArraysClose(await result.data(), [2, 4, 6, 8]);
  });


  it('conv4dmm3x3all1', async () => {
    const x = tf.tensor4d(
        [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
        ],
        [1, 3, 3, 2]);
    const f = tf.tensor4d(
        [
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
        ],
        [3, 3, 2, 2]);
    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());
    expectArraysClose(await result.data(), [18, 19]);
  });


  it('conv4dmm3x3x1', async () => {
    const x = tf.tensor4d(
        [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          2,
        ],
        [1, 3, 3, 2]);
    const f = tf.tensor4d(
        [
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
        ],
        [3, 3, 2, 2]);
    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());
    expectArraysClose(await result.data(), [19, 23]);
  });

  it('conv4dmm3x3f1', async () => {
    const x = tf.tensor4d(
        [
          0,
          1,
          3,
          1,
          2,
          3,
          2,
          2,
          3,
          4,
          5,
          1,
          4,
          2,
          1,
          0,
          1,
          2,
        ],
        [1, 3, 3, 2]);
    const f = tf.tensor4d(
        [
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        [3, 3, 2, 2]);
    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());
    expectArraysClose(await result.data(), [37, 37]);
  });


  it('conv4dmm3x3a', async () => {
    const x = tf.tensor4d(
        [
          0,
          1,
          3,
          1,
          2,
          3,
          2,
          2,
          3,
          4,
          5,
          1,
          4,
          2,
          1,
          0,
          1,
          2,
        ],
        [1, 3, 3, 2]);
    const f = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 1, 1, 1, 0, 1, 2, 3, 4, 5, 1, 1, 1,
          0, 1, 2, 3, 4, 5, 1, 1, 1, 0, 1, 2, 3, 4, 4, 5, 1, 1,
        ],
        [3, 3, 2, 2]);
    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());
    expectArraysClose(await result.data(), [66, 75]);
  });


  it('conv4dmm3x3b', async () => {
    const x = tf.tensor4d(
        [
          0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5,
        ],
        [1, 3, 3, 4]);
    const f = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [3, 3, 4, 4]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [439, 445, 433, 412]);
  });

  it('conv4dmm5x5', async () => {
    const x = tf.tensor4d(
        [
          0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4,
          5, 1, 4, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2,
          3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ],
        [1, 5, 5, 4]);
    const f = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [3, 3, 4, 4]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [
      496, 526, 484, 433, 446, 442, 483, 479, 397, 396, 395,
      376, 525, 499, 518, 528, 502, 515, 519, 478, 469, 460,
      451, 469, 424, 433, 415, 415, 526, 500, 465, 511, 488
    ]);
  });

  it('conv2dmm5x5', async () => {
    const x = tf.tensor4d(
        [
          0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4,
          2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ],
        [1, 5, 5, 1]);
    const f = tf.tensor4d([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3, 1, 1]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(
        await result.data(), [103, 84, 89, 68, 81, 101, 151, 183, 212]);
  });

  it('conv2dmm4x4', async () => {
    const x = tf.tensor4d(
        [0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0], [1, 4, 4, 1]);
    const f = tf.tensor4d([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3, 1, 1]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [121, 99, 103, 62]);
  });

  it('conv2dnn', async () => {
    /*
    const doTest = async (
        xShape: [number, number, number, number],
        fShape: [number, number, number, number], stride: [number, number],
        pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') => {
      const arrX = new Array(nRep).fill(0);
      const arrF = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const x = arrX.map((x) => tf.randomNormal<tf.Rank.R4>(xShape));
      const f = arrF.map((x) => tf.randomNormal<tf.Rank.R4>(fShape));
      await time(
          (r) => {
            res[r] = tf.conv2d(x[r], f[r], stride, pad, format);
            return [];
          },
          async () => {
            await res[res.length - 1].data();
            for (const t of res) {
              t.dispose();
            }
          },
          false, nTrail, nRep);
      x.forEach(t => t.dispose());
      f.forEach(t => t.dispose());
    };
    */

    // FAIL
    /*
    const result =
        await doTest([1, 4, 4, 1], [3, 3, 1, 1], [1, 1], 'valid', 'NHWC');
    expectArraysClose(result, [28, 19, 47, 28]);
    */

    const x = tf.tensor4d([0, 1, 3, 1, 2, 3, 2, 0, 0], [1, 3, 3, 1]);
    // const x = tf.tensor4d([1, 1, 3, 1, 2, 3, 2, 1, 2], [1, 3, 3, 1]);
    const f = tf.tensor4d([0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3, 1, 1]);
    // const f = tf.tensor4d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 1, 1]);
    // change f
    // 0 ,,0
    // 1,,1
    // change x:
    //
    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());
    /*
     const x = tf.tensor4d([0, 1, 3, 1, 2, 3, 2, 0, 0], [1, 3, 3, 1]);
     const f = tf.tensor4d([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 1, 1]);
     const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
     console.log(await result.data());
     */
    expectArraysClose(await result.data(), [45]);

    /*
    // PASS
    await doTest([1, 8, 8, 256], [1, 1, 256, 1024], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 512], [1, 1, 512, 2048], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 1024], [1, 1, 1024, 2048], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 1024], [1, 1, 1024, 512], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 2048], [1, 1, 2048, 17], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 2048], [1, 1, 2048, 32], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 2048], [1, 1, 2048, 34], [1, 1], 'same', 'NHWC');
    await doTest([1, 8, 8, 2048], [1, 1, 2048, 512], [1, 1], 'same', 'NHWC');
    await doTest([1, 15, 15, 128], [1, 1, 128, 512], [1, 1], 'same', 'NHWC');
    await doTest([1, 15, 15, 256], [1, 1, 256, 1024], [1, 1], 'same', 'NHWC');
    await doTest([1, 15, 15, 512], [1, 1, 512, 256], [1, 1], 'same', 'NHWC');
    await doTest([1, 15, 15, 512], [1, 1, 512, 1024], [1, 1], 'same', 'NHWC');
    await doTest([1, 15, 15, 1024], [1, 1, 1024, 256], [1, 1], 'same', 'NHWC');
    await doTest([1, 29, 29, 64], [1, 1, 64, 256], [1, 1], 'same', 'NHWC');
    await doTest([1, 29, 29, 128], [1, 1, 128, 512], [1, 1], 'same', 'NHWC');
    await doTest([1, 29, 29, 256], [1, 1, 256, 128], [1, 1], 'same', 'NHWC');
    await doTest([1, 29, 29, 256], [1, 1, 256, 512], [1, 1], 'same', 'NHWC');
    await doTest([1, 29, 29, 512], [1, 1, 512, 128], [1, 1], 'same', 'NHWC');
    await doTest([1, 57, 57, 64], [1, 1, 64, 64], [1, 1], 'same', 'NHWC');
    await doTest([1, 57, 57, 64], [1, 1, 64, 256], [1, 1], 'same', 'NHWC');
    // PASS
    await doTest([1, 57, 57, 256], [1, 1, 256, 64], [1, 1], 'same', 'NHWC');

    // FAIL
    await doTest([1, 8, 8, 512], [3, 3, 512, 512], [1, 1], 'same', 'NHWC');
    await doTest([1, 15, 15, 256], [3, 3, 256, 256], [1, 1], 'same', 'NHWC');

    // FAIL
    await doTest([1, 17, 17, 256], [3, 3, 256, 256], [2, 2], 'valid', 'NHWC');
    await doTest([1, 29, 29, 128], [3, 3, 128, 128], [1, 1], 'same', 'NHWC');
    await doTest([1, 31, 31, 128], [3, 3, 128, 128], [2, 2], 'valid', 'NHWC');
    await doTest([1, 57, 57, 64], [3, 3, 64, 64], [1, 1], 'same', 'NHWC');
    await doTest([1, 59, 59, 64], [3, 3, 64, 64], [2, 2], 'valid', 'NHWC');
    await doTest([1, 231, 231, 3], [7, 7, 3, 64], [2, 2], 'valid', 'NHWC');
    */
  });
});
