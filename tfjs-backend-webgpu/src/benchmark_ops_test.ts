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


describeWebGPU('padtexture0d', () => {
  it('padtexture0d1', async () => {
    const x = tf.tensor4d(
        [
          1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
          6, 7, 8, 9, 10, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10,
          1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 4, 5
        ],
        [1, 3, 3, 5]);
    const result = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 0);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,
      0, 0, 0, 0, 0, 0,  0,  0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 1, 2,
      3, 4, 5, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 1,  2, 3,
      4, 5, 6, 7, 8, 9,  10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  2,  3, 4,
      5, 6, 7, 8, 9, 10, 1,  2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,
      0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0
    ]);
  });

  it('padtexture0d2', async () => {
    const x = tf.tensor4d(
        [
          1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
          6, 7, 8, 9, 10, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10,
          1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 4, 5
        ],
        [1, 3, 3, 5]);
    const result = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 0);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });
});

describeWebGPU('Ops conv2dbenchmarks', () => {
  // Pass
  it('maxpooltexf1 x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 2]);
    // const filter = [1, 1];
    // const stride = [2, 2];
    const pad = 'same';
    const result = tf.maxPool(x, [1, 1], [2, 2], pad);
    console.log(result.shape);
    // 1,2,2,2
    console.log(await result.data());

    expectArraysClose(await result.data(), [1, 2, 5, 6, 4, 5, 8, 9]);
  });

  // Pass
  it('maxpooltexf3 x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 2]);
    // const filter = [1, 1];
    // const stride = [2, 2];
    const pad = 'same';
    const result = tf.maxPool(x, [3, 3], [2, 2], pad);
    console.log(result.shape);
    // 1,2,2,2
    console.log(await result.data());

    expectArraysClose(await result.data(), [8, 9, 8, 6, 8, 9, 8, 9]);
  });

  // Pass
  it('maxpooltexf3v x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 2]);
    // const filter = [1, 1];
    // const stride = [2, 2];
    const pad = 'valid';
    const result = tf.maxPool(x, [3, 3], [2, 2], pad);
    console.log(result.shape);
    // 1,2,2,2
    console.log(await result.data());

    expectArraysClose(await result.data(), [8, 9]);
  });

  // SUCCESS
  it('conmm1 conv2d conv2d x=[2,2,2,2] f=[1,1,2,2] s=1 d=1 p=0', async () => {
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

  // Pass
  it('conmm2 x=[3,3,2] f=[2,2,2,1] s=1 d=1 p=valid', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [3, 3, 2]);
    const w = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8], [2, 2, 2, 1]);

    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    console.log(resultData);
    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(resultData, new Float32Array([25.6, 53.5, 157.0, 220.9]));
  });


  // Fail, 4D squeeze to 3D input not supported
  it('conmm21 x=[1,3,3,2] f=[2,2,2,1] s=1 d=1 p=valid', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [1, 3, 3, 2]);
    const w = tf.tensor4d([1, 2, 5, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    console.log(resultData);
    console.log(result.shape);
    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(resultData, new Float32Array([262, 545, 1588, 2249]));
  });

  // Pass
  it('conmm22 x=[1,3,3,2] f=[2,2,2,2] s=1 d=1 p=valid', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [1, 3, 3, 2]);
    const w = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 2]);

    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    console.log(resultData);
    console.log(result.shape);
    expect(result.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(
        resultData,
        new Float32Array([196, 240, 431, 518, 1126, 1380, 1649, 2018]));
    // 25.600000381469727,53.5,157,220.89999389648438
  });

  it('conmm3 x=[2,2,2,1] f=[1,1,1,1] s=1 d=1 p=0', async () => {
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

  it('conmm x=[4,2,1] f=[4,2,1,1] s=1 d=1 p=same', async () => {
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


  it('conv4dmm3x3c1', async () => {
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

  // pass when run alone
  it('conv4dmem3x3b', async () => {
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


  // Pass when alone
  it('conv4dmm5x5f3c2', async () => {
    const x = tf.tensor4d(
        [
          0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3,
          4, 5, 1, 4, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ],
        [1, 5, 5, 2]);
    const f = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 3
        ],
        [3, 3, 2, 2]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [
      243, 229, 226, 172, 169, 175, 250, 265, 244, 245, 182, 149, 295, 249, 264,
      267, 278, 251
    ]);
  });

  // Pass
  it('conv4dmm4x4f3c2', async () => {
    const x = tf.tensor4d(
        [
          0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2,
        ],
        [1, 4, 4, 2]);
    const f = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 3
        ],
        [3, 3, 2, 2]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(
        await result.data(), [223, 207, 260, 225, 219, 229, 214, 226]);
  });

  // Pass when alone
  it('conv4dmem5x5f3c4', async () => {
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
      496, 526, 484, 433, 446, 442, 483, 479, 397, 396, 395, 376,
      525, 499, 518, 528, 502, 515, 519, 478, 469, 460, 451, 469,
      424, 433, 415, 415, 526, 500, 465, 511, 488, 551, 569, 551
    ]);
  });

  // Pass
  it('conv4dmm5x5f5', async () => {
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
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
          8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3,
          4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1,
        ],
        [5, 5, 4, 4]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [1333, 1341, 1288, 1253]);
  });

  // Pass
  it('conv4dmm7x7', async () => {
    const x = tf.tensor4d(
        [
          0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0, 1, 2, 3,
          4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2, 1, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4,
          2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 1, 4, 2, 1, 0, 1, 2, 3,
          4, 5, 6, 7, 8, 9, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5, 1, 4, 2,
          1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2, 3, 4, 5,
          1, 4, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 1, 2, 3, 2, 2,
          3, 4, 5, 1, 4, 2, 1, 0, 1, 2, 3, 4, 5, 1, 3, 1, 2, 1, 3, 1,
        ],
        [1, 7, 7, 4]);
    const f = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
          8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3,
          4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
          8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3,
          4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
          8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3,
          4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
          5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
          1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5,
          6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
          2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6,
          7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2,
          3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
          8, 0,
        ],
        [7, 7, 4, 4]);

    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [2503, 2550, 2453, 2401]);
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

  // Pass
  it('conv2df1', async () => {
    const x = tf.tensor4d(
        [
          2, 3, 2, 0, 0, 0, 1, 3, 1, 2, 3, 2, 0, 0, 0, 1,
          3, 1, 2, 3, 2, 0, 0, 0, 1, 3, 1, 2, 3, 2, 0, 0
        ],
        [1, 4, 4, 2]);
    // const x = tf.tensor4d([1, 1, 3, 1, 2, 3, 2, 1, 2], [1, 3, 3, 1]);
    const f = tf.tensor4d([1, 2, 3, 4], [1, 1, 2, 2]);
    const result = tf.conv2d(x, f, [1, 1], 'valid', 'NHWC');
    console.log(await result.data());

    expectArraysClose(await result.data(), [
      11, 16, 2,  4,  0, 0, 10, 14, 7,  10, 9, 14, 0, 0,  3, 4,
      6,  10, 11, 16, 2, 4, 0,  0,  10, 14, 7, 10, 9, 14, 0, 0
    ]);

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

  it('conv4dmmaddtexture', async () => {
    const size_x = 257;
    const size_y = 3;
    const firstMatrixSize: [number, number] = [size_x, size_y];
    const firstMatrix = createFloat32Array(size_x, size_y);
    const secondMatrixSize: [number, number] = [size_x, size_y];
    let secondMatrix = createFloat32Array(size_x, size_y);
    let a = tf.tensor2d(firstMatrix, firstMatrixSize);
    let b = tf.tensor2d(secondMatrix, secondMatrixSize);
    console.log(await tf.add(a, b).data());
    compareAddFloat32Array(
        await tf.add(a, b).data(), firstMatrix, secondMatrix, size_x, size_y);
  });
});

function createFloat32Array(w: number, h: number) {
  const matrix = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    matrix[i] = i + 1;  // Math.random();
  }
  return matrix;
}

function compareAddFloat32Array(
    result: any, firstMatrix: any, secondMatrix: any, w: number, h: number) {
  for (let i = 0; i < w * h; i++) {
    if (Math.abs(result[i] - (firstMatrix[i] + secondMatrix[i])) > 0.01) {
      console.error(name + ' mismatch at ' + i);
      return i + 1;
    }
  }
  return 0;
}

/*
    function createFloat32Array(w, h) {
      const matrix = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) {
        matrix[i] = i + 1;  // Math.random();
      }
      return matrix;
    }

    function compareAddFloat32Array(result, firstMatrix, secondMatrix, w, h) {
      for (let i = 0; i < w * h; i++) {
        if (Math.abs(result[i] - (firstMatrix[i] + secondMatrix[i])) > 0.01) {
          console.log(name + ' mismatch at ' + i);
          return i+1;
        }
      }
      return 0;
    }

    const size_x = 257;
    const size_y = 3;
    const firstMatrixSize= [size_x, size_y];
    const firstMatrix = createFloat32Array(size_x, size_y);
    const secondMatrixSize= [size_x, size_y];
    let secondMatrix = createFloat32Array(size_x, size_y);
    let a = tf.tensor2d(firstMatrix, firstMatrixSize);
    let b = tf.tensor2d(secondMatrix, secondMatrixSize);
    console.log(await tf.add(a, b).data());
    compareAddFloat32Array(await tf.add(a, b).data(),
   firstMatrix,secondMatrix,size_x,  size_y);

*/
