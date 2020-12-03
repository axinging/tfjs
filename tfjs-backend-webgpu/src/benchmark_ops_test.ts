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


describeWebGPU('webgputexturepad', () => {
  // pass when wpt = 1.
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
    console.log(result.shape);
    expectArraysClose(await result.data(), [
      0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,
      0, 0, 0, 0, 0, 0,  0,  0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 1, 2,
      3, 4, 5, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 1,  2, 3,
      4, 5, 6, 7, 8, 9,  10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  2,  3, 4,
      5, 6, 7, 8, 9, 10, 1,  2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,
      0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0
    ]);
  });

  // pass when wpt = 1.
  it('padtexture0d2', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2]);
    const result = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 0);
    console.log(await result.data());
    console.log(result.shape);
    expectArraysClose(await result.data(), [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0,
      0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });

  // failed due to sampler not supported.
  it('padtexture0d4', async () => {
    const x = tf.tensor4d([3], [1, 1, 1, 1]);
    const result = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 0);
    console.log(await result.data());
    console.log(result.shape);
    expectArraysClose(await result.data(), [0, 0, 0, 0, 3, 0, 0, 0, 0]);
  });
});

describeWebGPU('webgputexturemaxpool', () => {
  // Pass, MaxPoolWithFilterSizeEqualsOneProgram
  it('maxpooltexf1 x=[1, 3, 3, 2] f=[1,1] s=1 [0] => [0]', async () => {
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

  // Pass, Pool2DProgram
  it('maxpooltexf30 x=[1, 3, 3, 2] f=[1,1] s=1 [0] => [0]', async () => {
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
});

describeWebGPU('webgputextureconv', () => {
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
  });
});

describeWebGPU('webgputexturergba32fbounds', () => {
  it('texturergba32fboundsadd2d365', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
          18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,
          5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,
          4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,
          3,  4,  5,  6,  7,  8,  9,  10, 11, 11, 12, 13, 14, 15, 16, 17, 18,
          1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,  5,  6,  7,  8,
          9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,
          8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,
          7,  8,  9,  10, 11, 10, 20, 30
        ],
        [3, 65]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
          18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,
          5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,
          4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,
          3,  4,  5,  6,  7,  8,  9,  10, 11, 11, 12, 13, 14, 15, 16, 17, 18,
          1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,  5,  6,  7,  8,
          9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,
          8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,
          7,  8,  9,  10, 11, 10, 20, 30
        ],
        [3, 65]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
      36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
      36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 20, 40, 60
    ]);
  });

  it('texturergba32fboundsadd2d644', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11,
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [64, 4]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11,
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [64, 4]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22
    ]);
  });

  it('texturergba32fboundsadd2d464', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11,
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [4, 64]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11,
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [4, 64]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22
    ]);
  });


  it('texturergba32fboundsadd2d264', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [2, 64]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [2, 64]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
      22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,
      28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22
    ]);
  });
});

describeWebGPU('webgputexturebroadcast', () => {
  it('texturergba32fbroadcastadd2d1 2D+2D broadcast each with 1 dim',
     async () => {
       const a = tf.tensor2d([1, 2, 5], [1, 3]);
       const b = tf.tensor2d([7, 3], [2, 1]);
       const res = tf.add(a, b);
       expect(res.shape).toEqual([2, 3]);
       // 8,9,8,4,5,4
       expectArraysClose(await res.data(), [8, 9, 12, 4, 5, 8]);
     });

  it('texturergba32fbroadcastadd2d2 2D+2D broadcast each with 1 dim',
     async () => {
       const a = tf.tensor2d([1, 2, 5], [1, 3]);
       const b = tf.tensor2d([7, 3, 7, 3, 7, 3], [2, 3]);
       const res = tf.add(a, b);
       console.log(await res.data());
       expect(res.shape).toEqual([2, 3]);
       // 8,5,8,4,9,4
       expectArraysClose(await res.data(), [8, 5, 12, 4, 9, 8]);
     });

  it('texturergba32fbroadcastadd2da3 2D+2D broadcast each with 1 dim',
     async () => {
       const a = tf.tensor2d([1, 2, 5, 1, 2, 5], [2, 3]);
       const b = tf.tensor2d([7, 3], [2, 1]);
       const res = tf.add(a, b);
       console.log(await res.data());
       expect(res.shape).toEqual([2, 3]);
       //[8, 9, 5, 4, 5, 5]
       expectArraysClose(await res.data(), [8, 9, 12, 4, 5, 8]);
     });

  it('texturergba32fbroadcastadd2da4 2D+2D broadcast each with 1 dim',
     async () => {
       const a = tf.tensor2d([1, 2, 5, 1], [2, 2]);
       const b = tf.tensor2d([7, 3], [2, 1]);
       const res = tf.add(a, b);
       console.log(await res.data());
       expect(res.shape).toEqual([2, 2]);
       expectArraysClose(await res.data(), [8, 9, 8, 4]);
     });

  it('texturergba32fbroadcastadd1 A + B broadcast 2D + 1D', async () => {
    const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
    const b = tf.tensor1d([1, 2]);
    // const b = tf.tensor2d([1, 2,1,2],[2,2]);

    const result = tf.add(a, b);

    expect(result.shape).toEqual([2, 2]);
    const expected = [2, 4, -2, -2];

    expectArraysClose(await result.data(), expected);
  });

  it('texturergba32fbroadcastaddds2 A + B broadcasting same rank Tensors different shape',
     async () => {
       const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
       // const b = tf.tensor2d([2, 3], [2, 1]);
       const b = tf.tensor2d([2, 3], [1, 2]);

       const result = tf.add(a, b);
       console.log(await result.data());

       expect(result.shape).toEqual([2, 2]);
       const expected = [3, 5, -1, -1];

       expectArraysClose(await result.data(), expected);
     });

  it('texturergba32fxxbroadcastaddds3 A + B broadcasting same rank Tensors different shape',
     async () => {
       const a = tf.tensor2d([1, 2, -3, -4], [2, 2]);
       const b = tf.tensor2d([2, 3], [2, 1]);
       // const b = tf.tensor2d([2, 3], [1, 2]);

       const result = tf.add(a, b);
       console.log(await result.data());

       expect(result.shape).toEqual([2, 2]);
       const expected = [3, 4, 0, -1];

       expectArraysClose(await result.data(), expected);
     });

  it('texturergba32fxxadd2d83', async () => {
    const x = tf.tensor2d(
        [
          1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
          5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [8, 3]);

    const y = tf.tensor2d(
        [
          1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
          5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [8, 3]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 2,  4,  6,  8,
      10, 12, 14, 16, 2,  4,  6,  8,  10, 12, 14, 16
    ]);
  });
});

describeWebGPU('webgputexturescalar', () => {
  it('texturergba32faddscalar c + A', async () => {
    const c = tf.scalar(5);
    const a = tf.tensor1d([1, 2, 3]);

    const result = tf.add(c, a);

    expectArraysClose(await result.data(), [6, 7, 8]);
  });
});

describeWebGPU('webgputextureadd', () => {
  it('texturergba32fadd1d', async () => {
    const x =
        tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

    const y =
        tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);
  });

  it('texturergba32fadd4d1', async () => {
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 2, 2, 2]);

    const y = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 2, 2, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);
  });



  it('texturergba32fadd4dx4442', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [4, 4, 4, 2]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [4, 4, 4, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
      22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,
      28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22
    ]);
  });

  it('texturergba32fadd2d642', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [64, 2]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [64, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
      22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,
      28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22
    ]);
  });

  it('texturergba32fadd2d28', async () => {
    const x = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 8]);

    const y = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [2, 8]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);
  });

  it('texturergba32fadd2d88', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5, 6, 7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3, 4, 5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10
        ],
        [8, 8]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5, 6, 7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3, 4, 5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10
        ],
        [8, 8]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20
    ]);
  });

  it('texturergba32fadd2d653', async () => {
    const x = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
          18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,
          5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,
          4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,
          3,  4,  5,  6,  7,  8,  9,  10, 11, 11, 12, 13, 14, 15, 16, 17, 18,
          1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,  5,  6,  7,  8,
          9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,
          8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,
          7,  8,  9,  10, 11, 10, 20, 30
        ],
        [65, 3]);

    const y = tf.tensor2d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
          18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,
          5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,
          4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,
          3,  4,  5,  6,  7,  8,  9,  10, 11, 11, 12, 13, 14, 15, 16, 17, 18,
          1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,  5,  6,  7,  8,
          9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,
          8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 1,  2,  3,  4,  5,  6,
          7,  8,  9,  10, 11, 10, 20, 30
        ],
        [65, 3]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
      36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
      36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 20, 40, 60
    ]);
  });


  it('texturergba32fadd2d83', async () => {
    const x = tf.tensor2d(
        [
          1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
          5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [8, 3]);

    const y = tf.tensor2d(
        [
          1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4,
          5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8
        ],
        [8, 3]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 2,  4,  6,  8,
      10, 12, 14, 16, 2,  4,  6,  8,  10, 12, 14, 16
    ]);
  });

  it('texturergba32fadd3d882', async () => {
    const x = tf.tensor3d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [8, 8, 2]);

    const y = tf.tensor3d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [8, 8, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
      22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,
      28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22
    ]);
  });

  it('texturergba32fadd4dx2842', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [2, 8, 4, 2]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [2, 8, 4, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
      22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16,
      18, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,
      28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22
    ]);
  });

  it('texturergba32fadd4d4422', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5, 6, 7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3, 4, 5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10
        ],
        [4, 4, 2, 2]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5, 6, 7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3, 4, 5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1, 2, 3, 4,  5,  6,  7,  8,  9,  10
        ],
        [4, 4, 2, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
      30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24,
      26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20
    ]);
  });

  it('texturergba32fadd4d2422', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [2, 4, 2, 2]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [2, 4, 2, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 42, 64
    ]);
  });


  it('texturergba32fadd4d4222', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [4, 2, 2, 2]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [4, 2, 2, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 42, 64
    ]);
  });


  it('texturergba32fadd4d2242', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [2, 2, 4, 2]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [2, 2, 4, 2]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 42, 64
    ]);
  });


  it('texturergba32faddix4444', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11,
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [4, 4, 4, 4]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11,
          1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 1,  2,  3,  4,  5,  6,  7, 8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 1,  2,  3,  4,  5, 6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 1,  2,  3, 4,  5,  6,  7,  8,  9,  10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2,  3,  4,  5,  6,  7,  8,
          9,  1,  2,  3,  4,  5,  6,  7,  8, 9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 1,  2,  3,  4,  5,  6, 7,  8,  9,  10, 11, 12, 13,
          14, 15, 16, 17, 18, 1,  2,  3,  4, 5,  6,  7,  8,  9,  10, 11
        ],
        [4, 4, 4, 4]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
      2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18,
      20, 22, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 2,  4,  6,  8,  10, 12, 14, 16, 18, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 2,  4,  6,  8,  10, 12, 14,
      16, 18, 20, 22
    ]);
  });

  it('texturergba32fadd4d2224', async () => {
    const x = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [2, 2, 2, 4]);

    const y = tf.tensor4d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32
        ],
        [2, 2, 2, 4]);

    const result = tf.add(x, y);
    console.log(await result.data());
    expectArraysClose(await result.data(), [
      2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 42, 64
    ]);
  });
});
describeWebGPU('webgputextureadd', () => {
  it('texture2dadd257x3', async () => {
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

  it('texture2dadd257x37', async () => {
    const size_x = 257;
    const size_y = 37;
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

  it('texture2dadd2x257x9x37', async () => {
    const batch = 2;
    const row = 257;
    const col = 9;
    const channel = 37
    const size_x = batch * row * col;
    const size_y = channel;

    const firstMatrixSize: [number, number, number, number] =
        [batch, row, col, channel];
    const firstMatrix = createFloat32Array(size_x, size_y);
    const secondMatrixSize: [number, number, number, number] =
        [batch, row, col, channel];
    let secondMatrix = createFloat32Array(size_x, size_y);
    let a = tf.tensor4d(firstMatrix, firstMatrixSize);
    let b = tf.tensor4d(secondMatrix, secondMatrixSize);
    console.log(await tf.add(a, b).data());
    compareAddFloat32Array(
        await tf.add(a, b).data(), firstMatrix, secondMatrix, size_x, size_y);
  });

  it('texture4dadd1332', async () => {
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
    const result = tf.add(x, x);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [0, 2, 6, 2, 4, 6, 4, 4, 6, 8, 10, 2, 8, 4, 2, 0, 2, 4]);
  });

  it('texture3dadd332', async () => {
    const x = tf.tensor3d(
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
        [3, 3, 2]);
    const result = tf.add(x, x);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [0, 2, 6, 2, 4, 6, 4, 4, 6, 8, 10, 2, 8, 4, 2, 0, 2, 4]);
    // console.log(result.shape);
  });

  // Pass when run alone.
  it('texture2dadd36', async () => {
    const x = tf.tensor2d(
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
        [3, 6]);
    const result = tf.add(x, x);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [0, 2, 6, 2, 4, 6, 4, 4, 6, 8, 10, 2, 8, 4, 2, 0, 2, 4]);
    // console.log(result.shape);
  });

  it('texturerelu', async () => {
    const a = tf.tensor4d(
        [1, 2, 3, -3, 1, -2, 5, -3, 1, -2, 5, -3, 1, -2, 7, -3, 1, 9],
        [1, 3, 3, 2]);
    const result = tf.relu(a);
    console.log(await result.data());
    expectArraysClose(
        await result.data(),
        [1, 2, 3, 0, 1, 0, 5, 0, 1, 0, 5, 0, 1, 0, 7, 0, 1, 9]);
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
