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
/*
import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {test_util.expectArraysClose} from '../test_util';
*/
import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';

describeWebGPU('fusedxx', () => {
  /*
  it('basic', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2,
  inputDepth]; const outputDepth = 2; const fSize = 1; const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
  outputDepth]);

    const result = tf.fused.conv2d({x, filter: w, strides: stride, pad});
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-5, 2, -11, 5, -17, 8, -23, 11, -29, 14, -35, 17, -41, 20, -47,
  23];

    test_util.expectArraysClose(await result.data(), expected);
  });
  */
  //
  it('basic with relu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'relu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected = [0, 2, 0, 5, 0, 8, 0, 11, 0, 14, 0, 17, 0, 20, 0, 23];
    const re = await result.data();
    console.log(re);
    test_util.expectArraysClose(await result.data(), expected);
  });
  //
  /*
  it('basic with bias', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2,
  inputDepth]; const outputDepth = 2; const fSize = 1; const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
  outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      bias: tf.tensor1d([5, 6])
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [0, 8, -6, 11, -12, 14, -18, 17, -24, 20, -30, 23, -36, 26, -42,
  29];

    test_util.expectArraysClose(await result.data(), expected);
  });
*/
  //
  it('basic with elu', async () => {
    const inputDepth = 2;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 2;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inShape);
    const w =
        tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.fused.conv2d({
      x,
      filter: w,
      strides: stride,
      pad,
      dataFormat: 'NHWC',
      dilations: [1, 1],
      activation: 'elu'
    });
    expect(result.shape).toEqual([2, 2, 2, 2]);
    const expected =
        [-0.99326, 2, -1, 5, -1, 8, -1, 11, -1, 14, -1, 17, -1, 20, -1, 23];
    const re = await result.data();
    console.log(re);
    test_util.expectArraysClose(await result.data(), expected);
  });
  //
  /*
    it('basic with prelu', async () => {
      const inputDepth = 2;
      const inShape: [number, number, number, number] = [2, 2, 2,
    inputDepth]; const outputDepth = 2; const fSize = 1; const pad = 0;
      const stride = 1;

      const x = tf.tensor4d(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    inShape); const alpha = tf.tensor3d([0.25, 0.75], [1, 1, 2]); const w
    = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
    outputDepth]);

      const result = tf.fused.conv2d({
        x,
        filter: w,
        strides: stride,
        pad,
        dataFormat: 'NHWC',
        dilations: [1, 1],
        activation: 'prelu',
        preluActivationWeights: alpha
      });
      expect(result.shape).toEqual([2, 2, 2, 2]);
      const expected = [
        -1.25, 2, -2.75, 5, -4.25, 8, -5.75, 11, -7.25, 14, -8.75, 17,
    -10.25, 20, -11.75, 23
      ];

      test_util.expectArraysClose(await result.data(), expected);
    });

    it('basic with broadcasted bias and relu', async () => {
      const inputDepth = 2;
      const inShape: [number, number, number, number] = [2, 2, 2,
    inputDepth]; const outputDepth = 2; const fSize = 1; const pad = 0;
      const stride = 1;

      const x = tf.tensor4d(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    inShape); const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize,
    inputDepth, outputDepth]);

      const result = tf.fused.conv2d({
        x,
        filter: w,
        strides: stride,
        pad,
        dataFormat: 'NHWC',
        dilations: [1, 1],
        bias: tf.scalar(5),
        activation: 'relu'
      });
      expect(result.shape).toEqual([2, 2, 2, 2]);
      const expected = [0, 7, 0, 10, 0, 13, 0, 16, 0, 19, 0, 22, 0, 25, 0,
    28];

      test_util.expectArraysClose(await result.data(), expected);
    });
    */
});
