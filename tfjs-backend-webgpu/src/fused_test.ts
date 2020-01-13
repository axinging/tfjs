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
import {test_util.test_util.expectArraysClose} from '../test_util';
*/
import * as tf from '@tensorflow/tfjs-core';
import {Rank, test_util} from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';

describeWebGPU('fusedxx', () => {
  //
  it('gradient with clones input=[3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number] = [3, 3, inputDepth];
    const filterSize = 2;
    const stride = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.ones<Rank.R4>(filterShape);

    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor3d([3, 1, 2, 0], [2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor3D, filter: tf.Tensor4D) =>
            x.clone().conv2d(filter.clone(), stride, pad).clone());
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    test_util.expectArraysClose(await dx.data(), [3, 4, 1, 5, 6, 1, 2, 2, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    test_util.expectArraysClose(await dfilter.data(), [13, 19, 31, 37]);
  });

  /*
  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const stride = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.ones<Rank.R4>(filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) => x.conv2d(filter, stride, pad));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    test_util.expectArraysClose(
        await dx.data(),
        [3, 4, 1, 5, 6, 1, 2, 2, 0, 3, 4, 1, 5, 6, 1, 2, 2, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    test_util.expectArraysClose(
        await dfilter.data(), [13 * 2, 19 * 2, 31 * 2, 37 * 2]);
  });

  it('gradient x=[1,1,3,3] f=[2,2,1,1] s=1 p=0 NCHW', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [1, inputDepth, 3, 3];
    const filterSize = 2;
    const stride = 1;
    const pad = 0;
    const dataFormat = 'NCHW';

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.ones<Rank.R4>(filterShape);

    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0], [1, 1, 2, 2]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            x.conv2d(filter, stride, pad, dataFormat));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    test_util.expectArraysClose(await dx.data(), [3, 4, 1, 5, 6, 1, 2, 2, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    test_util.expectArraysClose(await dfilter.data(), [13, 19, 31, 37]);
  });

  it('gradient x=[2,1,3,3] f=[2,2,1,1] s=1 p=0 NCHW', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, inputDepth, 3, 3];
    const filterSize = 2;
    const stride = 1;
    const pad = 0;
    const dataFormat = 'NCHW';

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.ones<Rank.R4>(filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 1, 2, 2]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            x.conv2d(filter, stride, pad, dataFormat));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    test_util.expectArraysClose(
        await dx.data(),
        [3, 4, 1, 5, 6, 1, 2, 2, 0, 3, 4, 1, 5, 6, 1, 2, 2, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    test_util.expectArraysClose(await dfilter.data(), [26, 38, 62, 74]);
  });
*/

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor4D) =>
            tf.fused.conv2d({x, filter, strides, pad}));
    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    test_util.expectArraysClose(
        await dx.data(),
        [-3, 2, 1, -8, 1.5, 0.5, -4, 1, 0, -3, 2, 1, -8, 1.5, 0.5, -4, 1, 0]);

    expect(dfilter.shape).toEqual(filterShape);
    test_util.expectArraysClose(await dfilter.data(), [26, 38, 62, 74]);
  });

  it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number, number] = [2, 3, 3, inputDepth];
    const filterSize = 2;
    const strides = 1;
    const pad = 0;

    const filterShape: [number, number, number, number] =
        [filterSize, filterSize, inputDepth, outputDepth];
    const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
    const bias = tf.ones([2, 2, 2, 1]);

    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
    const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

    const fusedGrads =
        tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
          x,
          filter: w,
          strides,
          pad,
          dataFormat: 'NHWC',
          dilations: [1, 1],
          bias: b
        }));
    const [dxFused, dfilterFused, dbiasFused] =
        fusedGrads([x, filter, bias], dy);

    const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
      const conv = tf.conv2d(x, filter, strides, pad);
      const sum = tf.add(conv, bias);
      return sum;
    });
    const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

    test_util.expectArraysClose(await dxFused.array(), await dx.array());
    test_util.expectArraysClose(
        await dfilterFused.array(), await dfilter.array());
    test_util.expectArraysClose(await dbiasFused.array(), await dbias.array());
  });
  /*
    it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and relu',
       async () => {
         const inputDepth = 1;
         const outputDepth = 1;
         const inputShape: [number, number, number, number] =
             [2, 3, 3, inputDepth];
         const filterSize = 2;
         const strides = 1;
         const pad = 0;

         const filterShape: [number, number, number, number] =
             [filterSize, filterSize, inputDepth, outputDepth];
         const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
         const bias = tf.ones([2, 2, 2, 1]);

         const x = tf.tensor4d(
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    inputShape); const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

         const fusedGrads =
             tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
               x,
               filter: w,
               strides,
               pad,
               dataFormat: 'NHWC',
               dilations: [1, 1],
               bias: b,
               activation: 'relu'
             }));
         const [dxFused, dfilterFused, dbiasFused] =
             fusedGrads([x, filter, bias], dy);

         const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
           const conv = tf.conv2d(x, filter, strides, pad);
           const sum = tf.add(conv, bias);
           return tf.relu(sum);
         });
         const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

         test_util.expectArraysClose(await dxFused.array(), await dx.array());
         test_util.expectArraysClose(
             await dfilterFused.array(), await dfilter.array());
         test_util.expectArraysClose(
             await dbiasFused.array(), await dbias.array());
       });

    it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias and elu', async ()
    => { const inputDepth = 1; const outputDepth = 1; const inputShape: [number,
    number, number, number] = [2, 3, 3, inputDepth]; const filterSize = 2; const
    strides = 1; const pad = 0;

      const filterShape: [number, number, number, number] =
          [filterSize, filterSize, inputDepth, outputDepth];
      const filter = tf.tensor4d([-1, 1, -2, 0.5], filterShape);
      const bias = tf.ones([2, 2, 2, 1]);

      const x = tf.tensor4d(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], inputShape);
      const dy = tf.tensor4d([3, 1, 2, 0, 3, 1, 2, 0], [2, 2, 2, 1]);

      const fusedGrads =
          tf.grads((x: tf.Tensor4D, w: tf.Tensor4D, b) => tf.fused.conv2d({
            x,
            filter: w,
            strides,
            pad,
            dataFormat: 'NHWC',
            dilations: [1, 1],
            bias: b,
            activation: 'elu'
          }));
      const [dxFused, dfilterFused, dbiasFused] =
          fusedGrads([x, filter, bias], dy);

      const grads = tf.grads((x: tf.Tensor4D, filter: tf.Tensor4D, bias) => {
        const conv = tf.conv2d(x, filter, strides, pad);
        const sum = tf.add(conv, bias);
        return tf.elu(sum);
      });
      const [dx, dfilter, dbias] = grads([x, filter, bias], dy);

      test_util.expectArraysClose(await dxFused.array(), await dx.array());
      test_util.expectArraysClose(
          await dfilterFused.array(), await dfilter.array());
      test_util.expectArraysClose(await dbiasFused.array(), await
    dbias.array());
    });
    */
});
