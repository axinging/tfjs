/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {Rank} from '../types';
*/
import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';


describeWebGPU('depthwiseConv2D', () => {
  it('input=1x3x3x1,f=2,s=1,d=1,p=valid,chMul=1', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.230664, 0.987388, 0.0685208, 0.419224, 0.887861, 0.731641,
          0.0741907, 0.409265, 0.351377
        ],
        [1, 3, 3, inDepth]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 2, 2, 1]);
    const expected = [1.07022, 1.03167, 0.67041, 0.778863];
    test_util.expectArraysClose(await result.data(), expected);
  });

  it('input=1x5x5x1,f=3,s=1,d=1,p=valid,chMul=1', async () => {
    const fSize = 3;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0.149194, 0.089009, 0.654891, 0.083324, 0.537043, 0.644331, 0.563037,
          0.211859, 0.633501, 0.186427, 0.777034, 0.50001,  0.607341, 0.95303,
          0.696479, 0.050387, 0.62045,  0.728049, 0.028043, 0.437009, 0.712881,
          0.741935, 0.974474, 0.621102, 0.171411
        ],
        [1, 5, 5, inDepth]);
    const w = tf.tensor4d(
        [
          0.125386, 0.975199, 0.640437, 0.281895, 0.990968, 0.347208, 0.889702,
          0.180695, 0.691992
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 3, 3, 1]);
    const expected = [
      2.540022, 2.505885, 2.454062, 2.351701, 2.459601, 3.076421, 3.29848,
      3.437421, 2.93419
    ];
    test_util.expectArraysClose(await result.data(), expected);
  });

  it('depthwisexx,input=1x5x5x1,f=3,s=1,d=1,p=same,chMul=1', async () => {
    const fSize = 3;
    const pad = 'same';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
          6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 1
        ],
        [1, 5, 5, inDepth]);
    const w = tf.tensor4d(
        [0, 1, 2, 0, 0, 1, 1, 0, 0],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad);
    expect(result.shape).toEqual([1, 5, 5, 1]);
    console.log('..................');
    console.log(await result.data());

    /*
   const arr = await result.data();
   for (let i = 0; i < arr.length; i++) {
     console.log(arr[i] + ',');
   }
   */
    const expected = [
      1, 7, 9,  4,  1,  8,  8, 13, 18, 10, 21, 12, 10,
      8, 6, 13, 23, 21, 12, 2, 5,  9,  13, 15, 5
    ];
    test_util.expectArraysClose(await result.data(), expected);
  });
  /*
    it('input=1x5x5x1,f=3,s=1,d=1,p=valid,chMul=1', async () => {
      const xSize = 7;
      const fSize = 7;
      const pad = 'valid';
      const stride = 1;
      const chMul = 1;
      const inDepth = 1;

      const x = tf.tensor4d(
          [
            0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2,
            3, 4, 5, 6, 0, 1, 2, 1, 2, 1, 0, 0, 1, 1, 1, 1, 1,
            2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0
          ],
          [1, xSize, xSize, inDepth]);
      const w = tf.tensor4d(
          [
            0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2,
            3, 4, 5, 6, 0, 1, 2, 1, 2, 1, 0, 0, 1, 1, 1, 1, 1,
            2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0
          ],
          [fSize, fSize, inDepth, chMul],
      );

      const result = tf.depthwiseConv2d(x, w, stride, pad);
      console.log(await result.data());
      expect(result.shape).toEqual([1, 3, 3, 1]);
      const expected = [
        2.540022, 2.505885, 2.454062, 2.351701, 2.459601, 3.076421, 3.29848,
        3.437421, 2.93419
      ];
      test_util.expectArraysClose(await result.data(), expected);
    });
    */
});
