/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import { describeWebGPU } from './test_util';
import { test_util } from '@tensorflow/tfjs-core';

describeWebGPU('greaterEqual', () => {
  // Tensor1D - int32 will lead to NaNs in Tensor1D - float32 fail
  // Tensor1D:
  it('Tensor1D - int32', async () => {
    let a = tf.tensor1d([1, 4, 5], 'int32');
    let b = tf.tensor1d([2, 3, 5], 'int32');
    let res = tf.greaterEqual(a, b);

    expect(res.dtype).toBe('bool');
    test_util.expectArraysClose(await res.data(), [0, 1, 1]);

    a = tf.tensor1d([2, 2, 2], 'int32');
    b = tf.tensor1d([2, 2, 2], 'int32');
    res = tf.greaterEqual(a, b);

    expect(res.dtype).toBe('bool');
    test_util.expectArraysClose(await res.data(), [1, 1, 1]);

    a = tf.tensor1d([0, 0], 'int32');
    b = tf.tensor1d([3, 3], 'int32');
    res = tf.greaterEqual(a, b);

    expect(res.dtype).toBe('bool');
    test_util.expectArraysClose(await res.data(), [0, 0]);
  });

  it('NaNs in Tensor1D - float32', async () => {
   for (let i=0; i<100; i++) {
    const a = tf.tensor1d([1.1, NaN, 2.1], 'float32');
    const b = tf.tensor1d([2.1, 3.1, NaN], 'float32');
    const res = tf.greaterEqual(a, b);

    expect(res.dtype).toBe('bool');
    test_util.expectArraysClose(await res.data(), [0, 0, 0]);
   }
  });
});
