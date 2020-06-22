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

import {logKernelTime, startLog} from './profiler';
import {describeWebGPU} from './test_util';

let kernels: any = [];
describeWebGPU('timestamp', () => {
  // Fail reason for NaNs in Tensor1D - float32
  // Tensor1D:
  it('timestamp', async () => {
    await tf.ready()
    tf.ENV.set('DEBUG', true);
    {
      startLog(kernels);
      let a = tf.tensor1d([1, 4, 5], 'int32');
      let b = tf.tensor1d([2, 3, 5], 'int32');
      let res = tf.add(a, b);
      console.log(await res);
    }
    {
      let a = tf.tensor1d([1, 4, 5], 'int32');
      let b = tf.tensor1d([2, 3, 5], 'int32');
      let res = tf.mul(a, b);
      console.log(await res);
    }
    await logKernelTime(kernels);
  });
});
