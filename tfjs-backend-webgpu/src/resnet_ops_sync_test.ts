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
import {describeWebGPU} from './test_util';

export function getMeanAndMin(kernels: any, trials: number, reps: number) {
  let sum = 0;
  let min = Number.MAX_VALUE;
  kernels.forEach((kernel: {name: string|number; kernelTimeMs: any;}) => {
    sum += kernel.kernelTimeMs;
    if (kernel.kernelTimeMs < min) {
      min = kernel.kernelTimeMs;
    }
  });

  return [sum / trials, min * reps];
}

const frequencyMultipler = 83.3;
function getMeanAndMinByStart(
    kernels: any, trials: number, reps: number, start: number) {
  let sum = 0;
  let min = Number.MAX_VALUE;
  for (let i = 0; i < trials * reps; i++) {
    const kernel = kernels[start + i]
    const adjustTime = kernel.kernelTimeMs*frequencyMultipler;
    sum += adjustTime;
    if (adjustTime < min) {
      min = adjustTime;
    }
  }

  return [sum / trials, min * reps];
}

function time(
    doRep: () => tf.Tensor[] | tf.Tensor, disposeAfterEachTrial = false,
    trials = 20, reps = 20, description: string = 'default') {
  let toDispose: tf.Tensor[] = [];
  const dispose = () => {
    for (const t of toDispose) {
      t.dispose();
    }
    toDispose = [];
  };

  for (let t = 0; t < trials; ++t) {
    let result;
    for (let r = 0; r < reps; ++r) {
      result = doRep();

      toDispose = toDispose.concat(Array.isArray(result) ? result : [result]);
    }
    if (disposeAfterEachTrial) {
      dispose();
    }
  }
}

let nRep = 50;
let nTrail = 50;
let isWarmup = true;
let testCounter = 0;
let testNames: string[] = [];
function testConv2d(
    xShape: [number, number, number, number],
    fShape: [number, number, number, number], stride: [number, number],
    pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') {
  jasmine.DEFAULT_TIMEOUT_INTERVAL = 100000;
  if (isWarmup === false) {
    testCounter++;
  }
  const x = tf.randomNormal<tf.Rank.R4>(xShape);
  const f = tf.randomNormal<tf.Rank.R4>(fShape);
  testNames.push(`conv2d xShape:${xShape} fShape:${fShape} stride:${
      stride} pad:${pad} format:${format}`);
  time(
      () => {
        const res = tf.conv2d(x, f, stride, pad, format);
        res.dispose();
        return [];
      },
      false, nTrail, nRep,
      `conv2d xShape:${xShape} fShape:${fShape} stride:${stride} pad:${
          pad} format:${format}`);

  x.dispose();
  f.dispose();
}

async function test() {
  testConv2d([1, 8, 8, 256], [1, 1, 256, 1024], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 512], [1, 1, 512, 2048], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 1024], [1, 1, 1024, 2048], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 1024], [1, 1, 1024, 512], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 2048], [1, 1, 2048, 17], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 2048], [1, 1, 2048, 32], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 2048], [1, 1, 2048, 34], [1, 1], 'same', 'NHWC');
  testConv2d([1, 8, 8, 2048], [1, 1, 2048, 512], [1, 1], 'same', 'NHWC');

  testConv2d([1, 15, 15, 128], [1, 1, 128, 512], [1, 1], 'same', 'NHWC');
  testConv2d([1, 15, 15, 256], [1, 1, 256, 1024], [1, 1], 'same', 'NHWC');
  testConv2d([1, 15, 15, 512], [1, 1, 512, 256], [1, 1], 'same', 'NHWC');
  testConv2d([1, 15, 15, 512], [1, 1, 512, 1024], [1, 1], 'same', 'NHWC');
  testConv2d([1, 15, 15, 1024], [1, 1, 1024, 256], [1, 1], 'same', 'NHWC');

  testConv2d([1, 29, 29, 64], [1, 1, 64, 256], [1, 1], 'same', 'NHWC');
  testConv2d([1, 29, 29, 128], [1, 1, 128, 512], [1, 1], 'same', 'NHWC');
  testConv2d([1, 29, 29, 256], [1, 1, 256, 128], [1, 1], 'same', 'NHWC');
  testConv2d([1, 29, 29, 256], [1, 1, 256, 512], [1, 1], 'same', 'NHWC');
  testConv2d([1, 29, 29, 512], [1, 1, 512, 128], [1, 1], 'same', 'NHWC');

  testConv2d([1, 57, 57, 64], [1, 1, 64, 64], [1, 1], 'same', 'NHWC');
  testConv2d([1, 57, 57, 64], [1, 1, 64, 256], [1, 1], 'same', 'NHWC');
  testConv2d([1, 57, 57, 256], [1, 1, 256, 64], [1, 1], 'same', 'NHWC');

  testConv2d([1, 8, 8, 512], [3, 3, 512, 512], [1, 1], 'same', 'NHWC');
  testConv2d([1, 15, 15, 256], [3, 3, 256, 256], [1, 1], 'same', 'NHWC');
  testConv2d([1, 17, 17, 256], [3, 3, 256, 256], [2, 2], 'valid', 'NHWC');
  testConv2d([1, 29, 29, 128], [3, 3, 128, 128], [1, 1], 'same', 'NHWC');
  testConv2d([1, 31, 31, 128], [3, 3, 128, 128], [2, 2], 'valid', 'NHWC');
  testConv2d([1, 57, 57, 64], [3, 3, 64, 64], [1, 1], 'same', 'NHWC');
  testConv2d([1, 59, 59, 64], [3, 3, 64, 64], [2, 2], 'valid', 'NHWC');
  // This fail due to max texture size.
  // testConv2d([1, 231, 231, 3], [7, 7, 3, 64], [2, 2], 'valid', 'NHWC');
}

describeWebGPU('Ops resnetconvsyncv1', () => {
  jasmine.DEFAULT_TIMEOUT_INTERVAL = 200000;
  it('conv2d', async () => {
    // Warm up.
    await test();
    nRep = 10;
    nTrail = 10;
    isWarmup = false;
    // Profile starts.
    const times = await tf.profile(async () => {
      test();
    });
    console.log('testNumber = ' + testCounter);
    console.log(JSON.stringify(times.kernels));
    const fmt = (n: number) => n.toFixed(3);
    let mean = 0;
    let min = 0;

    for (let i = 0; i < testCounter; i++) {
      [mean, min] =
          getMeanAndMinByStart(times.kernels, nTrail, nRep, nTrail * nRep * i);
      console.log(
          testNames[i] + '  ' +
          `Mean time: ${fmt(mean)} ms -> ${fmt(mean / nRep)} / rep` +
          '  ' +
          `Min time: ${fmt(min)} ms -> ${fmt(min / nRep)} / rep`);
    }
  });
});