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

// Below code is from:
// https://github.com/axinging/tfjs-examples/commit/fa4b53e06d90b06711e39e7eac13794f4bcd8147
let oldLog: any;
export function startLog(kernels: any) {
  tf.ENV.set('DEBUG', true);
  oldLog = console.log;
  console.log = (msg: any) => {
    let parts: any = [];
    if (typeof msg === 'string') {
      parts = msg.split('\t').map(x => x.slice(2));
    }

    if (parts.length > 2) {
      // heuristic for determining whether we've caught a profiler
      // log statement as opposed to a regular console.log
      // TODO(https://github.com/tensorflow/tfjs/issues/563): return timing
      // information as part of tf.profile
      const scopes =
          parts[0].trim().split('||').filter((s: any) => s !== 'unnamed scope');

      kernels.push({
        scopes: scopes,
        time: Number.parseFloat(parts[1]),
        output: parts[2].trim(),
        inInfo: parts[4],
        gpuProgramsInfo: parts[5]
      });
    } else {
      oldLog.call(oldLog, msg);
    }
  }
}

function sleep(timeMs: any) {
  return new Promise(resolve => setTimeout(resolve, timeMs));
}

export async function logKernelTime(
    kernels: any, trials: any, reps: any, desc: string) {
  await sleep(100);
  let i = 0;
  const times: any[] = [];
  kernels.forEach((k: any) => {
    i = i + 1;
    if (i > reps) {
      times.push(k.time);
      const USE_ALL_DATA = false;
      if (USE_ALL_DATA)
        oldLog.call(
            oldLog,
            (i - reps) + '  ' + k.scopes[k.scopes.length - 1] + '  ' +
                k.time.toFixed(3) + '  ' + k.output + '  ' + k.inInfo + '  ' +
                k.gpuProgramsInfo);
    }
  });
  const mean = times.reduce((a, b) => a + b, 0) / (trials * reps);
  const min = Math.min(...times);
  const fmt = (n: number) => n.toFixed(3);
  // console.log(`${fmt(mean / reps)}`);
  // console.log(`Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
  // console.log(`Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  // oldLog.call(oldLog, desc+ " Mean: "+ fmt(mean) +"  Min: "+fmt(min));
  const frequencyMultipler = 83.3;
  oldLog.call(
      oldLog,
      desc + ' Mean: ' + fmt(mean * frequencyMultipler) +
          '  Min: ' + fmt(min * frequencyMultipler));

  console.log = oldLog;
}