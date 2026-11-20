/* Forward pass for the trained digit classifier. Tensors use channel-first order. */
(function (root) {
  'use strict';
  const data =
    typeof module !== 'undefined' && module.exports ? require('./model-data.js') : root.ResNetData;
  const S = 8,
    AREA = 64,
    CHANNELS = 8;
  function relu(input) {
    return Float64Array.from(input, (x) => Math.max(0, x));
  }
  function convolution(input, inputChannels, outputChannels, weights, bias) {
    const output = new Float64Array(outputChannels * AREA);
    for (let oc = 0; oc < outputChannels; oc++) {
      for (let row = 0; row < S; row++)
        for (let col = 0; col < S; col++) {
          let value = bias[oc];
          for (let ic = 0; ic < inputChannels; ic++) {
            const kernelOffset = (oc * inputChannels + ic) * 9;
            for (let ky = 0; ky < 3; ky++) {
              const iy = row + ky - 1;
              if (iy < 0 || iy >= S) continue;
              for (let kx = 0; kx < 3; kx++) {
                const ix = col + kx - 1;
                if (ix < 0 || ix >= S) continue;
                value += input[ic * AREA + iy * S + ix] * weights[kernelOffset + ky * 3 + kx];
              }
            }
          }
          output[oc * AREA + row * S + col] = value;
        }
    }
    return output;
  }
  function scaleTensor(input, scale) {
    return Float64Array.from(input, (v) => v * scale);
  }
  function add(a, b) {
    return Float64Array.from(a, (v, i) => v + b[i]);
  }
  function softmax(logits) {
    const max = Math.max(...logits);
    const exp = Float64Array.from(logits, (v) => Math.exp(v - max));
    const total = exp.reduce((sum, v) => sum + v, 0);
    return Float64Array.from(exp, (v) => v / total);
  }
  function forward(pixels, options = {}) {
    if (!pixels || pixels.length !== AREA || !Array.from(pixels).every(Number.isFinite))
      throw new TypeError('Input must contain exactly 64 finite grayscale values.');
    const weights = options.weights || data.weights;
    const shortcutScales = options.shortcutScales || [1, 1, 1];
    const branchScales = options.branchScales || [1, 1, 1];
    if (
      shortcutScales.length !== 3 ||
      branchScales.length !== 3 ||
      ![...shortcutScales, ...branchScales].every(Number.isFinite)
    )
      throw new TypeError('Each scale array must contain three finite numbers.');
    const input = Float64Array.from(pixels);
    const stem = relu(
      convolution(input, 1, CHANNELS, weights['stem.weight'], weights['stem.bias'])
    );
    let current = stem;
    const blocks = [];
    for (let index = 0; index < 3; index++) {
      const prefix = `blocks.${index}.`;
      const blockInput = current;
      const conv1 = relu(
        convolution(
          blockInput,
          CHANNELS,
          CHANNELS,
          weights[prefix + 'conv1.weight'],
          weights[prefix + 'conv1.bias']
        )
      );
      const branch = convolution(
        conv1,
        CHANNELS,
        CHANNELS,
        weights[prefix + 'conv2.weight'],
        weights[prefix + 'conv2.bias']
      );
      const scaledBranch = scaleTensor(branch, branchScales[index]);
      const shortcut = scaleTensor(blockInput, shortcutScales[index]);
      const sum = add(shortcut, scaledBranch);
      const output = relu(sum);
      blocks.push({ input: blockInput, conv1, branch, scaledBranch, shortcut, sum, output });
      current = output;
    }
    const flat = Float64Array.from(current);
    const logits = new Float64Array(10);
    for (let digit = 0; digit < 10; digit++) {
      let value = weights['head.bias'][digit];
      for (let i = 0; i < flat.length; i++)
        value += flat[i] * weights['head.weight'][digit * flat.length + i];
      logits[digit] = value;
    }
    const probabilities = softmax(logits);
    let prediction = 0;
    for (let i = 1; i < 10; i++) if (logits[i] > logits[prediction]) prediction = i;
    return { input, stem, blocks, flat, logits, probabilities, prediction };
  }
  const api = { forward, convolution, relu, add, softmax, shape: [8, 8, 8] };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ResNetEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
