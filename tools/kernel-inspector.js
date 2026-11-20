/* Exact 3×3 convolution decomposition using the classifier's trained weights. */
(function (root) {
  'use strict';
  const data =
    typeof module !== 'undefined' && module.exports
      ? require('../src/model/model-data.js')
      : root.ResNetData;
  function integerInRange(value, name, upper) {
    if (!Number.isInteger(value) || value < 0 || value >= upper)
      throw new RangeError(`${name} must be an integer from 0 to ${upper - 1}.`);
  }
  function inspect(result, { block, layer, outChannel, row, col, inChannel = 0 }) {
    integerInRange(block, 'block', 3);
    if (layer !== 1 && layer !== 2) throw new RangeError('layer must be 1 or 2.');
    integerInRange(outChannel, 'outChannel', 8);
    integerInRange(inChannel, 'inChannel', 8);
    integerInRange(row, 'row', 8);
    integerInRange(col, 'col', 8);
    if (!result || !result.blocks || !result.blocks[block])
      throw new TypeError('A classifier forward result is required.');
    const source = layer === 1 ? result.blocks[block].input : result.blocks[block].conv1;
    if (!source || source.length !== 512)
      throw new TypeError('The source tensor must have shape 8 × 8 × 8.');
    const weights = data.weights[`blocks.${block}.conv${layer}.weight`];
    const bias = data.weights[`blocks.${block}.conv${layer}.bias`][outChannel];
    const kernel = [],
      patch = [],
      products = [],
      padding = [],
      channels = [];
    let totalBeforeActivation = bias;
    for (let channel = 0; channel < 8; channel++) {
      let contribution = 0;
      const kernelOffset = (outChannel * 8 + channel) * 9;
      for (let ky = 0; ky < 3; ky++)
        for (let kx = 0; kx < 3; kx++) {
          const iy = row + ky - 1,
            ix = col + kx - 1;
          const isPadding = iy < 0 || iy >= 8 || ix < 0 || ix >= 8;
          const weight = weights[kernelOffset + ky * 3 + kx];
          const activation = isPadding ? 0 : source[channel * 64 + iy * 8 + ix];
          const product = activation * weight;
          contribution += product;
          // Match the forward engine's accumulation order, including bias first.
          if (!isPadding) totalBeforeActivation += product;
          if (channel === inChannel) {
            kernel.push(weight);
            patch.push(activation);
            products.push(product);
            padding.push(isPadding);
          }
        }
      channels.push(contribution);
    }
    return {
      kernel,
      patch,
      products,
      channelContribution: channels[inChannel],
      bias,
      totalBeforeActivation,
      value: layer === 1 ? Math.max(0, totalBeforeActivation) : totalBeforeActivation,
      channels,
      padding,
    };
  }
  const api = { inspect };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ResNetKernel = api;
})(typeof window !== 'undefined' ? window : globalThis);
