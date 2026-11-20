'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const kernel = require('../tools/kernel-inspector.js');
const engine = require('../src/model/inference.js');
const data = require('../src/model/model-data.js');
test('Every kernel decomposition reproduces the traced convolution at center and padded edges', () => {
  const result = engine.forward(data.samples[0].pixels);
  for (let block = 0; block < 3; block++)
    for (const layer of [1, 2])
      for (let outChannel = 0; outChannel < 8; outChannel++)
        for (const [row, col] of [
          [0, 0],
          [0, 7],
          [7, 0],
          [7, 7],
          [3, 4],
        ]) {
          const decomposition = kernel.inspect(result, { block, layer, outChannel, row, col });
          const expected =
            result.blocks[block][layer === 1 ? 'conv1' : 'branch'][outChannel * 64 + row * 8 + col];
          assert.equal(decomposition.value, expected);
          assert.ok(
            Math.abs(
              decomposition.bias +
                decomposition.channels.reduce((a, b) => a + b, 0) -
                decomposition.totalBeforeActivation
            ) < 1e-12
          );
        }
});
test('The selected channel contribution equals nine explicit patch × kernel products', () => {
  const result = engine.forward(data.samples[4].pixels, {
    shortcutScales: [0.4, 0, 1],
    branchScales: [0.3, 1, 0.7],
  });
  for (let inChannel = 0; inChannel < 8; inChannel++) {
    const view = kernel.inspect(result, {
      block: 2,
      layer: 2,
      outChannel: 5,
      row: 3,
      col: 4,
      inChannel,
    });
    assert.equal(view.kernel.length, 9);
    assert.equal(view.patch.length, 9);
    assert.deepEqual(
      view.products,
      view.patch.map((v, i) => v * view.kernel[i])
    );
    assert.equal(
      view.channelContribution,
      view.products.reduce((a, b) => a + b, 0)
    );
    assert.equal(view.channelContribution, view.channels[inChannel]);
    assert.deepEqual(view.padding, Array(9).fill(false));
  }
});
test('Top-left padding explicitly marks out-of-bounds inputs with zero values', () => {
  const result = engine.forward(data.samples[0].pixels);
  const view = kernel.inspect(result, { block: 0, layer: 1, outChannel: 0, row: 0, col: 0 });
  assert.deepEqual(view.padding, [true, true, true, true, false, false, true, false, false]);
  view.padding.forEach((padded, i) => {
    if (padded) assert.equal(view.patch[i], 0);
  });
});
test('Reject invalid coordinates, layer and tensors', () => {
  const result = engine.forward(data.samples[0].pixels);
  const options = { block: 0, layer: 1, outChannel: 0, row: 0, col: 0 };
  for (const key of ['block', 'outChannel', 'inChannel', 'row', 'col'])
    assert.throws(() => kernel.inspect(result, { ...options, [key]: -1 }), RangeError);
  assert.throws(() => kernel.inspect(result, { ...options, layer: 3 }), RangeError);
  assert.throws(() => kernel.inspect({}, options), TypeError);
});
