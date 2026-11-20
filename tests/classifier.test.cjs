'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const engine = require('../src/model/inference.js');
const data = require('../src/model/model-data.js');
const fixtures = require('./classifier-fixtures.json');
function close(actual, expected, tolerance = 1e-9) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++)
    assert.ok(
      Math.abs(actual[i] - expected[i]) < tolerance,
      `element ${i}: ${actual[i]} != ${expected[i]}`
    );
}
test('JavaScript matches independent PyTorch traces for baseline and interventions', () => {
  for (const fixture of fixtures) {
    const actual = engine.forward(data.samples[fixture.sampleIndex].pixels, fixture);
    for (const key of ['input', 'stem', 'flat', 'logits', 'probabilities'])
      close(actual[key], fixture.expected[key]);
    assert.equal(actual.prediction, fixture.expected.prediction);
    for (let index = 0; index < 3; index++)
      for (const key of ['input', 'conv1', 'branch', 'scaledBranch', 'shortcut', 'sum', 'output'])
        close(actual.blocks[index][key], fixture.expected.blocks[index][key]);
  }
});
test('All held-out predictions and exported probabilities match PyTorch', () => {
  let correct = 0;
  for (const sample of data.samples) {
    const result = engine.forward(sample.pixels);
    assert.equal(result.prediction, sample.expectedPrediction);
    close(result.probabilities, sample.expectedProbabilities, 6e-10);
    assert.ok(Math.abs(result.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    if (result.prediction === sample.label) correct++;
  }
  assert.equal(correct, data.metrics.testCorrect);
  assert.equal(data.samples.length, data.meta.testCount);
});
test('Residual additions preserve shapes and both path contributions exactly', () => {
  const result = engine.forward(data.samples[0].pixels);
  for (const block of result.blocks) {
    assert.equal(block.input.length, 512);
    for (let i = 0; i < 512; i++) {
      assert.equal(block.shortcut[i], block.input[i]);
      assert.equal(block.sum[i], block.input[i] + block.branch[i]);
      assert.equal(block.output[i], Math.max(0, block.sum[i]));
    }
  }
});
test('Zero branch scales produce exact identity through the nonnegative residual blocks', () => {
  const result = engine.forward(data.samples[0].pixels, { branchScales: [0, 0, 0] });
  for (const block of result.blocks) close(block.output, result.stem, 1e-15);
});
test('Reject malformed or nonfinite input and interventions', () => {
  assert.throws(() => engine.forward([0]), TypeError);
  assert.throws(() => engine.forward(Array(64).fill(NaN)), TypeError);
  assert.throws(() => engine.forward(Array(64).fill(0), { branchScales: [1, NaN, 1] }), TypeError);
});
