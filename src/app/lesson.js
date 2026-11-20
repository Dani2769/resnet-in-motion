/* Walkthrough text, block phases, and timeline groups. */
(function (root) {
  'use strict';
  const pad = (value) => String(value).padStart(2, '0');
  const phases = Object.freeze(['split', 'conv1', 'branch', 'add', 'relu']);
  const steps = [
    {
      title: 'Start with a handwritten digit.',
      text: 'The 8 × 8 image supplies 64 grayscale values between 0 and 1.',
    },
    {
      title: 'One image becomes eight feature maps.',
      text: 'Eight learned 3 × 3 filters detect patterns. ReLU clips negative responses to zero.',
    },
    ...[0, 1, 2].flatMap((block) => [
      {
        title: `Block ${pad(block + 1)}: the input takes both paths.`,
        text: 'Both paths receive x. The shortcut carries it unchanged.',
        block,
        phase: 'split',
      },
      {
        title: `Block ${pad(block + 1)}: convolution, then ReLU.`,
        text: 'Conv 1 combines nearby values across all eight channels. ReLU clips negatives.',
        block,
        phase: 'conv1',
      },
      {
        title: `Block ${pad(block + 1)}: produce the learned change.`,
        text: 'Conv 2 produces the signed change F(x), with the same shape as x.',
        block,
        phase: 'branch',
      },
      {
        title: `Block ${pad(block + 1)}: add both contributions.`,
        text: 'Add corresponding values to get z = x + F(x).',
        block,
        phase: 'add',
      },
      {
        title: `Block ${pad(block + 1)}: apply ReLU to the sum.`,
        text: 'ReLU gives y = max(0, z). The output continues forward.',
        block,
        phase: 'relu',
      },
    ]),
    {
      title: 'Turn the features into ten scores.',
      text: 'Flatten the final maps into 512 values. A linear layer scores each digit.',
    },
    {
      title: 'Read the model’s prediction.',
      text: 'Softmax converts scores into probabilities that sum to one. The largest determines the prediction.',
    },
  ];
  const timelineGroups = [
    { label: 'Input', start: 0, end: 0 },
    { label: 'Stem', start: 1, end: 1 },
    { label: 'Block 01', start: 2, end: 6 },
    { label: 'Block 02', start: 7, end: 11 },
    { label: 'Block 03', start: 12, end: 16 },
    { label: 'Classifier', start: 17, end: 18 },
  ];

  function blockStep(block, selectedPhase = 'split') {
    return 2 + block * phases.length + phases.indexOf(selectedPhase);
  }

  function nextDestination(index) {
    const next = steps[index + 1];
    if (!next) return null;
    if (index === 0) return 'stem';
    if (index === steps.length - 3) return 'classifier';
    if (index === steps.length - 2) return 'prediction';
    if (next.phase === 'split') return `Block ${pad(next.block + 1)}`;
    return { conv1: 'Conv 1', branch: 'learned change', add: 'add', relu: 'ReLU' }[next.phase];
  }

  steps.forEach(Object.freeze);
  timelineGroups.forEach(Object.freeze);
  const api = Object.freeze({
    phases,
    steps: Object.freeze(steps),
    timelineGroups: Object.freeze(timelineGroups),
    blockStep,
    nextDestination,
  });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ResNetLesson = api;
})(typeof window !== 'undefined' ? window : globalThis);
