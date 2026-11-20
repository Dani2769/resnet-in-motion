/* Exercises controller events with a DOM stub and a simulated renderer. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Data = require('../src/model/model-data.js');
const Engine = require('../src/model/inference.js');

function controller({ reducedMotion = false } = {}) {
  const nodes = [],
    ids = new Map(),
    scheduled = new Map(),
    scheduledDelays = [],
    errors = [];
  let document,
    scene,
    nextTimer = 0;
  function element(tagName) {
    const attributes = new Map(),
      classes = new Set(),
      listeners = new Map(),
      parts = new Map();
    let identifier = '';
    const node = {
      tagName: tagName.toUpperCase(),
      dataset: {},
      style: {},
      children: [],
      value: '',
      hidden: false,
      disabled: false,
      open: false,
      textContent: '',
      innerHTML: '',
      width: 144,
      height: 144,
      set id(value) {
        identifier = value;
        ids.set(value, node);
      },
      get id() {
        return identifier;
      },
      setAttribute(name, value) {
        attributes.set(name, String(value));
        if (name === 'id') node.id = String(value);
        if (name === 'class') {
          classes.clear();
          String(value)
            .split(/\s+/)
            .forEach((value) => classes.add(value));
        }
        if (name.startsWith('data-'))
          node.dataset[name.slice(5).replace(/-([a-z])/g, (_, value) => value.toUpperCase())] =
            String(value);
        if (name === 'hidden') node.hidden = true;
        if (name === 'value') node.value = String(value);
        if (name === 'width' || name === 'height') node[name] = Number(value);
      },
      getAttribute: (name) => attributes.get(name) ?? null,
      removeAttribute(name) {
        attributes.delete(name);
      },
      classList: {
        add: (value) => classes.add(value),
        contains: (value) => classes.has(value),
        toggle(value, on) {
          if (on) classes.add(value);
          else classes.delete(value);
        },
      },
      append(child) {
        node.children.push(child);
      },
      querySelector(selector) {
        if (!parts.has(selector)) parts.set(selector, element('div'));
        return parts.get(selector);
      },
      addEventListener(type, callback) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(callback);
      },
      dispatch(type, detail = {}) {
        let stopped = false;
        const event = {
          target: node,
          preventDefault() {},
          stopPropagation() {
            stopped = true;
          },
          ...detail,
        };
        for (const callback of listeners.get(type) || []) callback(event);
        if (!stopped && type === 'keydown') document.dispatch(type, event);
      },
      closest(selector) {
        return selector
          .split(',')
          .map((value) => value.trim().toUpperCase())
          .includes(node.tagName)
          ? node
          : null;
      },
      focus() {
        document.activeElement = node;
      },
      showModal() {
        node.open = true;
      },
      close() {
        node.open = false;
      },
      getContext: () => ({ clearRect() {}, fillRect() {}, strokeRect() {} }),
    };
    nodes.push(node);
    return node;
  }
  function matches(node, selector) {
    if (selector === 'dialog[open]') return node.tagName === 'DIALOG' && node.open;
    if (selector.startsWith('.')) return node.classList.contains(selector.slice(1));
    const attribute = /^\[([\w-]+)(?:="([^"]*)")?\]$/.exec(selector);
    return attribute
      ? node.getAttribute(attribute[1]) !== null &&
          (attribute[2] === undefined || node.getAttribute(attribute[1]) === attribute[2])
      : false;
  }
  const documentListeners = new Map();
  document = {
    hidden: false,
    activeElement: null,
    getElementById(id) {
      assert.ok(ids.has(id), `Main HTML must supply #${id}`);
      return ids.get(id);
    },
    createElement: element,
    querySelector: (selector) => nodes.find((node) => matches(node, selector)) || null,
    querySelectorAll: (selector) => nodes.filter((node) => matches(node, selector)),
    addEventListener(type, callback) {
      if (!documentListeners.has(type)) documentListeners.set(type, []);
      documentListeners.get(type).push(callback);
    },
    dispatch(type, event) {
      for (const callback of documentListeners.get(type) || []) callback(event);
    },
  };
  const html = fs.readFileSync(path.join(__dirname, '../src/index.html'), 'utf8');
  for (const match of html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/gi)) {
    const node = element(match[1]);
    for (const attribute of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g))
      node.setAttribute(attribute[1], attribute[2] ?? '');
  }
  ids.get('speed').value = '2600';
  class Scene {
    constructor(canvas, options) {
      scene = this;
      this.options = options;
      this.state = {};
      this.version = 0;
      this.syncCalls = 0;
      this.replays = 0;
      this.cameraResets = 0;
    }
    key() {
      return [
        this.version,
        this.state.expandedBlock,
        this.state.blockPhase,
        this.state.channel,
        this.state.row,
        this.state.col,
      ].join('/');
    }
    setData(data) {
      this.data = data;
      this.version++;
    }
    setState(next) {
      const before = this.key();
      Object.assign(this.state, next);
      this.syncCalls++;
      assert.ok(
        this.syncCalls < 1000,
        'Renderer callbacks must not cause a controller/renderer feedback loop'
      );
      if (this.key() !== before && this.state.expandedBlock >= 0) {
        if (['add', 'relu'].includes(this.state.blockPhase))
          this.emit(this.state.reducedMotion ? 'complete' : 'moving');
      }
    }
    payload(stage) {
      return {
        phase: this.state.blockPhase,
        stage,
        complete: stage === 'complete',
        block: this.state.selectedBlock,
        channel: this.state.channel,
        row: this.state.row,
        col: this.state.col,
        data: this.data,
      };
    }
    emit(stage, override = {}) {
      this.options.onOperationStage({ ...this.payload(stage), ...override });
    }
    replayOperation() {
      this.replays++;
      this.emit(this.state.reducedMotion ? 'complete' : 'moving');
      return true;
    }
    resetCamera() {
      this.cameraResets++;
    }
    zoomBy() {}
    resize() {}
    destroy() {}
  }
  const window = {
    ResNetLesson: require('../src/app/lesson.js'),
    ResNetData: Data,
    ResNetEngine: Engine,
    ResNetScene: Scene,
    addEventListener() {},
    matchMedia: () => ({ matches: reducedMotion, addEventListener() {} }),
  };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, '../src/app/controller.js'), 'utf8'),
    {
      window,
      document,
      console: { error: (error) => errors.push(error) },
      setTimeout(callback, delay) {
        scheduled.set(++nextTimer, callback);
        scheduledDelays.push(delay);
        return nextTimer;
      },
      clearTimeout: (id) => scheduled.delete(id),
    },
    { filename: 'playground-app.js' }
  );
  assert.equal(errors.length, 0, errors.map((error) => error.stack).join('\n'));
  return {
    scene,
    scheduledDelays,
    elements: Object.fromEntries(ids),
    snapshot: () => JSON.parse(JSON.stringify(window.ResNetPlayground.getSnapshot())),
    text: (id) => ids.get(id).innerHTML.replace(/<[^>]*>/g, ''),
    click: (id) => ids.get(id).dispatch('click'),
    dataButton(name, value) {
      const button = nodes.find((node) => node.dataset[name] === String(value));
      assert.ok(button, `${name}=${value} button exists`);
      return button;
    },
    selectCell({ channel, row, col }) {
      if (channel !== undefined)
        nodes.find((node) => node.dataset.channel === String(channel)).dispatch('click');
      for (const [axis, value] of Object.entries({ row, col }))
        if (value !== undefined) {
          const slider = ids.get('pixel-' + axis);
          slider.value = value;
          slider.dispatch('input');
        }
    },
    tick() {
      const [id, callback] = scheduled.entries().next().value || [];
      if (callback) {
        scheduled.delete(id);
        callback();
      }
      return !!callback;
    },
  };
}

test('opening a block starts at its split and reveals addition only when both values arrive', () => {
  const browser = controller();
  browser.dataButton('expandBlock', 1).dispatch('click');
  assert.equal(browser.snapshot().stepCount, 19);
  assert.equal(browser.snapshot().step, 7);
  assert.equal(browser.snapshot().blockPhase, 'split');
  const block = browser.scene.data.blocks[1],
    index = 27;
  assert.match(browser.text('pixel-equation'), /x =/);
  browser.click('next');
  assert.equal(browser.snapshot().blockPhase, 'conv1');
  assert.ok(browser.text('pixel-equation').includes(block.conv1[index].toFixed(3)));
  browser.click('next');
  assert.equal(browser.snapshot().blockPhase, 'branch');
  assert.ok(browser.text('pixel-equation').includes(block.branch[index].toFixed(3)));
  browser.click('next');
  assert.equal(browser.snapshot().blockPhase, 'add');
  assert.match(browser.text('pixel-equation'), /≈ …/);
  const syncCalls = browser.scene.syncCalls;
  browser.scene.emit('combined');
  assert.ok(browser.text('pixel-equation').includes('≈ ' + block.sum[index].toFixed(3)));
  assert.equal(
    browser.scene.syncCalls,
    syncCalls,
    'An operation notification only updates the readout'
  );
  browser.click('next');
  assert.equal(browser.snapshot().blockPhase, 'relu');
  assert.match(browser.text('pixel-equation'), /ReLU → …/);
  browser.scene.emit('complete');
  assert.ok(browser.text('pixel-equation').includes('ReLU → ' + block.output[index].toFixed(3)));
});

test('late events from an earlier sample or pixel cannot reveal a stale sum', () => {
  const browser = controller();
  browser.dataButton('expandBlock', 0).dispatch('click');
  browser.dataButton('phase', 'add').dispatch('click');
  const stale = browser.scene.payload('complete');
  browser.click('new-sample');
  assert.equal(browser.snapshot().step, 0);
  assert.equal(browser.snapshot().expandedBlock, -1);
  browser.dataButton('expandBlock', 0).dispatch('click');
  browser.dataButton('phase', 'add').dispatch('click');
  browser.scene.options.onOperationStage(stale);
  assert.match(browser.text('pixel-equation'), /≈ …/);
  const previousPixel = browser.scene.payload('complete');
  browser.elements['pixel-col'].value = 4;
  browser.elements['pixel-col'].dispatch('input');
  browser.scene.options.onOperationStage(previousPixel);
  assert.match(browser.text('pixel-equation'), /≈ …/);
  browser.scene.emit('combined');
  assert.doesNotMatch(browser.text('pixel-equation'), /≈ …/);
});

test('the clipping aid selects a real negative sum and leaves the fixed prediction untouched', () => {
  const browser = controller(),
    probabilities = browser.snapshot().probabilities;
  browser.dataButton('expandBlock', 0).dispatch('click');
  browser.dataButton('phase', 'relu').dispatch('click');
  assert.equal(browser.elements['show-clipped-cell'].hidden, false);
  browser.click('show-clipped-cell');
  const { channel, row, col } = browser.snapshot(),
    index = channel * 64 + row * 8 + col;
  assert.ok(browser.scene.data.blocks[0].sum[index] < 0);
  assert.equal(browser.scene.data.blocks[0].output[index], 0);
  browser.scene.emit('complete');
  assert.match(browser.text('pixel-equation'), /ReLU → 0\.000/);
  assert.match(browser.elements['operation-caption'].textContent, /negative.*clips it to zero/);
  assert.deepEqual(browser.snapshot().probabilities, probabilities);
});

test('replay works without auto-advance and reduced motion immediately shows measured results', () => {
  const browser = controller({ reducedMotion: true });
  browser.dataButton('expandBlock', 2).dispatch('click');
  browser.dataButton('phase', 'add').dispatch('click');
  assert.doesNotMatch(browser.text('pixel-equation'), /≈ …/);
  browser.click('replay-operation');
  assert.equal(browser.scene.replays, 1);
  assert.equal(browser.snapshot().playing, false);
  assert.doesNotMatch(browser.text('pixel-equation'), /≈ …/);
  browser.elements['pixel-row'].dispatch('keydown', { key: 'Escape' });
  assert.equal(browser.snapshot().expandedBlock, -1);
  assert.equal(browser.elements['inspection-hud'].hidden, true);
});

test('auto-advance crosses five block stages and returns to the classifier at checkpoint nineteen', () => {
  const browser = controller();
  browser.dataButton('expandBlock', 2).dispatch('click');
  browser.click('play');
  for (let advance = 0; advance < 6; advance++) assert.equal(browser.tick(), true);
  assert.equal(browser.snapshot().step, 18);
  assert.equal(browser.snapshot().expandedBlock, -1);
  assert.equal(browser.snapshot().playing, false);
  assert.equal(browser.elements.next.disabled, true);
  assert.equal(browser.elements['step-count'].textContent, 'Step 19 / 19');
  assert.equal(browser.tick(), false);
});

test('Quick waits for the finite arithmetic clips to reveal their measured result', () => {
  const browser = controller();
  browser.elements.speed.value = '1500';
  browser.dataButton('expandBlock', 0).dispatch('click');
  browser.dataButton('phase', 'add').dispatch('click');
  browser.click('play');
  assert.equal(browser.scheduledDelays.at(-1), 2300);
  browser.dataButton('phase', 'relu').dispatch('click');
  browser.click('play');
  assert.equal(browser.scheduledDelays.at(-1), 1900);
});

test('rounded measured addends use approximation when their displayed decimals do not sum exactly', () => {
  const browser = controller(),
    trace = browser.scene.data;
  const roundedThousandths = (value) => Math.round(Number(value.toFixed(3)) * 1000);
  let found;
  for (let block = 0; block < trace.blocks.length && !found; block++) {
    const values = trace.blocks[block];
    for (let index = 0; index < values.sum.length; index++) {
      if (
        roundedThousandths(values.input[index]) + roundedThousandths(values.branch[index]) !==
        roundedThousandths(values.sum[index])
      ) {
        found = { block, index };
        break;
      }
    }
  }
  assert.ok(found, 'The real model contains a rounding-carry example');
  const { block, index } = found,
    values = trace.blocks[block];
  assert.equal(
    values.sum[index],
    values.input[index] + values.branch[index],
    'The underlying measured addition stays exact'
  );
  browser.dataButton('expandBlock', block).dispatch('click');
  browser.dataButton('phase', 'add').dispatch('click');
  browser.dataButton('channel', Math.floor(index / 64)).dispatch('click');
  browser.elements['pixel-row'].value = Math.floor((index % 64) / 8);
  browser.elements['pixel-row'].dispatch('input');
  browser.elements['pixel-col'].value = index % 8;
  browser.elements['pixel-col'].dispatch('input');
  browser.scene.emit('combined');
  assert.ok(browser.text('pixel-equation').includes('≈ ' + values.sum[index].toFixed(3)));
  assert.doesNotMatch(browser.text('pixel-equation'), / = /);
});

test('resetting an already open network view reaches the renderer camera reset without changing inference', () => {
  const browser = controller(),
    initial = browser.snapshot(),
    probabilities = initial.probabilities;
  browser.click('reset-camera');
  browser.click('reset-camera');
  assert.equal(
    browser.scene.cameraResets,
    2,
    'Each overview reset reaches the camera API, including when expandedBlock is already -1'
  );
  assert.equal(browser.snapshot().step, initial.step);
  assert.equal(browser.snapshot().sampleId, initial.sampleId);
  assert.deepEqual(browser.snapshot().probabilities, probabilities);
  browser.dataButton('expandBlock', 1).dispatch('click');
  browser.click('reset-camera');
  assert.equal(
    browser.snapshot().expandedBlock,
    1,
    'The overview-only control cannot unexpectedly close an expanded lesson'
  );
  assert.equal(browser.scene.cameraResets, 2);
});

test('Conv 1 and Conv 2 remain ordinary residual-block checkpoints when a plane cell is clicked', () => {
  const browser = controller(),
    probabilities = browser.snapshot().probabilities;
  browser.dataButton('expandBlock', 1).dispatch('click');
  browser.selectCell({ channel: 3 });
  const block = browser.scene.data.blocks[1];
  for (const [phase, tensor, row, col] of [
    ['conv1', 'conv1', 2, 4],
    ['branch', 'branch', 5, 1],
  ]) {
    browser.dataButton('phase', phase).dispatch('click');
    const checkpoint = browser.snapshot().step;
    browser.scene.options.onPixelSelect({ tensor, row, col });
    assert.equal(browser.snapshot().blockPhase, phase);
    assert.equal(browser.snapshot().step, checkpoint);
    assert.equal(browser.snapshot().expandedBlock, 1);
    assert.equal(browser.snapshot().mode, 'expanded');
    assert.equal(browser.snapshot().row, row);
    assert.equal(browser.snapshot().col, col);
    assert.equal(browser.snapshot().channel, 3);
    assert.equal(browser.elements['view-context'].textContent, 'Residual block 02');
    assert.equal(browser.elements['inspection-hud'].hidden, false);
    assert.equal(browser.elements['block-stages'].hidden, false);
    assert.ok(
      browser.text('pixel-equation').includes(block[tensor][3 * 64 + row * 8 + col].toFixed(3))
    );
  }
  for (const id of [
    'open-convolution',
    'convolution-controls',
    'kernel-lesson-controls',
    'kernel-details',
  ])
    assert.equal(browser.elements[id], undefined, `The removed drilldown control #${id} is absent`);
  assert.deepEqual(browser.snapshot().probabilities, probabilities);
  assert.equal(browser.snapshot().stepCount, 19);
});

test('canvas arrows inspect the eight-by-eight block pixel and Escape returns straight to the network', () => {
  const browser = controller();
  browser.dataButton('expandBlock', 2).dispatch('click');
  browser.dataButton('phase', 'branch').dispatch('click');
  browser.selectCell({ channel: 6, row: 0, col: 7 });
  const before = browser.snapshot(),
    canvas = browser.elements['network-scene'];
  canvas.dispatch('keydown', { key: 'ArrowUp' });
  canvas.dispatch('keydown', { key: 'ArrowRight' });
  assert.equal(browser.snapshot().row, 0);
  assert.equal(browser.snapshot().col, 7);
  canvas.dispatch('keydown', { key: 'ArrowLeft' });
  canvas.dispatch('keydown', { key: 'ArrowDown' });
  assert.equal(browser.snapshot().row, 1);
  assert.equal(browser.snapshot().col, 6);
  assert.equal(browser.snapshot().channel, before.channel);
  assert.equal(browser.snapshot().step, before.step);
  assert.match(
    canvas.getAttribute('aria-label'),
    /Click a cell or use arrow keys to inspect values/
  );
  canvas.dispatch('keydown', { key: 'Escape' });
  assert.equal(browser.snapshot().expandedBlock, -1);
  assert.equal(browser.elements['view-context'].textContent, 'Network');
  assert.equal(browser.elements['inspection-hud'].hidden, true);
  assert.equal(browser.elements['block-stages'].hidden, true);
  assert.deepEqual(browser.snapshot().probabilities, before.probabilities);
});

test('all three block tabs retain their five stages and the regular channel and pixel selectors', () => {
  const browser = controller(),
    probabilities = browser.snapshot().probabilities,
    phases = ['split', 'conv1', 'branch', 'add', 'relu'];
  for (let block = 0; block < 3; block++) {
    browser.dataButton('expandBlock', block).dispatch('click');
    assert.equal(browser.snapshot().blockPhase, 'split');
    browser.selectCell({ channel: 2, row: 4, col: 5 });
    for (let index = 0; index < phases.length; index++) {
      browser.dataButton('phase', phases[index]).dispatch('click');
      assert.equal(browser.snapshot().step, 2 + block * 5 + index);
      assert.equal(browser.snapshot().expandedBlock, block);
      assert.equal(browser.snapshot().channel, 2);
      assert.equal(browser.elements['row-value'].textContent, 4);
      assert.equal(browser.elements['col-value'].textContent, 5);
    }
  }
  assert.deepEqual(browser.snapshot().probabilities, probabilities);
  browser.click('model-info');
  assert.equal(browser.elements['model-dialog'].open, true);
  browser.click('close-model');
  browser.click('help-button');
  assert.equal(browser.elements['help-dialog'].open, true);
  browser.click('close-help');
  assert.equal(browser.snapshot().expandedBlock, 2);
  assert.equal(browser.snapshot().blockPhase, 'relu');
});

test('the six timeline groups retain all nineteen checkpoints and route without forcing a block open', () => {
  const browser = controller(),
    labels = ['Input', 'Stem', 'Block 01', 'Block 02', 'Block 03', 'Classifier'];
  const starts = [0, 1, 2, 7, 12, 17],
    sizes = [1, 1, 5, 5, 5, 2];
  let checkpoints = 0;
  for (let index = 0; index < labels.length; index++) {
    const group = browser.dataButton('timelineGroup', index),
      label = browser.dataButton('jumpStep', starts[index]);
    const track = group.children.find((child) => child.classList.contains('timeline-group-track'));
    assert.equal(label.textContent, labels[index]);
    assert.ok(label.classList.contains('timeline-group-label'));
    assert.equal(track.children.length, sizes[index]);
    checkpoints += track.children.length;
    for (const button of track.children) {
      assert.ok(button.getAttribute('aria-label').startsWith('Step '));
      assert.equal(button.children[0].tagName, 'I');
    }
  }
  assert.equal(checkpoints, 19);
  browser.dataButton('jumpStep', 7).dispatch('click');
  assert.equal(browser.snapshot().step, 7);
  assert.equal(
    browser.snapshot().expandedBlock,
    -1,
    'A route label does not force expansion in network view'
  );
  assert.equal(browser.dataButton('jumpStep', 7).getAttribute('aria-current'), 'step');
  assert.match(
    browser.dataButton('jumpStep', 7).getAttribute('aria-label'),
    /Block 02, current section/
  );
  assert.ok(browser.dataButton('timelineGroup', 3).classList.contains('current'));
  assert.ok(browser.dataButton('timelineGroup', 2).classList.contains('complete'));
  browser.dataButton('expandBlock', 0).dispatch('click');
  browser.dataButton('jumpStep', 7).dispatch('click');
  assert.equal(
    browser.snapshot().expandedBlock,
    1,
    'A route label keeps the expanded presentation while switching blocks'
  );
  browser.dataButton('jumpStep', 17).dispatch('click');
  assert.equal(browser.snapshot().step, 17);
  assert.equal(browser.snapshot().expandedBlock, -1);
  assert.equal(browser.dataButton('jumpStep', 17).getAttribute('aria-current'), 'step');
  assert.equal(browser.dataButton('jumpStep', 7).getAttribute('aria-current'), null);
});

test('Next and walkthrough labels describe the upcoming action while preserving the nineteen-step route', () => {
  const browser = controller();
  const nextLabels = [
    'stem',
    'Block 01',
    'Conv 1',
    'learned change',
    'add',
    'ReLU',
    'Block 02',
    'Conv 1',
    'learned change',
    'add',
    'ReLU',
    'Block 03',
    'Conv 1',
    'learned change',
    'add',
    'ReLU',
    'classifier',
    'prediction',
  ];
  assert.equal(browser.elements.play.textContent, 'Play walkthrough');
  browser.click('play');
  assert.equal(browser.elements.play.textContent, 'Pause walkthrough');
  assert.equal(browser.elements.play.getAttribute('aria-label'), 'Pause walkthrough');
  assert.match(browser.elements.play.title, /Space/);
  browser.click('play');
  assert.equal(browser.snapshot().playing, false);
  for (let step = 0; step < nextLabels.length; step++) {
    assert.equal(browser.snapshot().step, step);
    assert.equal(browser.elements.next.textContent, 'Next: ' + nextLabels[step]);
    assert.equal(browser.elements.next.getAttribute('aria-label'), 'Next step');
    assert.equal(browser.elements.next.disabled, false);
    browser.click('next');
  }
  assert.equal(browser.elements.next.textContent, 'Complete');
  assert.equal(browser.elements.next.disabled, true);
  assert.equal(browser.elements.play.textContent, 'Replay walkthrough');
  assert.equal(browser.elements['step-count'].textContent, 'Step 19 / 19');
  browser.click('play');
  assert.equal(browser.snapshot().step, 0);
  assert.equal(browser.snapshot().playing, true);
  assert.equal(browser.elements.play.textContent, 'Pause walkthrough');
  browser.click('play');
});

test('phase progress, the selection address, and the sole Network return control stay in sync', () => {
  const browser = controller();
  assert.equal(browser.elements['view-overview'].textContent, 'Network');
  browser.dataButton('expandBlock', 1).dispatch('click');
  browser.dataButton('phase', 'add').dispatch('click');
  assert.equal(browser.elements['step-title'].textContent, 'Add both contributions.');
  for (const phase of ['split', 'conv1', 'branch'])
    assert.ok(browser.dataButton('phase', phase).classList.contains('complete'));
  assert.ok(browser.dataButton('phase', 'add').classList.contains('selected'));
  assert.equal(browser.dataButton('phase', 'add').classList.contains('complete'), false);
  assert.equal(browser.dataButton('phase', 'relu').classList.contains('complete'), false);
  browser.selectCell({ channel: 7, row: 0, col: 6 });
  assert.equal(browser.elements['selection-address'].textContent, 'Channel 7 · row 0 · col 6');
  assert.match(
    browser.elements['selection-address'].getAttribute('aria-label'),
    /indices run from zero to seven/
  );
  assert.equal(browser.elements['value-context'].textContent, 'INSPECT ONE VALUE');
  assert.equal(browser.elements['view-overview'].textContent, '← Network');
  assert.equal(browser.elements['collapse-block'], undefined);
  browser.click('view-overview');
  assert.equal(browser.snapshot().expandedBlock, -1);
  assert.equal(browser.elements['view-overview'].textContent, 'Network');
  assert.equal(browser.elements['inspection-hud'].hidden, true);
  assert.equal(browser.elements['view-context'].textContent, 'Network');
});
