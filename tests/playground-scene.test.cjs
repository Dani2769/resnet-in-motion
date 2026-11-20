const test = require('node:test');
const assert = require('node:assert/strict');
const { ResNetScene, PALETTE } = require('../src/rendering/scene.js');
const Engine = require('../src/model/inference.js');
const Data = require('../src/model/model-data.js');

// The canvas stub records draw calls and pointer listeners for geometry and interaction checks.
function surface(width = 900, height = 385) {
  const callbacks = new Map(),
    labels = [],
    gradients = { addColorStop() {} };
  const geometry = [];
  const context = new Proxy(
    {
      createRadialGradient() {
        return gradients;
      },
      createLinearGradient() {
        return gradients;
      },
      measureText(text) {
        return { width: text.length * 5.5 };
      },
      fillText(text, x, y) {
        assert.ok(Number.isFinite(x) && Number.isFinite(y));
        labels.push({
          text,
          x,
          y,
          font: this.font,
          alpha: this.globalAlpha,
          color: this.fillStyle,
        });
      },
    },
    {
      get(object, key) {
        if (key in object) return object[key];
        return (...args) => {
          for (const value of args)
            if (typeof value === 'number')
              assert.ok(Number.isFinite(value), `${key}: finite drawing coordinate`);
          if (key === 'moveTo' || key === 'lineTo' || key === 'arc') geometry.push(args);
        };
      },
      set(object, key, value) {
        object[key] = value;
        return true;
      },
    }
  );
  const canvas = {
    style: {},
    clientWidth: width,
    clientHeight: height,
    getContext: () => context,
    getBoundingClientRect: () => ({ width, height, left: 20, top: 30 }),
    addEventListener(name, callback) {
      callbacks.set(name, callback);
    },
    removeEventListener(name, callback) {
      if (callbacks.get(name) === callback) callbacks.delete(name);
    },
    setPointerCapture() {},
  };
  return {
    canvas,
    callbacks,
    labels,
    geometry,
    changeSize(w, h) {
      width = w;
      height = h;
    },
  };
}

function animationClock() {
  const oldRAF = global.requestAnimationFrame,
    oldCancel = global.cancelAnimationFrame;
  const oldPerformance = Object.getOwnPropertyDescriptor(global, 'performance');
  let counter = 0,
    stamp = 0;
  const scheduled = new Map();
  Object.defineProperty(global, 'performance', {
    configurable: true,
    writable: true,
    value: { now: () => stamp },
  });
  global.requestAnimationFrame = (fn) => {
    scheduled.set(++counter, fn);
    return counter;
  };
  global.cancelAnimationFrame = (id) => scheduled.delete(id);
  return {
    scheduled,
    tick(interval = 40) {
      stamp += interval;
      const batch = [...scheduled];
      scheduled.clear();
      for (const [, fn] of batch) fn(stamp);
    },
    restore() {
      global.requestAnimationFrame = oldRAF;
      global.cancelAnimationFrame = oldCancel;
      if (oldPerformance) Object.defineProperty(global, 'performance', oldPerformance);
      else delete global.performance;
    },
  };
}

function trace(options = {}) {
  const result = Engine.forward(Data.samples.find((s) => s.label === 6).pixels, options);
  result.shortcutScales = options.shortcutScales || [1, 1, 1];
  result.branchScales = options.branchScales || [1, 1, 1];
  return result;
}

test('real activation planes and every guided step have finite geometry across viewport sizes', () => {
  const clock = animationClock();
  try {
    const result = trace();
    for (const [width, height] of [
      [360, 320],
      [820, 385],
      [1300, 445],
    ]) {
      const mock = surface(width, height),
        scene = new ResNetScene(mock.canvas);
      scene.setData(result);
      for (let step = 0; step < 19; step++) {
        scene.setState({
          step,
          selectedBlock: Math.min(2, Math.max(0, Math.floor((step - 2) / 5))),
          reducedMotion: true,
        });
        scene.render();
      }
      assert.ok(mock.geometry.length > 2000);
      for (const preset of ['overview', 'front', 'top']) {
        scene.setCameraPreset(preset);
        clock.tick();
        scene.render();
        const input = scene.project(-8.35, 0, 0),
          head = scene.project(7.5, 0, 0);
        assert.ok(input.x > 0 && input.x < width, `${preset}: input stays in view at ${width}`);
        assert.ok(head.x > 0 && head.x < width, `${preset}: classifier stays in view at ${width}`);
      }
      scene.destroy();
    }
  } finally {
    clock.restore();
  }
});

test('block picking uses the projected geometry and a touch tap never traps page scrolling', () => {
  const clock = animationClock();
  try {
    const mock = surface(),
      selected = [],
      scene = new ResNetScene(mock.canvas, { onSelect: (i) => selected.push(i) });
    scene.setState({ reducedMotion: true });
    scene.setData(trace());
    assert.equal(mock.canvas.style.touchAction, 'pan-y');
    for (let block = 0; block < 3; block++) {
      const hit = scene.hitTargets.filter((h) => h.block === block).at(-1);
      const e = {
        clientX: 20 + (hit.left + hit.right) / 2,
        clientY: 30 + hit.top + 8,
        pointerType: 'touch',
        pointerId: 1,
      };
      mock.callbacks.get('pointerdown')(e);
      mock.callbacks.get('pointerup')(e);
      assert.equal(selected.at(-1), block);
    }
    const before = { ...scene.goal },
      e = { clientX: 160, clientY: 150, pointerType: 'touch', pointerId: 2 };
    mock.callbacks.get('pointerdown')(e);
    mock.callbacks.get('pointermove')({ ...e, clientY: 230 });
    mock.callbacks.get('pointerup')({ ...e, clientY: 230 });
    assert.deepEqual(scene.goal, before, 'single-finger scrolling does not orbit the camera');
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('resizing preserves CSS geometry, camera presets reset, and zoom is bounded', () => {
  const clock = animationClock();
  try {
    const mock = surface(),
      scene = new ResNetScene(mock.canvas);
    scene.setState({ reducedMotion: true, channel: 99 });
    assert.equal(scene.state.channel, 7);
    scene.setData(trace());
    scene.focusBlock(2);
    clock.tick();
    assert.equal(scene.state.mode, 'focus');
    assert.equal(scene.state.selectedBlock, 2);
    assert.equal(scene.camera.target, scene.goal.target);
    scene.zoomBy(1e5);
    assert.equal(scene.goal.zoom, 3.4);
    scene.zoomBy(1e-5);
    assert.equal(scene.goal.zoom, 0.6);
    mock.changeSize(420, 320);
    scene.resize();
    assert.equal(scene.width, 420);
    assert.equal(scene.height, 320);
    assert.ok(scene.canvas.width >= 420 && scene.canvas.height >= 320);
    scene.resetCamera();
    clock.tick();
    assert.equal(scene.goal.zoom, 1);
    assert.equal(scene.state.mode, 'overview');
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('zeroing either contribution mutes its path and the displayed branch uses the scaled tensor', () => {
  const clock = animationClock();
  try {
    const mock = surface(),
      scene = new ResNetScene(mock.canvas),
      paths = [],
      planes = [];
    const segment = scene.segment.bind(scene),
      plane = scene.plane.bind(scene);
    scene.segment = (...args) => {
      paths.push(args);
      segment(...args);
    };
    scene.plane = (...args) => {
      planes.push(args);
      plane(...args);
    };
    scene.setState({ step: 2, selectedBlock: 0, reducedMotion: true });
    scene.setData(trace());
    let arch = paths.filter((p) => p[0].length > 4)[0];
    assert.equal(arch[1], PALETTE.identity);
    assert.equal(arch[2], true);
    assert.ok(
      paths.some((p) => p[0].length === 4 && p[1] === PALETTE.branch && p[2] === true),
      'both paths contribute'
    );
    paths.length = 0;
    planes.length = 0;
    const noShortcut = trace({ shortcutScales: [0, 1, 1] });
    scene.setData(noShortcut);
    arch = paths.filter((p) => p[0].length > 4)[0];
    assert.equal(arch[2], false);
    assert.ok(arch[3] <= 0.1);
    paths.length = 0;
    planes.length = 0;
    const noBranch = trace({ branchScales: [0, 1, 1] });
    scene.setData(noBranch);
    assert.ok(noBranch.blocks[0].scaledBranch.every((v) => v === 0));
    assert.equal(
      planes[3][1],
      noBranch.blocks[0].scaledBranch,
      'branch tile values match the intervened contribution'
    );
    const learnedPath = paths.find((p) => p[0].length === 4 && p[1] === PALETTE.branch);
    assert.equal(learnedPath[2], false);
    assert.ok(learnedPath[3] <= 0.1);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('reduced motion stops scheduling frames; destroy cancels animation and input listeners', () => {
  const clock = animationClock();
  try {
    const mock = surface(),
      scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    scene.setState({ step: 5, reducedMotion: false });
    clock.tick();
    assert.equal(clock.scheduled.size, 1, 'animated computation requests another frame');
    scene.setState({ playing: false });
    clock.tick();
    clock.tick();
    assert.equal(clock.scheduled.size, 0, 'Pause stops the travelling-light animation loop');
    scene.setState({ playing: true });
    clock.tick();
    assert.equal(clock.scheduled.size, 1);
    scene.setState({ reducedMotion: true });
    clock.tick();
    clock.tick();
    assert.equal(clock.scheduled.size, 0, 'static reduced-motion view has no animation loop');
    scene.focusBlock(1);
    clock.tick();
    clock.tick();
    assert.equal(clock.scheduled.size, 0, 'reduced-motion camera settles without a repeating loop');
    scene.setState({ reducedMotion: false });
    clock.tick();
    assert.equal(clock.scheduled.size, 1);
    scene.destroy();
    assert.equal(clock.scheduled.size, 0);
    assert.equal(mock.callbacks.size, 0);
    scene.setState({ step: 2 });
    assert.equal(clock.scheduled.size, 0, 'destroyed scene cannot revive its loop');
  } finally {
    clock.restore();
  }
});

test('expansion opens real block tensors into separate 3D planes with a common measured scale', () => {
  const clock = animationClock();
  try {
    const result = trace();
    for (const [width, height] of [
      [360, 1000],
      [360, 1100],
      [900, 710],
      [1300, 710],
    ]) {
      const mock = surface(width, height),
        scene = new ResNetScene(mock.canvas);
      scene.setData(result);
      scene.setState({ reducedMotion: true, playing: false, channel: 2, row: 4, col: 3 });
      for (let block = 0; block < 3; block++) {
        scene.expandBlock(block);
        assert.equal(scene.isExpanded, true);
        assert.equal(scene.state.expandedBlock, block);
        assert.equal(scene.expansion, 1);
        assert.deepEqual(
          scene.expandedPlanes.map((p) => p.name),
          ['input', 'conv1', 'branch', 'sum', 'output']
        );
        assert.equal(scene.expandedPlanes[0].tensor, result.blocks[block].shortcut);
        assert.equal(scene.expandedPlanes[2].tensor, result.blocks[block].scaledBranch);
        assert.equal(
          scene.expandedPlanes[3].tensor,
          result.blocks[block].sum,
          'sum is shown before ReLU'
        );
        assert.equal(scene.expandedPlanes[4].tensor, result.blocks[block].output);
        assert.equal(
          new Set(scene.expandedPlanes.filter((p) => p.name !== 'conv1').map((p) => p.limit)).size,
          1,
          'addition maps share one brightness scale'
        );
        assert.equal(scene.pixelTargets.length, 320, 'all five actual 8×8 maps can be picked');
        for (const target of scene.pixelTargets)
          for (const point of target.points) {
            assert.ok(
              point.x > 0 && point.x < width,
              `expanded map stays horizontally in ${width}px viewport`
            );
            const layout = scene.expandedLayout();
            assert.ok(
              point.y > layout.top && point.y < layout.bottom,
              `expanded map stays between viewport controls at ${width}px`
            );
          }
      }
      scene.collapseBlock();
      assert.equal(scene.isExpanded, false);
      assert.equal(scene.expansion, 0);
      assert.equal(scene.state.mode, 'overview');
      scene.destroy();
    }
  } finally {
    clock.restore();
  }
});

test('clicking a projected expanded feature cell selects its actual row and column', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 500),
      selected = [],
      scene = new ResNetScene(mock.canvas, { onPixelSelect: (value) => selected.push(value) });
    scene.setData(trace());
    scene.setState({ reducedMotion: true, playing: false });
    scene.expandBlock(1);
    const cell = scene.pixelTargets.find((t) => t.tensor === 'sum' && t.row === 6 && t.col === 2);
    const x = cell.points.reduce((sum, p) => sum + p.x, 0) / 4,
      y = cell.points.reduce((sum, p) => sum + p.y, 0) / 4;
    const event = { clientX: x + 20, clientY: y + 30, pointerType: 'touch', pointerId: 1 };
    mock.callbacks.get('pointerdown')(event);
    mock.callbacks.get('pointerup')(event);
    assert.deepEqual(selected.at(-1), { row: 6, col: 2, tensor: 'sum' });
    assert.equal(scene.state.row, 6);
    assert.equal(scene.state.col, 2);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('spreading and collapsing geometry animate finitely while the forward tour is paused', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 500),
      scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    scene.setState({ step: 5, playing: false, reducedMotion: false });
    scene.expandBlock(0);
    assert.equal(scene.expansion, 0);
    clock.tick();
    assert.ok(scene.expansion > 0 && scene.expansion < 1, 'geometry actually interpolates');
    for (let frame = 0; frame < 150 && clock.scheduled.size; frame++) clock.tick();
    assert.equal(scene.expansion, 1);
    assert.equal(clock.scheduled.size, 0, 'paused scene stops after finite expansion');
    scene.collapseBlock();
    clock.tick();
    assert.ok(scene.expansion > 0 && scene.expansion < 1);
    for (let frame = 0; frame < 150 && clock.scheduled.size; frame++) clock.tick();
    assert.equal(scene.expansion, 0);
    assert.equal(clock.scheduled.size, 0);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('selected pixel outlines remain above every tensor face when the expanded block is orbited', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 500),
      ctx = mock.canvas.getContext('2d'),
      operations = [];
    ctx.fill = () => operations.push({ kind: 'fill', style: ctx.fillStyle });
    ctx.stroke = () =>
      operations.push({ kind: 'stroke', style: ctx.strokeStyle, width: ctx.lineWidth });
    const scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    scene.setState({ reducedMotion: true, playing: false, row: 5, col: 1 });
    scene.expandBlock(1);
    for (const yaw of [-1.23, -0.14, 0.92, 2.2]) {
      operations.length = 0;
      scene.camera.yaw = yaw;
      scene.render();
      const outlines = operations.flatMap((op, i) =>
        op.kind === 'stroke' && op.style === '#f4f7ff' ? [i] : []
      );
      assert.equal(outlines.length, 5, 'the same selected cell is marked on all five feature maps');
      const lastFace = operations.findLastIndex((op) => op.kind === 'fill');
      assert.ok(
        outlines.every((i) => i > lastFace),
        'no tensor or node face can paint over the selection annotation'
      );
      for (const index of outlines) {
        assert.equal(
          operations[index - 1].style,
          '#07101b',
          'a dark halo sits immediately under the white selection'
        );
        assert.ok(operations[index - 1].width > operations[index].width);
      }
      assert.equal(
        scene.pixelTargets.length,
        320,
        'the extra stroke does not add a measurement or hit target'
      );
    }
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('explicit block phases reveal only the measured tensors reached by the 19-step lesson', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 550),
      scene = new ResNetScene(mock.canvas),
      result = trace();
    scene.setData(result);
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      step: 7,
      blockPhase: 'split',
      reducedMotion: true,
      playing: false,
    });
    const alpha = (name) => scene.expandedPlanes.find((plane) => plane.name === name).alpha;
    assert.equal(alpha('input'), 1);
    assert.ok(alpha('conv1') < 0.2);
    assert.ok(alpha('branch') < 0.2);
    assert.ok(alpha('sum') < 0.2);
    assert.ok(alpha('output') < 0.2);
    scene.setState({ step: 8, blockPhase: 'conv1' });
    assert.equal(alpha('conv1'), 1);
    assert.ok(alpha('branch') < 0.2);
    mock.labels.length = 0;
    scene.setState({ step: 9, blockPhase: 'branch' });
    assert.equal(alpha('branch'), 1);
    assert.ok(alpha('sum') < 0.2);
    assert.ok(mock.labels.some((label) => label.text === result.blocks[1].branch[27].toFixed(3)));
    assert.ok(
      !mock.labels.some((label) => label.text === result.blocks[1].sum[27].toFixed(3)),
      'future sum is not labelled before addition'
    );
    scene.setState({ step: 10, blockPhase: 'add' });
    assert.equal(alpha('sum'), 1);
    assert.ok(alpha('output') < 0.2);
    scene.setState({ step: 11, blockPhase: 'relu' });
    assert.equal(alpha('output'), 1);
    for (let step = 2; step <= 16; step++) {
      scene.setState({ step, blockPhase: undefined });
      assert.equal(
        scene.currentPhase(),
        ['split', 'conv1', 'branch', 'add', 'relu'][(step - 2) % 5]
      );
    }
    scene.setState({ step: 17, blockPhase: null });
    assert.equal(scene.currentPhase(), null);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('addition animates the actual selected values together, reveals the raw sum and can replay while paused', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 550),
      events = [],
      scene = new ResNetScene(mock.canvas, { onOperationStage: (event) => events.push(event) }),
      result = trace();
    scene.setData(result);
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      step: 9,
      blockPhase: 'branch',
      channel: 0,
      row: 3,
      col: 3,
      reducedMotion: true,
      playing: false,
    });
    assert.equal(events.length, 0, 'no operation callback before an addition or ReLU');
    mock.labels.length = 0;
    scene.setState({ step: 10, blockPhase: 'add', reducedMotion: false });
    scene.render();
    const first = events.at(-1),
      block = result.blocks[1],
      index = 27;
    assert.equal(first.stage, 'moving');
    assert.equal(first.complete, false);
    assert.equal(first.data, result);
    assert.equal(first.block, 1);
    assert.equal(first.x, block.input[index]);
    assert.equal(first.branch, block.branch[index]);
    assert.equal(first.sum, block.sum[index]);
    assert.deepEqual(
      scene.operationTokens.map((token) => [token.kind, token.value]),
      [
        ['input', block.input[index]],
        ['branch', block.branch[index]],
      ]
    );
    assert.ok(
      !mock.labels.some((label) => label.text === block.sum[index].toFixed(3)),
      'sum caption waits for the contributions to arrive'
    );
    for (let frame = 0; frame < 60 && scene.operation.progress < 0.75; frame++) clock.tick();
    assert.ok(events.some((event) => event.stage === 'combined'));
    assert.equal(scene.expandedPlanes.find((plane) => plane.name === 'sum').alpha, 1);
    assert.equal(scene.operationTokens[0].kind, 'sum');
    assert.equal(scene.operationTokens[0].value, block.sum[index]);
    for (let frame = 0; frame < 60 && clock.scheduled.size; frame++) clock.tick();
    assert.equal(events.at(-1).stage, 'complete');
    assert.equal(scene.operation.progress, 1);
    assert.equal(
      clock.scheduled.size,
      0,
      'finite manual explanation does not create an idle animation loop'
    );
    const count = events.length;
    assert.equal(scene.replayOperation(), true);
    assert.equal(scene.operation.progress, 0);
    assert.equal(events.length, count + 1);
    assert.equal(events.at(-1).stage, 'moving');
    scene.setState({ reducedMotion: true });
    clock.tick();
    clock.tick();
    assert.equal(events.at(-1).stage, 'complete');
    assert.equal(clock.scheduled.size, 0);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('ReLU animation clips a measured negative sum to exactly zero and synchronizes a changed selection', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 550),
      events = [],
      scene = new ResNetScene(mock.canvas, { onOperationStage: (event) => events.push(event) }),
      result = trace();
    const index = Array.from(result.blocks[2].sum).findIndex((value) => value < -0.2);
    assert.ok(index >= 0, 'the trained trace contains a negative pre-activation');
    const channel = Math.floor(index / 64),
      row = Math.floor((index % 64) / 8),
      col = index % 8;
    scene.setData(result);
    scene.setState({
      expandedBlock: 2,
      selectedBlock: 2,
      step: 16,
      blockPhase: 'relu',
      channel,
      row,
      col,
      reducedMotion: true,
      playing: false,
    });
    assert.equal(events.at(-1).stage, 'complete');
    assert.equal(events.at(-1).sum, result.blocks[2].sum[index]);
    assert.equal(events.at(-1).output, 0);
    assert.ok(mock.labels.some((label) => label.text === 'negative → 0'));
    scene.setState({ reducedMotion: false });
    scene.replayOperation();
    scene.render();
    assert.equal(scene.operationTokens[0].kind, 'sum');
    assert.equal(scene.operationTokens[0].value, result.blocks[2].sum[index]);
    for (let frame = 0; frame < 80 && clock.scheduled.size; frame++) clock.tick();
    assert.equal(scene.operationTokens[0].kind, 'output');
    assert.equal(scene.operationTokens[0].value, 0);
    assert.equal(scene.expandedPlanes.find((plane) => plane.name === 'output').alpha, 1);
    scene.setState({ row: (row + 1) % 8, reducedMotion: true });
    assert.equal(events.at(-1).row, (row + 1) % 8);
    assert.equal(events.at(-1).stage, 'complete');
    const replacement = Engine.forward(Data.samples.find((sample) => sample.label === 8).pixels);
    scene.setData(replacement);
    assert.equal(
      events.at(-1).data,
      replacement,
      'operation callbacks belong to the current measured trace'
    );
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('hovering outlines the complete block without continuously rendering or trapping touch scrolling', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 550),
      scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    scene.setState({ step: 0, reducedMotion: true, playing: false });
    clock.tick();
    clock.tick();
    const hit = scene.hitTargets.filter((target) => target.block === 1).at(-1);
    const event = {
      clientX: 20 + (hit.left + hit.right) / 2,
      clientY: 30 + hit.top + 8,
      pointerType: 'mouse',
    };
    mock.callbacks.get('pointermove')(event);
    assert.equal(scene.hoveredBlock, 1);
    assert.equal(mock.canvas.style.cursor, 'pointer');
    clock.tick();
    assert.ok(mock.labels.some((label) => label.text === 'Click to unfold'));
    assert.equal(clock.scheduled.size, 0, 'hover feedback settles while the lesson is paused');
    mock.callbacks.get('pointerleave')();
    clock.tick();
    assert.equal(scene.hoveredBlock, -1);
    assert.equal(clock.scheduled.size, 0);
    mock.callbacks.get('pointermove')({ ...event, pointerType: 'touch' });
    assert.equal(scene.hoveredBlock, -1);
    assert.equal(mock.canvas.style.touchAction, 'pan-y');
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('finite operations use elapsed wall time at 5fps rather than the capped decorative pulse clock', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 550),
      events = [],
      scene = new ResNetScene(mock.canvas, { onOperationStage: (event) => events.push(event) });
    scene.setData(trace());
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      blockPhase: 'branch',
      step: 9,
      reducedMotion: true,
      playing: false,
    });
    clock.tick(10000); // An idle interval before clicking Add is not animation time.
    scene.setState({ blockPhase: 'add', step: 10, reducedMotion: false });
    assert.equal(scene.operation.progress, 0);
    for (let frame = 0; frame < 9; frame++) clock.tick(200);
    assert.equal(scene.operation.progress, 0.9);
    assert.equal(events.at(-1).stage, 'combined');
    clock.tick(200);
    assert.equal(scene.operation.progress, 1);
    assert.equal(events.at(-1).stage, 'complete');
    assert.equal(clock.scheduled.size, 0, 'the 2-second addition ends after ten slow frames');
    scene.setState({ blockPhase: 'relu', step: 11 });
    for (let frame = 0; frame < 7; frame++) clock.tick(200);
    assert.equal(scene.operation.progress, 0.875);
    assert.equal(events.at(-1).stage, 'moving');
    clock.tick(200);
    assert.equal(scene.operation.progress, 1);
    assert.equal(events.at(-1).stage, 'complete');
    assert.equal(clock.scheduled.size, 0, 'the 1.6-second ReLU ends after eight slow frames');
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('expanded frames skip all overview tensors and zero-opacity planes create no tile geometry', () => {
  const clock = animationClock();
  try {
    const mock = surface(900, 550),
      scene = new ResNetScene(mock.canvas),
      result = trace();
    scene.setData(result);
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      blockPhase: 'branch',
      step: 9,
      reducedMotion: true,
      playing: false,
    });
    const drawPlane = scene.plane.bind(scene),
      calls = [];
    scene.plane = (...args) => {
      calls.push(args[3]);
      return drawPlane(...args);
    };
    for (const expansion of [1, 0.985]) {
      calls.length = 0;
      scene.expansion = expansion;
      scene.render();
      assert.equal(calls.length, 5, 'only the five expanded stages construct tensor faces');
      assert.ok(
        calls.every((options) => options.selectedOnly),
        'none is an overview channel stack'
      );
    }
    const list = [];
    drawPlane(list, result.stem, 0, { alpha: 0 });
    assert.equal(list.length, 0);
    scene.layerAlpha = 0;
    drawPlane(list, result.stem, 0);
    assert.equal(list.length, 0);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

function tapTarget(mock, target) {
  const x = target.points.reduce((sum, p) => sum + p.x, 0) / target.points.length,
    y = target.points.reduce((sum, p) => sum + p.y, 0) / target.points.length;
  const event = { button: 0, pointerType: 'mouse', pointerId: 1, clientX: 20 + x, clientY: 30 + y };
  mock.callbacks.get('pointerdown')(event);
  mock.callbacks.get('pointerup')(event);
}

test('clicking either convolution map selects its normal block cell without navigating or opening a lesson', () => {
  const clock = animationClock();
  try {
    for (const [width, height] of [
      [357, 1100],
      [1100, 680],
    ]) {
      const mock = surface(width, height),
        events = [],
        scene = new ResNetScene(mock.canvas, { onPixelSelect: (event) => events.push(event) }),
        result = trace();
      scene.setData(result);
      scene.setState({
        expandedBlock: 1,
        selectedBlock: 1,
        step: 9,
        blockPhase: 'branch',
        channel: 2,
        reducedMotion: true,
        playing: false,
      });
      const camera = { ...scene.camera };
      for (const tensor of ['conv1', 'branch']) {
        const target = scene.pixelTargets.find(
          (target) => target.tensor === tensor && target.row === 2 && target.col === 5
        );
        assert.ok(target);
        const x = target.points.reduce((sum, p) => sum + p.x, 0) / 4,
          y = target.points.reduce((sum, p) => sum + p.y, 0) / 4;
        mock.callbacks.get('pointermove')({
          pointerType: 'mouse',
          clientX: x + 20,
          clientY: y + 30,
        });
        assert.equal(mock.canvas.style.cursor, 'crosshair');
        tapTarget(mock, target);
        assert.deepEqual(events.at(-1), { row: 2, col: 5, tensor });
        assert.equal(scene.state.row, 2);
        assert.equal(scene.state.col, 5);
        assert.equal(scene.state.mode, 'expanded');
        assert.equal(scene.state.expandedBlock, 1);
        assert.equal(scene.state.step, 9);
        assert.equal(scene.state.blockPhase, 'branch');
        assert.deepEqual(scene.camera, camera);
        assert.equal(scene.data, result);
        assert.deepEqual(
          scene.expandedPlanes.map((plane) => plane.name),
          ['input', 'conv1', 'branch', 'sum', 'output']
        );
        assert.equal(scene.pixelTargets.length, 320);
        assert.ok(
          !mock.labels.some((label) => label.text.includes('↗')),
          'map captions do not imply another drilldown'
        );
      }
      clock.tick();
      assert.equal(clock.scheduled.size, 0);
      scene.destroy();
    }
  } finally {
    clock.restore();
  }
});

test('residual value clips replace stale samples and clear block picking when their data is removed', () => {
  const clock = animationClock();
  try {
    const mock = surface(1100, 680),
      events = [],
      scene = new ResNetScene(mock.canvas, { onOperationStage: (event) => events.push(event) }),
      result = trace();
    scene.setData(result);
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      step: 9,
      blockPhase: 'branch',
      channel: 6,
      row: 7,
      col: 0,
      reducedMotion: true,
      playing: false,
    });
    scene.setState({ blockPhase: 'add', step: 10, reducedMotion: false });
    clock.tick(300);
    const replacement = Engine.forward(Data.samples.find((sample) => sample.label === 8).pixels);
    scene.setData(replacement);
    assert.equal(events.at(-1).data, replacement);
    assert.equal(scene.operation.progress, 0);
    assert.equal(scene.operation.sum, replacement.blocks[1].sum[6 * 64 + 7 * 8]);
    assert.equal(scene.expandedPlanes.length, 5);
    scene.setData(null);
    assert.equal(scene.operation, null);
    assert.equal(scene.pixelTargets.length, 0);
    assert.equal(scene.hitTargets.length, 0);
    assert.equal(scene.expandedPlanes.length, 0);
    assert.equal(scene.replayOperation(), false);
    const count = events.length;
    clock.tick(1000);
    assert.equal(events.length, count);
    assert.equal(clock.scheduled.size, 0);
    scene.setData(result);
    assert.equal(events.at(-1).data, result);
    assert.equal(scene.pixelTargets.length, 320);
    scene.setState({ reducedMotion: true });
    clock.tick();
    assert.equal(events.at(-1).complete, true);
    assert.equal(clock.scheduled.size, 0);
    scene.collapseBlock();
    assert.equal(scene.operation, null);
    assert.equal(scene.isExpanded, false);
    assert.ok(scene.hitTargets.length > 0);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('polished block layouts keep large measured maps clear of headers and HUDs, with readable future-stage titles', () => {
  const clock = animationClock();
  try {
    for (const [width, height] of [
      [357, 1100],
      [700, 710],
      [1100, 710],
    ]) {
      const mock = surface(width, height),
        scene = new ResNetScene(mock.canvas),
        result = trace();
      scene.setData(result);
      mock.labels.length = 0;
      scene.setState({
        expandedBlock: 1,
        selectedBlock: 1,
        blockPhase: 'split',
        step: 7,
        reducedMotion: true,
        playing: false,
      });
      const layout = scene.expandedLayout();
      for (const cell of scene.pixelTargets)
        for (const point of cell.points)
          assert.ok(
            point.x > 12 && point.x < width - 12 && point.y > layout.top && point.y < layout.bottom
          );
      const inputCells = scene.pixelTargets.filter((target) => target.tensor === 'input'),
        xs = inputCells.flatMap((cell) => cell.points.map((point) => point.x));
      assert.ok(
        Math.max(...xs) - Math.min(...xs) >= (width < 620 ? 80 : width >= 900 ? 100 : 70),
        'feature maps have enough area to read the 8×8 structure'
      );
      if (width < 620) {
        assert.equal(layout.tall, true);
        assert.equal(scene.expandedPlanes[0].position[1], scene.expandedPlanes[1].position[1]);
        assert.ok(scene.expandedPlanes[1].position[1] > scene.expandedPlanes[2].position[1]);
        assert.ok(scene.expandedPlanes[2].position[1] > scene.expandedPlanes[3].position[1]);
      }
      const future = mock.labels.find((label) => label.text === 'Change F(x)');
      assert.ok(future);
      assert.ok(
        future.alpha >= 0.85,
        'a future-stage title stays readable even when its tensor is ghosted'
      );
      assert.equal(
        future.color,
        PALETTE.muted,
        'future titles are subdued without making their text unreadable'
      );
      const hidden = scene.expandedPlanes.find((plane) => plane.name === 'branch');
      assert.ok(hidden.alpha < 0.2);
      assert.equal(scene.pixelTargets.length, 320);
      assert.ok(!mock.labels.some((label) => label.text.includes('↗')));
      if (!layout.small) {
        assert.equal(layout.bottom, height - 250);
        const hudTop = height - 160 - 62;
        for (const stage of scene.expandedPlanes) {
          const labelBottom =
            scene.project(stage.position[0], stage.position[1] - stage.size / 2 - 0.32, 0).y +
            18 +
            5.5;
          assert.ok(
            labelBottom <= hudTop - 20,
            'map value glyphs clear the taller HUD with a useful gap'
          );
        }
      } else assert.equal(layout.bottom, height - 410, 'phone safe area is unchanged');
      scene.setState({ blockPhase: 'add', step: 10, reducedMotion: false });
      scene.render();
      for (const token of scene.operationTokens)
        assert.ok(
          token.y >= layout.top + 15 && token.y <= layout.bottom - 15,
          'finite value tokens clear the same UI reservations'
        );
      clock.tick(2000);
      assert.equal(scene.operation.sum, result.blocks[1].sum[27]);
      scene.destroy();
    }
  } finally {
    clock.restore();
  }
});

test('future map captions say Later while only an active addition or ReLU shows a pending ellipsis', () => {
  const clock = animationClock();
  try {
    const mock = surface(1100, 710),
      ctx = mock.canvas.getContext('2d'),
      strokes = [],
      picked = [],
      scene = new ResNetScene(mock.canvas, { onPixelSelect: (event) => picked.push(event) });
    ctx.stroke = () => strokes.push({ color: ctx.strokeStyle, width: ctx.lineWidth });
    const result = trace();
    scene.setData(result);
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      blockPhase: 'split',
      step: 7,
      reducedMotion: true,
      playing: false,
    });
    const refresh = () => {
      mock.labels.length = 0;
      strokes.length = 0;
      scene.render();
    };
    const caption = (name) => {
      const stage = scene.expandedPlanes.find((stage) => stage.name === name),
        p = scene.project(stage.position[0], stage.position[1] - stage.size / 2 - 0.32, 0);
      return mock.labels.find(
        (label) => Math.abs(label.x - p.x) < 0.01 && Math.abs(label.y - p.y - 18) < 0.01
      );
    };
    const selection = (expected) => {
      assert.equal(
        strokes.filter((stroke) => stroke.color === '#07101b').length,
        expected,
        'only available maps receive the dark selection halo'
      );
      assert.equal(
        strokes.filter((stroke) => stroke.color === '#f4f7ff').length,
        5,
        'one selected-cell measurement remains on each map'
      );
      assert.equal(scene.pixelTargets.length, 320);
      assert.equal(
        new Set(scene.pixelTargets.map((cell) => `${cell.tensor}/${cell.row}/${cell.col}`)).size,
        320,
        'halo strokes do not create duplicate hit targets'
      );
    };
    refresh();
    selection(1);
    for (const name of ['conv1', 'branch', 'sum', 'output']) {
      assert.equal(caption(name).text, 'Later');
      assert.equal(caption(name).color, PALETTE.muted);
    }
    scene.setState({ blockPhase: 'branch', step: 9 });
    refresh();
    selection(3);
    assert.equal(caption('sum').text, 'Later');
    assert.equal(caption('output').text, 'Later');
    scene.setState({ blockPhase: 'add', step: 10, reducedMotion: false });
    refresh();
    selection(3);
    assert.equal(caption('sum').text, '…');
    assert.equal(caption('output').text, 'Later');
    clock.tick(1500);
    refresh();
    selection(4);
    assert.equal(caption('sum').text, result.blocks[1].sum[27].toFixed(3));
    assert.equal(caption('output').text, 'Later');
    scene.setState({ blockPhase: 'relu', step: 11 });
    refresh();
    selection(4);
    assert.equal(caption('output').text, '…');
    assert.ok(!mock.labels.some((label) => label.text === 'Later'));
    clock.tick(1600);
    refresh();
    selection(5);
    assert.equal(caption('output').text, result.blocks[1].output[27].toFixed(3));
    const target = scene.pixelTargets.find(
      (cell) => cell.tensor === 'output' && cell.row === 4 && cell.col === 5
    );
    tapTarget(mock, target);
    assert.deepEqual(picked, [{ row: 4, col: 5, tensor: 'output' }]);
    assert.equal(scene.pixelTargets.length, 320);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('unfolding and folding finish in700ms of real time at5fps without ghost overview headings', () => {
  const clock = animationClock();
  try {
    const mock = surface(1100, 680),
      scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    scene.setState({ reducedMotion: true, playing: false });
    clock.tick(10000);
    const labels = scene.drawLabels.bind(scene),
      plane = scene.plane.bind(scene),
      calls = [];
    let overviewLabels = 0;
    scene.drawLabels = (...args) => {
      overviewLabels++;
      return labels(...args);
    };
    scene.plane = (...args) => {
      calls.push(args[3]);
      return plane(...args);
    };
    scene.setState({
      expandedBlock: 1,
      selectedBlock: 1,
      step: 10,
      blockPhase: 'add',
      reducedMotion: false,
    });
    assert.equal(scene.expansion, 0, 'a preceding idle interval does not count as unfold time');
    overviewLabels = 0;
    clock.tick(200);
    assert.ok(scene.expansion > 0.5 && scene.expansion < 1);
    assert.equal(overviewLabels, 0, 'overview headings disappear as soon as the block takes focus');
    clock.tick(200);
    calls.length = 0;
    clock.tick(200);
    assert.ok(scene.expansion > 0.98);
    assert.equal(calls.length, 5);
    assert.ok(
      calls.every((options) => options.selectedOnly),
      'only the expanded tensors draw near the end of the finite unfold'
    );
    clock.tick(200);
    assert.equal(scene.expansion, 1);
    assert.equal(scene.expansionClip, null);
    assert.equal(
      scene.operation.progress,
      0.4,
      'the separate measured addition retains its own duration'
    );
    scene.collapseBlock();
    overviewLabels = 0;
    clock.tick(200);
    assert.ok(scene.expansion > 0 && scene.expansion < 0.5);
    assert.equal(
      overviewLabels,
      0,
      'folding does not paint network headings over the remaining block'
    );
    clock.tick(200);
    clock.tick(200);
    clock.tick(200);
    assert.equal(scene.expansion, 0);
    assert.equal(scene.expansionClip, null);
    assert.ok(overviewLabels > 0);
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('larger phone overview captions remain separate where input and classifier stages are close together', () => {
  const clock = animationClock();
  try {
    const mock = surface(357, 520),
      scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    mock.labels.length = 0;
    scene.setState({ step: 0, reducedMotion: true, playing: false });
    for (const pair of [['Input', 'Stem']]) {
      const labels = pair.map((value) => mock.labels.find((label) => label.text === value));
      assert.ok(labels.every(Boolean));
      assert.ok(
        Math.abs(labels[0].y - labels[1].y) > 12,
        'neighboring stage captions use distinct baselines instead of colliding'
      );
      for (const label of labels) {
        assert.ok(Number(/ ([\d.]+)px /.exec(label.font)[1]) >= 11);
        assert.ok(label.x > 8 && label.x < 349);
      }
    }
    const classifier = mock.labels.find((label) => label.text === 'Classifier');
    assert.ok(classifier);
    assert.ok(classifier.x > 8 && classifier.x < 349);
    assert.ok(!mock.labels.some((label) => label.text === 'Flat' || label.text === 'Predict'));
    scene.destroy();
  } finally {
    clock.restore();
  }
});

test('the measured1089px overview has one classifier caption with space beside the final block', () => {
  const clock = animationClock();
  try {
    const mock = surface(1089, 560),
      scene = new ResNetScene(mock.canvas);
    scene.setData(trace());
    mock.labels.length = 0;
    scene.setState({ step: 17, reducedMotion: true, playing: false });
    assert.equal(mock.labels.filter((label) => label.text === 'Classifier').length, 1);
    assert.equal(mock.labels.filter((label) => label.text === '512 → 10 scores').length, 1);
    assert.ok(
      !mock.labels.some((label) =>
        ['Flatten', 'Predict', '512 values', '10 classes'].includes(label.text)
      )
    );
    const lastShape = mock.labels
        .filter((label) => label.text === '8 × 8 × 8')
        .sort((a, b) => b.x - a.x)[0],
      head = mock.labels.find((label) => label.text === '512 → 10 scores');
    const shapeRight = lastShape.x + (lastShape.text.length * 5.5) / 2,
      headLeft = head.x - (head.text.length * 5.5) / 2;
    assert.ok(
      headLeft - shapeRight > 12,
      'the classifier shape caption clears the final residual-block caption'
    );
    mock.changeSize(1400, 560);
    scene.resize();
    mock.labels.length = 0;
    scene.render();
    assert.ok(mock.labels.some((label) => label.text === 'Flatten'));
    assert.ok(mock.labels.some((label) => label.text === 'Predict'));
    assert.ok(!mock.labels.some((label) => label.text === 'Classifier'));
    mock.labels.length = 0;
    scene.setState({ step: 9, selectedBlock: 1 });
    assert.ok(mock.labels.some((label) => label.text === 'Current block'));
    assert.ok(!mock.labels.some((label) => label.text === 'Following'));
    scene.destroy();
  } finally {
    clock.restore();
  }
});
