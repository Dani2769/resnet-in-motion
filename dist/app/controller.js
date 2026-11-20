/* Coordinates inference, walkthrough controls, and the canvas renderer. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const Data = window.ResNetData,
    Engine = window.ResNetEngine;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const formatNumber = (value, digits = 3) =>
    (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits);
  const formatProbability = (value) =>
    value >= 0.9995 ? '>99.9%' : (value * 100).toFixed(1) + '%';
  const padNumber = (value) => String(value).padStart(2, '0');
  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const state = {
    step: 0,
    playing: false,
    selectedBlock: 0,
    channel: 0,
    row: 3,
    col: 3,
    sample: Data.samples.find((sample) => sample.label === 6),
    showLabels: true,
    mode: 'overview',
    expandedBlock: -1,
    reducedMotion: motionQuery.matches,
  };
  let result,
    scene,
    timer = null,
    dataVersion = 0,
    syncingScene = false;
  let operation = { key: '', stage: 'moving', complete: false };
  const { phases, steps, timelineGroups, blockStep, nextDestination } = window.ResNetLesson;
  const lastStep = steps.length - 1;
  const phase = () => steps[state.step].phase || null;

  // Input and prediction panels.
  function createButton(label, attributes, handler) {
    const button = document.createElement('button');
    button.textContent = label;
    for (const [key, value] of Object.entries(attributes || {})) button.setAttribute(key, value);
    button.addEventListener('click', handler);
    return button;
  }
  function drawInput() {
    const canvas = $('input-image'),
      ctx = canvas.getContext('2d'),
      size = canvas.width / 8;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.sample.pixels.forEach((value, index) => {
      ctx.fillStyle = `rgb(${[232, 238, 246].map((target, channel) => Math.round([11, 17, 25][channel] + (target - [11, 17, 25][channel]) * value)).join(',')})`;
      ctx.fillRect((index % 8) * size, Math.floor(index / 8) * size, size, size);
      ctx.strokeStyle = '#a6bdcb0b';
      ctx.lineWidth = 0.6;
      ctx.strokeRect((index % 8) * size, Math.floor(index / 8) * size, size, size);
    });
    canvas.setAttribute(
      'aria-label',
      `Handwritten digit ${state.sample.label}, held-out sample ${state.sample.id}`
    );
    $('input-caption').textContent = `A handwritten ${state.sample.label}`;
    $('sample-id').textContent = '#' + state.sample.id;
    document.querySelectorAll('[data-digit]').forEach((button) => {
      const selected = Number(button.dataset.digit) === state.sample.label;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected);
    });
  }
  function renderPrediction() {
    $('prediction-number').textContent = result.prediction;
    $('prediction-number').setAttribute('aria-label', `Predicted digit ${result.prediction}`);
    $('confidence').textContent = formatProbability(result.probabilities[result.prediction]);
    for (let digit = 0; digit < 10; digit++) {
      const row = $('prob-' + digit),
        probability = result.probabilities[digit];
      row.classList.toggle('winner', digit === result.prediction);
      row.querySelector('.prob-fill').style.width = `${probability * 100}%`;
      row.querySelector('.prob-value').textContent =
        probability < 0.0005 ? '<0.1%' : formatProbability(probability);
      row.setAttribute('aria-label', `Digit ${digit}: ${(probability * 100).toFixed(3)} percent`);
      row.title = `Digit ${digit}: ${(probability * 100).toFixed(4)}% · score ${formatNumber(result.logits[digit], 4)}`;
    }
    const match = state.sample.label === result.prediction;
    $('result-status').textContent =
      `${match ? '✓' : '↗'} True label: ${state.sample.label} · held-out image`;
    $('result-status').classList.toggle('mismatch', !match);
  }
  // Measured-value inspection and finite animation state.
  function operationKey() {
    return state.expandedBlock >= 0 && ['add', 'relu'].includes(phase())
      ? [dataVersion, state.selectedBlock, phase(), state.channel, state.row, state.col].join('/')
      : '';
  }
  function prepareOperation() {
    const key = operationKey();
    if (operation.key !== key) operation = { key, stage: 'moving', complete: false };
  }
  function renderNumericHUD() {
    if (!result) return;
    const block = result.blocks[state.selectedBlock],
      index = state.channel * 64 + state.row * 8 + state.col,
      current = phase();
    const number = (value, color) => `<span class="${color}">${formatNumber(value)}</span>`;
    const ready = state.reducedMotion || operation.stage === 'combined' || operation.complete;
    let equation, caption;
    if (current === 'conv1') {
      equation = `<small>Conv 1 + ReLU:</small> ${number(block.conv1[index], 'purple')}`;
      caption = 'Selected Conv 1 response after ReLU.';
    } else if (current === 'branch') {
      equation = `<small>F(x) =</small> ${number(block.branch[index], 'purple')}`;
      caption = 'The learned change can be positive or negative.';
    } else if (current === 'add') {
      equation = `${number(block.input[index], 'mint')} + ${number(block.branch[index], 'purple')} ≈ ${ready ? number(block.sum[index], 'amber') : '<span class="pending-value">…</span>'}`;
      caption = ready ? 'This is the sum before ReLU.' : 'Follow x and F(x) to the addition.';
    } else if (current === 'relu') {
      const complete = state.reducedMotion || operation.complete;
      equation = `${number(block.sum[index], 'amber')} <small>→ ReLU →</small> ${complete ? number(block.output[index], 'amber') : '<span class="pending-value">…</span>'}`;
      caption = complete
        ? block.sum[index] < 0
          ? 'ReLU takes the negative sum and clips it to zero.'
          : block.sum[index] === 0
            ? 'Zero stays zero.'
            : 'ReLU keeps the positive sum.'
        : 'The sum enters ReLU.';
    } else {
      equation = `<small>x =</small> ${number(block.input[index], 'mint')}`;
      caption = 'This input value enters both paths.';
    }
    $('pixel-equation').innerHTML = equation;
    $('operation-caption').textContent = caption;
    $('operation-caption').dataset.operationPhase = current || 'split';
    $('operation-caption').dataset.stage = operation.stage;
  }
  function renderInspector() {
    $('pixel-row').value = state.row;
    $('pixel-col').value = state.col;
    $('row-value').textContent = state.row;
    $('col-value').textContent = state.col;
    $('selection-address').textContent =
      `Channel ${state.channel} · row ${state.row} · col ${state.col}`;
    $('selection-address').setAttribute(
      'aria-label',
      `Feature channel ${state.channel}, row ${state.row}, column ${state.col}. Channel, row and column indices run from zero to seven.`
    );
    document.querySelectorAll('[data-channel]').forEach((button) => {
      const selected = Number(button.dataset.channel) === state.channel;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected);
    });
    renderNumericHUD();
  }
  // Scene controls and walkthrough presentation.
  function renderSceneControls() {
    const expanded = state.expandedBlock >= 0,
      current = phase();
    document.querySelector('.scene-center').classList.toggle('is-expanded', expanded);
    $('expansion-controls').hidden = !expanded;
    $('inspection-hud').hidden = !expanded;
    $('block-stages').hidden = !expanded;
    $('view-context').textContent = expanded
      ? `Residual block ${padNumber(state.selectedBlock + 1)}`
      : 'Network';
    $('expanded-title').textContent = `Inside residual block ${padNumber(state.selectedBlock + 1)}`;
    $('expanded-description').textContent = current
      ? `Stage ${phases.indexOf(current) + 1} of 5`
      : 'Choose a stage.';
    document.querySelector('.scene-hint').textContent = expanded
      ? 'Choose a stage · drag to orbit'
      : 'Open a block to inspect · drag to orbit';
    $('view-overview').classList.toggle('selected', !expanded);
    $('view-overview').setAttribute('aria-pressed', !expanded);
    $('view-overview').textContent = expanded ? '← Network' : 'Network';
    $('view-overview').setAttribute(
      'aria-label',
      expanded ? 'Return to whole network' : 'Whole network'
    );
    document.querySelectorAll('[data-expand-block]').forEach((button) => {
      const selected = Number(button.dataset.expandBlock) === state.expandedBlock;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected);
    });
    document.querySelectorAll('[data-phase]').forEach((button) => {
      const selected = button.dataset.phase === current;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', selected);
      button.classList.toggle(
        'complete',
        phases.indexOf(button.dataset.phase) < phases.indexOf(current)
      );
      if (selected) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    $('replay-operation').hidden = !expanded || !['add', 'relu'].includes(current);
    $('replay-operation').textContent = current === 'relu' ? '↺ Replay ReLU' : '↺ Replay addition';
    $('show-clipped-cell').hidden =
      !expanded ||
      current !== 'relu' ||
      !result.blocks[state.selectedBlock].sum.some((value) => value < 0);
    $('value-context').textContent = 'INSPECT ONE VALUE';
    $('channel-label').textContent = 'CHANNEL';
    $('channels').setAttribute('aria-label', 'Feature channel');
    $('scene-footnote').textContent = expanded
      ? 'Actual values · input, change, sum and output share one colour scale'
      : 'Actual model values · brightness scaled per tensor · pulses illustrate computation';
    $('network-scene').setAttribute(
      'aria-label',
      expanded
        ? `Expanded residual block ${state.selectedBlock + 1}, stage ${current}. Use Next step to follow split, Conv 1, learned change, addition, then ReLU. Click a cell or use arrow keys to inspect values. Escape returns to the network.`
        : 'Interactive 3D residual network. Click a block to unfold it. Keys 1, 2, 3 open blocks; Escape returns to the network.'
    );
  }
  function renderJourney() {
    const current = steps[state.step];
    $('step-count').textContent = `Step ${padNumber(state.step + 1)} / ${steps.length}`;
    const title = current.block === undefined ? current.title : current.title.split(': ')[1];
    $('step-title').textContent = title.charAt(0).toUpperCase() + title.slice(1);
    $('step-description').textContent =
      (state.step === lastStep
        ? `Prediction: ${result.prediction} (${formatProbability(result.probabilities[result.prediction])}). `
        : '') + current.text;
    $('previous').disabled = state.step === 0;
    $('next').disabled = state.step === lastStep;
    const next = steps[state.step + 1];
    const destination = nextDestination(state.step);
    $('next').textContent = next ? `Next: ${destination}` : 'Complete';
    $('next').setAttribute('aria-label', 'Next step');
    const playback = state.playing
      ? 'Pause walkthrough'
      : state.step === lastStep
        ? 'Replay walkthrough'
        : 'Play walkthrough';
    $('play').textContent = playback;
    $('play').setAttribute('aria-label', playback);
    $('play').title = 'Play or pause the walkthrough (Space)';
    $('play').setAttribute('aria-pressed', state.playing);
    $('inspect-block').hidden =
      current.block === undefined || state.expandedBlock === current.block;
    document.querySelectorAll('[data-step]').forEach((button) => {
      const index = Number(button.dataset.step);
      button.classList.toggle('current', index === state.step);
      button.classList.toggle('complete', index < state.step);
      if (index === state.step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    document.querySelectorAll('[data-timeline-group]').forEach((group) => {
      const range = timelineGroups[Number(group.dataset.timelineGroup)];
      group.classList.toggle('current', state.step >= range.start && state.step <= range.end);
      group.classList.toggle('complete', state.step > range.end);
    });
    document.querySelectorAll('[data-jump-step]').forEach((button) => {
      const range = timelineGroups.find((group) => group.start === Number(button.dataset.jumpStep));
      const current = state.step >= range.start && state.step <= range.end;
      if (current) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
      button.setAttribute(
        'aria-label',
        `${range.label}, ${current ? 'current' : state.step > range.end ? 'completed' : 'upcoming'} section. Jump to step ${range.start + 1}.`
      );
    });
  }
  // Controller-to-renderer synchronization.
  function syncScene() {
    if (!scene || syncingScene) return;
    syncingScene = true;
    try {
      scene.setState({
        step: state.step,
        blockPhase: phase(),
        selectedBlock: state.selectedBlock,
        channel: state.channel,
        row: state.row,
        col: state.col,
        expandedBlock: state.expandedBlock,
        mode: state.mode,
        playing: state.playing,
        showLabels: state.showLabels,
        reducedMotion: state.reducedMotion,
      });
    } finally {
      syncingScene = false;
    }
  }
  function renderView() {
    prepareOperation();
    renderSceneControls();
    renderInspector();
    renderJourney();
    syncScene();
  }
  function onOperationStage(event) {
    if (
      !result ||
      state.expandedBlock < 0 ||
      event.phase !== phase() ||
      !operation.key ||
      operation.key !== operationKey()
    )
      return;
    if (event.data && event.data !== result) return;
    if (event.block !== undefined && event.block !== state.selectedBlock) return;
    if (event.channel !== undefined && event.channel !== state.channel) return;
    if (event.row !== undefined && event.row !== state.row) return;
    if (event.col !== undefined && event.col !== state.col) return;
    operation.stage = event.stage;
    operation.complete = !!event.complete;
    // Renderer notifications update only the HUD; they must never restart a clip.
    renderNumericHUD();
  }
  // Inference and navigation actions.
  function compute() {
    result = Engine.forward(state.sample.pixels);
    dataVersion++;
    prepareOperation();
    scene.setData(result);
    drawInput();
    renderPrediction();
    renderView();
  }
  function pause() {
    state.playing = false;
    clearTimeout(timer);
    timer = null;
  }
  function gotoStep(index, { keepPlaying = false } = {}) {
    if (!keepPlaying) pause();
    state.step = clamp(index, 0, lastStep);
    const block = steps[state.step].block;
    if (block !== undefined) {
      state.selectedBlock = block;
      if (state.expandedBlock >= 0) state.expandedBlock = block;
    } else if (state.expandedBlock >= 0) {
      state.expandedBlock = -1;
      state.mode = 'overview';
    }
    if (state.step === lastStep) pause();
    renderView();
  }
  function queueNext() {
    clearTimeout(timer);
    if (!state.playing) return;
    // Leave time to read the result after the addition or ReLU animation ends.
    const minimum =
      state.expandedBlock >= 0 && !state.reducedMotion
        ? phase() === 'add'
          ? 2300
          : phase() === 'relu'
            ? 1900
            : 0
        : 0;
    timer = setTimeout(
      () => {
        gotoStep(state.step + 1, { keepPlaying: true });
        if (state.playing) queueNext();
      },
      Math.max(Number($('speed').value), minimum)
    );
  }
  function restartJourney() {
    state.selectedBlock = 0;
    gotoStep(0);
  }
  function togglePlay() {
    if (state.playing) {
      pause();
      renderView();
      return;
    }
    if (state.step === lastStep) restartJourney();
    state.playing = true;
    renderView();
    queueNext();
  }
  function selectBlock(index) {
    pause();
    state.selectedBlock = clamp(index | 0, 0, 2);
    state.expandedBlock = state.selectedBlock;
    state.mode = 'expanded';
    state.step = blockStep(state.selectedBlock);
    renderView();
  }
  function collapseBlock() {
    state.expandedBlock = -1;
    state.mode = 'overview';
    renderView();
  }
  function selectPhase(selected) {
    if (state.expandedBlock >= 0) gotoStep(blockStep(state.selectedBlock, selected));
  }
  function replayOperation() {
    if (state.expandedBlock < 0 || !['add', 'relu'].includes(phase())) return;
    pause();
    operation.key = '';
    renderView();
    scene.replayOperation();
  }
  function showClippedCell() {
    if (state.expandedBlock < 0 || phase() !== 'relu') return;
    const sum = result.blocks[state.selectedBlock].sum;
    let selected = -1,
      smallest = 0;
    // Search the selected channel first so the example can stay on the current map.
    for (let index = state.channel * 64; index < state.channel * 64 + 64; index++)
      if (sum[index] < smallest) {
        selected = index;
        smallest = sum[index];
      }
    if (selected < 0)
      for (let index = 0; index < sum.length; index++)
        if (sum[index] < smallest) {
          selected = index;
          smallest = sum[index];
        }
    if (selected < 0) return;
    pause();
    const channel = Math.floor(selected / 64),
      row = Math.floor((selected % 64) / 8),
      col = selected % 8;
    const unchanged = channel === state.channel && row === state.row && col === state.col;
    state.channel = channel;
    state.row = row;
    state.col = col;
    if (unchanged) operation.key = '';
    renderView();
    if (unchanged) scene.replayOperation();
  }
  function nextSample(label) {
    const matches = Data.samples.filter((sample) => sample.label === label),
      index = matches.findIndex((sample) => sample.id === state.sample.id);
    pause();
    state.sample = matches[(index + 1) % matches.length];
    state.step = 0;
    state.selectedBlock = 0;
    state.expandedBlock = -1;
    state.mode = 'overview';
    syncScene();
    compute();
  }
  // Initial DOM construction.
  function buildUI() {
    for (let digit = 0; digit < 10; digit++) {
      $('digit-picker').append(
        createButton(
          digit,
          {
            'data-digit': digit,
            'aria-pressed': false,
            'aria-label': `Load a held-out handwritten ${digit}`,
          },
          () => nextSample(digit)
        )
      );
      const row = document.createElement('div');
      row.id = 'prob-' + digit;
      row.className = 'prob-row';
      row.innerHTML = `<span class="prob-digit">${digit}</span><div class="prob-track"><div class="prob-fill"></div></div><span class="prob-value"></span>`;
      $('probability-bars').append(row);
    }
    timelineGroups.forEach((group, groupIndex) => {
      const wrapper = document.createElement('div');
      wrapper.setAttribute('class', 'timeline-group');
      wrapper.setAttribute('data-timeline-group', groupIndex);
      const label = createButton(
        group.label,
        {
          class: 'timeline-group-label',
          'data-jump-step': group.start,
          title: `Jump to ${group.label}`,
        },
        () => gotoStep(group.start)
      );
      const track = document.createElement('div');
      track.setAttribute('class', 'timeline-group-track');
      for (let index = group.start; index <= group.end; index++) {
        const step = steps[index],
          button = createButton(
            '',
            {
              'data-step': index,
              'aria-label': `Step ${index + 1}: ${step.title}`,
              title: step.title,
            },
            () => gotoStep(index)
          );
        const dot = document.createElement('i');
        button.append(dot);
        if (step.block !== undefined) button.classList.add('block-step');
        track.append(button);
      }
      wrapper.append(label);
      wrapper.append(track);
      $('timeline').append(wrapper);
    });
    for (let channel = 0; channel < 8; channel++)
      $('channels').append(
        createButton(
          channel,
          {
            'data-channel': channel,
            'aria-label': `Inspect channel ${channel}`,
            'aria-pressed': false,
          },
          () => {
            state.channel = channel;
            renderView();
          }
        )
      );
    const meta = Data.meta;
    $('training-description').textContent =
      `scikit-learn digits: ${meta.trainCount.toLocaleString()} training, ${meta.validationCount} validation, ${meta.testCount} test images. Seed ${meta.seed}. ${meta.optimizer} (${meta.learningRate}), ${meta.epochs} epochs. Validation loss selected epoch ${meta.selectedEpoch}. Test: ${Data.metrics.testCorrect}/${meta.testCount} correct (${(Data.metrics.testAccuracy * 100).toFixed(2)}%).`;
    $('metric-accuracy').innerHTML =
      `${(Data.metrics.testAccuracy * 100).toFixed(2)}<span>%</span>`;
  }
  // UI events, keyboard shortcuts, and lifecycle.
  function bindUI() {
    $('next').addEventListener('click', () => gotoStep(state.step + 1));
    $('previous').addEventListener('click', () => gotoStep(state.step - 1));
    $('restart').addEventListener('click', restartJourney);
    $('play').addEventListener('click', togglePlay);
    $('speed').addEventListener('change', queueNext);
    $('new-sample').addEventListener('click', () => nextSample(state.sample.label));
    document
      .querySelectorAll('[data-block]')
      .forEach((button) =>
        button.addEventListener('click', () => selectBlock(Number(button.dataset.block)))
      );
    document
      .querySelectorAll('[data-expand-block]')
      .forEach((button) =>
        button.addEventListener('click', () => selectBlock(Number(button.dataset.expandBlock)))
      );
    document
      .querySelectorAll('[data-phase]')
      .forEach((button) =>
        button.addEventListener('click', () => selectPhase(button.dataset.phase))
      );
    $('replay-operation').addEventListener('click', replayOperation);
    $('show-clipped-cell').addEventListener('click', showClippedCell);
    $('inspect-block').addEventListener('click', () => selectBlock(state.selectedBlock));
    $('view-overview').addEventListener('click', collapseBlock);
    $('reset-camera').addEventListener('click', () => {
      if (state.expandedBlock >= 0) return;
      collapseBlock();
      scene.resetCamera();
    });
    $('zoom-in').addEventListener('click', () => scene.zoomBy(1.15));
    $('zoom-out').addEventListener('click', () => scene.zoomBy(1 / 1.15));
    $('labels-toggle').addEventListener('click', () => {
      state.showLabels = !state.showLabels;
      $('labels-toggle').textContent = 'Labels ' + (state.showLabels ? 'on' : 'off');
      $('labels-toggle').setAttribute('aria-pressed', state.showLabels);
      renderView();
    });
    $('pixel-row').addEventListener('input', (event) => {
      const row = Number(event.target.value);
      state.row = row;
      renderView();
    });
    $('pixel-col').addEventListener('input', (event) => {
      const col = Number(event.target.value);
      state.col = col;
      renderView();
    });
    $('network-scene').addEventListener('keydown', (event) => {
      if (
        state.expandedBlock < 0 ||
        !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)
      )
        return;
      event.preventDefault();
      event.stopPropagation();
      if (event.key === 'ArrowLeft') state.col = clamp(state.col - 1, 0, 7);
      if (event.key === 'ArrowRight') state.col = clamp(state.col + 1, 0, 7);
      if (event.key === 'ArrowUp') state.row = clamp(state.row - 1, 0, 7);
      if (event.key === 'ArrowDown') state.row = clamp(state.row + 1, 0, 7);
      renderView();
    });
    $('model-info').addEventListener('click', () => {
      pause();
      renderView();
      $('model-dialog').showModal();
    });
    $('close-model').addEventListener('click', () => $('model-dialog').close());
    $('help-button').addEventListener('click', () => {
      pause();
      renderView();
      $('help-dialog').showModal();
    });
    $('close-help').addEventListener('click', () => $('help-dialog').close());
    $('fullscreen').addEventListener('click', async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await $('playground').requestFullscreen();
      } catch (_) {
        $('fullscreen').title =
          'Fullscreen unavailable in this browser; use browser zoom to enlarge.';
      }
    });
    document.addEventListener('fullscreenchange', () => {
      $('fullscreen').setAttribute(
        'aria-label',
        document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen'
      );
      scene.resize();
    });
    document.addEventListener('keydown', (event) => {
      if (event.ctrlKey || event.altKey || event.metaKey || document.querySelector('dialog[open]'))
        return;
      if (event.key === 'Escape') {
        event.preventDefault();
        collapseBlock();
        $('network-scene').focus();
        return;
      }
      if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName) || event.target.isContentEditable)
        return;
      if (event.code === 'Space' && !event.target.closest('button, summary')) {
        event.preventDefault();
        togglePlay();
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault();
        gotoStep(state.step + 1);
      }
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        gotoStep(state.step - 1);
      }
      if (event.key.toLowerCase() === 'r') restartJourney();
      if (['1', '2', '3'].includes(event.key)) selectBlock(Number(event.key) - 1);
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        pause();
        renderView();
      }
    });
    motionQuery.addEventListener('change', (event) => {
      state.reducedMotion = event.matches;
      renderView();
    });
    window.addEventListener('pagehide', (event) => {
      pause();
      renderView();
      if (!event.persisted) scene.destroy();
    });
    window.addEventListener('pageshow', (event) => {
      if (event.persisted) scene.resize();
    });
  }
  // Application startup and read-only debugging API.
  try {
    buildUI();
    scene = new window.ResNetScene($('network-scene'), {
      onSelect: selectBlock,
      onPixelSelect: (pixel) => {
        state.row = clamp(pixel.row, 0, 7);
        state.col = clamp(pixel.col, 0, 7);
        renderView();
      },
      onOperationStage,
      onCameraChange: (camera) => {
        if (!syncingScene && camera.mode === 'overview' && state.expandedBlock >= 0) {
          state.expandedBlock = -1;
          state.mode = 'overview';
          renderView();
        }
      },
    });
    bindUI();
    compute();
    window.ResNetPlayground = {
      getSnapshot: () => ({
        step: state.step,
        stepCount: steps.length,
        blockPhase: phase(),
        selectedBlock: state.selectedBlock,
        channel: state.channel,
        row: state.row,
        col: state.col,
        sampleId: state.sample.id,
        label: state.sample.label,
        prediction: result.prediction,
        probabilities: Array.from(result.probabilities),
        playing: state.playing,
        mode: state.mode,
        expandedBlock: state.expandedBlock,
        operationStage: operation.stage,
      }),
    };
  } catch (error) {
    $('app-error').hidden = false;
    $('app-error').textContent =
      'The playground could not start. Please reload the page. Details: ' + error.message;
    console.error(error);
  }
})();
