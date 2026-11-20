/* Draws model activations as feature maps and animates their flow through the network. */
(function (root) {
  'use strict';
  const PALETTE = {
    ink: '#0b1017',
    text: '#f2f5fb',
    muted: '#a7b5c7',
    branch: '#b7a6ff',
    identity: '#5ee7cf',
    sum: '#f8ca7a',
    negative: '#f6a0ae',
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const color = (hex, alpha) => `rgba(${rgb(hex).join(',')},${clamp(alpha, 0, 1)})`;
  const mix = (a, b, t) =>
    `rgb(${rgb(a)
      .map((v, i) => Math.round(lerp(v, rgb(b)[i], clamp(t, 0, 1))))
      .join(',')})`;
  const BLOCKS = [-6.3, -2.2, 1.9];
  const PRESETS = {
    overview: { yaw: -0.42, pitch: 0.14, zoom: 1, target: -0.7, centerY: 0.57 },
    front: { yaw: -0.21, pitch: 0.04, zoom: 1, target: -0.7, centerY: 0.57 },
    top: { yaw: -0.65, pitch: 0.68, zoom: 0.95, target: -0.7, centerY: 0.57 },
  };
  const containsPoint = (points, x, y) => {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const a = points[i],
        b = points[j];
      if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x)
        inside = !inside;
    }
    return inside;
  };

  class ResNetScene {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d', { alpha: false });
      this.options = options;
      this.state = {
        step: -1,
        selectedBlock: -1,
        expandedBlock: -1,
        blockPhase: undefined,
        channel: 0,
        row: 3,
        col: 3,
        showLabels: true,
        showConnections: true,
        explode: 1,
        mode: 'overview',
        reducedMotion: false,
        playing: true,
      };
      this.camera = { ...PRESETS.overview };
      this.goal = { ...this.camera };
      this.data = null;
      this.width = 900;
      this.height = 460;
      this.hitTargets = [];
      this.drag = null;
      this.time = 0;
      this.lastTime = 0;
      this.alive = true;
      this.dirty = true;
      this.listeners = [];
      this.tensorColors = new WeakMap();
      this.lastRender = 0;
      this.animation = null;
      this.autoFitFocus = false;
      this.expansion = 0;
      this.expansionGoal = 0;
      this.expansionClip = null;
      this.expansionIndex = 0;
      this.pixelTargets = [];
      this.expandedPlanes = [];
      this.layerAlpha = 1;
      this.hoveredBlock = -1;
      this.operation = null;
      this.operationKey = null;
      this.dataVersion = 0;
      this.operationTokens = [];
      this.bindInteractions();
      if (typeof ResizeObserver !== 'undefined') {
        this.observer = new ResizeObserver(() => this.resize());
        this.observer.observe(canvas);
      }
      this.resize();
      this.frame = (stamp) => this.tick(stamp);
      this.wake();
    }

    bindInteractions() {
      const canvas = this.canvas;
      this.listen(canvas, 'pointerdown', (e) => {
        if (e.pointerType === 'touch') {
          this.drag = {
            x: e.clientX,
            y: e.clientY,
            startX: e.clientX,
            startY: e.clientY,
            touch: true,
            moved: false,
          };
          return;
        }
        if (e.button !== 0) return;
        this.drag = {
          x: e.clientX,
          y: e.clientY,
          startX: e.clientX,
          startY: e.clientY,
          moved: false,
        };
        canvas.setPointerCapture?.(e.pointerId);
        canvas.style.cursor = 'grabbing';
      });
      this.listen(canvas, 'pointermove', (e) => {
        if (!this.drag) {
          this.updateHover(e);
          return;
        }
        const dx = e.clientX - this.drag.x,
          dy = e.clientY - this.drag.y;
        this.drag.moved ||=
          Math.hypot(e.clientX - this.drag.startX, e.clientY - this.drag.startY) > 5;
        // One-finger gestures remain normal page scrolling. A tap selects a block.
        if (!this.drag.touch && this.drag.moved) {
          this.goal.yaw = clamp(this.goal.yaw + dx * 0.005, -1.23, 0.92);
          this.goal.pitch = clamp(this.goal.pitch + dy * 0.004, -0.42, 0.85);
          this.autoFitFocus = false;
          this.camera = { ...this.goal };
          this.dirty = true;
          this.wake();
          this.options.onCameraChange?.({ ...this.camera, mode: this.state.mode });
        }
        this.drag.x = e.clientX;
        this.drag.y = e.clientY;
      });
      const pointerEnd = (e) => {
        if (!this.drag) return;
        if (!this.drag.moved) {
          const rect = canvas.getBoundingClientRect(),
            x = e.clientX - rect.left,
            y = e.clientY - rect.top;
          const pixel =
            this.expansion > 0.75 &&
            this.pixelTargets
              .filter((t) => containsPoint(t.points, x, y))
              .sort((a, b) => a.depth - b.depth)[0];
          if (pixel) {
            this.state.row = pixel.row;
            this.state.col = pixel.col;
            this.options.onPixelSelect?.({ row: pixel.row, col: pixel.col, tensor: pixel.tensor });
            this.syncOperation();
            this.dirty = true;
            this.render();
            this.wake();
          } else if (this.expansion < 0.75) {
            const hit = this.hitTargets
              .filter((t) => x >= t.left && x <= t.right && y >= t.top && y <= t.bottom)
              .sort((a, b) => a.distance - b.distance)[0];
            if (hit) this.options.onSelect?.(hit.block);
          }
        }
        this.drag = null;
        canvas.style.cursor = 'grab';
      };
      this.listen(canvas, 'pointerleave', () => {
        if (!this.drag && this.hoveredBlock >= 0) {
          this.hoveredBlock = -1;
          this.dirty = true;
          this.wake();
        }
      });
      this.listen(canvas, 'pointerup', pointerEnd);
      this.listen(canvas, 'pointercancel', () => {
        this.drag = null;
        canvas.style.cursor = 'grab';
      });
      this.listen(
        canvas,
        'wheel',
        (e) => {
          if (!e.ctrlKey && !e.metaKey) return;
          e.preventDefault();
          this.zoomBy(Math.exp(-e.deltaY * 0.0014));
        },
        { passive: false }
      );
      this.listen(canvas, 'dblclick', () => this.resetCamera());
      canvas.style.cursor = 'grab';
      canvas.style.touchAction = 'pan-y';
    }

    tick(stamp) {
      if (!this.alive) return;
      this.animation = null;
      const elapsed = Math.max(0, stamp - (this.lastTime || stamp)),
        delta = Math.min(60, elapsed);
      this.lastTime = stamp;
      if (!this.state.reducedMotion) this.time += delta / 1000;
      let moving = this.advanceExpansion(elapsed, stamp);
      for (const key of ['yaw', 'pitch', 'zoom', 'target', 'centerY']) {
        if (Math.abs(this.camera[key] - this.goal[key]) > 0.0002) {
          this.camera[key] = lerp(
            this.camera[key],
            this.goal[key],
            this.state.reducedMotion ? 1 : 0.095
          );
          moving = true;
        }
      }
      const operating = this.advanceOperation(elapsed, stamp);
      const decorative = !this.state.reducedMotion && this.state.playing && this.state.step >= 0;
      if (this.dirty || moving || operating || (decorative && stamp - this.lastRender >= 32)) {
        this.render();
        this.lastRender = stamp;
      }
      if (this.dirty || moving || operating || decorative) this.wake();
    }

    listen(target, name, handler, options) {
      target.addEventListener(name, handler, options);
      this.listeners.push([target, name, handler, options]);
    }
    // Expansion and finite operation animations.
    startExpansion(goal) {
      this.expansionGoal = goal;
      this.expansionClip =
        this.state.reducedMotion || Math.abs(this.expansion - goal) < 0.001
          ? null
          : {
              from: this.expansion,
              to: goal,
              startedAt: root.performance?.now?.() ?? null,
              progress: 0,
              duration: 700,
            };
      if (!this.expansionClip) this.expansion = goal;
    }
    advanceExpansion(delta, stamp) {
      const clip = this.expansionClip;
      if (!clip) {
        this.expansion = this.expansionGoal;
        return false;
      }
      const progress =
        Number.isFinite(clip.startedAt) && Number.isFinite(stamp)
          ? (stamp - clip.startedAt) / clip.duration
          : clip.progress + delta / clip.duration;
      clip.progress = this.state.reducedMotion ? 1 : clamp(Math.max(clip.progress, progress), 0, 1);
      this.expansion = lerp(clip.from, clip.to, 1 - (1 - clip.progress) ** 3);
      this.dirty = true;
      if (clip.progress >= 1) {
        this.expansion = clip.to;
        this.expansionClip = null;
        return false;
      }
      return true;
    }
    currentPhase() {
      const phases = ['split', 'conv1', 'branch', 'add', 'relu'];
      if (this.state.blockPhase !== undefined)
        return phases.includes(this.state.blockPhase) ? this.state.blockPhase : null;
      return this.state.step >= 2 && this.state.step <= 16
        ? phases[(this.state.step - 2) % 5]
        : null;
    }
    phaseAlpha(name) {
      const phase = this.currentPhase();
      if (!phase) return 1;
      const rank = ['split', 'conv1', 'branch', 'add', 'relu'].indexOf(phase);
      const available = { input: 0, conv1: 1, branch: 2, sum: 3, output: 4 }[name];
      if (rank < available) return 0.16;
      if (phase === 'add' && name === 'sum' && this.operation?.stage === 'moving') return 0.2;
      if (phase === 'relu' && name === 'output' && this.operation?.stage !== 'complete') return 0.2;
      const focused =
        phase === 'split'
          ? ['input']
          : phase === 'conv1'
            ? ['input', 'conv1']
            : phase === 'branch'
              ? ['input', 'branch']
              : ['input', 'branch', 'sum', 'output'];
      return focused.includes(name) ? 1 : 0.6;
    }
    emitOperationStage(stage) {
      const operation = this.operation;
      if (!operation || operation.stage === stage || !this.alive) return;
      operation.stage = stage;
      this.options.onOperationStage?.({
        phase: operation.phase,
        stage,
        complete: stage === 'complete',
        block: operation.block,
        channel: operation.channel,
        row: operation.row,
        col: operation.col,
        data: this.data,
        x: operation.x,
        branch: operation.branch,
        sum: operation.sum,
        output: operation.output,
      });
    }
    syncOperation(force = false) {
      if (!this.alive) return false;
      const phase = this.currentPhase();
      if (!this.data || !this.isExpanded || !['add', 'relu'].includes(phase)) {
        this.operation = null;
        this.operationKey = null;
        return false;
      }
      const key = [
        phase,
        this.state.expandedBlock,
        this.state.channel,
        this.state.row,
        this.state.col,
        this.dataVersion,
      ].join('/');
      if (!force && key === this.operationKey) return true;
      const block = this.data.blocks[this.state.expandedBlock],
        index = this.state.channel * 64 + this.state.row * 8 + this.state.col;
      this.operationKey = key;
      this.operation = {
        phase,
        block: this.state.expandedBlock,
        channel: this.state.channel,
        row: this.state.row,
        col: this.state.col,
        x: (block.shortcut || block.input)[index],
        branch: (block.scaledBranch || block.branch)[index],
        sum: block.sum[index],
        output: block.output[index],
        duration: phase === 'add' ? 2000 : 1600,
        startedAt: root.performance?.now?.() ?? null,
        progress: this.state.reducedMotion ? 1 : 0,
        stage: null,
      };
      this.emitOperationStage(this.state.reducedMotion ? 'complete' : 'moving');
      this.dirty = true;
      this.wake();
      return true;
    }
    advanceOperation(delta, stamp) {
      const operation = this.operation;
      if (!operation || operation.progress >= 1) return false;
      // Frame timestamps and performance.now share a clock. Use elapsed time
      // so addition and ReLU finish on schedule even at low frame rates.
      const progress =
        Number.isFinite(operation.startedAt) && Number.isFinite(stamp)
          ? (stamp - operation.startedAt) / operation.duration
          : operation.progress + delta / operation.duration;
      operation.progress = this.state.reducedMotion
        ? 1
        : clamp(Math.max(operation.progress, progress), 0, 1);
      if (operation.progress >= 1) this.emitOperationStage('complete');
      else if (operation.phase === 'add' && operation.progress >= 0.72)
        this.emitOperationStage('combined');
      this.dirty = true;
      return operation.progress < 1;
    }
    replayOperation() {
      if (!this.syncOperation(true)) return false;
      this.render();
      this.wake();
      return true;
    }
    hitBlock(x, y) {
      return (
        this.hitTargets
          .filter((t) => x >= t.left && x <= t.right && y >= t.top && y <= t.bottom)
          .sort(
            (a, b) => Math.abs((a.left + a.right) / 2 - x) - Math.abs((b.left + b.right) / 2 - x)
          )[0]?.block ?? -1
      );
    }
    updateHover(event) {
      if (event.pointerType === 'touch') return;
      const rect = this.canvas.getBoundingClientRect(),
        x = event.clientX - rect.left,
        y = event.clientY - rect.top;
      const hovered = this.expansion < 0.02 && this.expansionGoal === 0 ? this.hitBlock(x, y) : -1;
      if (hovered !== this.hoveredBlock) {
        this.hoveredBlock = hovered;
        this.dirty = true;
        this.wake();
      }
      const cell =
        this.expansion > 0.75 && this.pixelTargets.some((t) => containsPoint(t.points, x, y));
      this.canvas.style.cursor = hovered >= 0 ? 'pointer' : cell ? 'crosshair' : 'grab';
    }
    drawHoverOutline(index) {
      const x = BLOCKS[index],
        corners = [];
      for (const yy of [-1.2, 3.25])
        for (const zz of [-1.1, 1.1])
          for (const xx of [x - 0.35, x + 3.35]) corners.push([xx, yy, zz]);
      for (let i = 0; i < 8; i++)
        for (const bit of [1, 2, 4])
          if (!(i & bit)) this.path([corners[i], corners[i | bit]], PALETTE.branch, 1, [], 0.45);
      const position = this.project(x + 1.3, 3.72, 0);
      this.text(
        'Click to unfold',
        clamp(position.x, 65, this.width - 65),
        clamp(position.y, 78, this.height - 36),
        { size: 12, color: PALETTE.identity, weight: 650, backing: true }
      );
    }
    wake() {
      if (this.alive && this.frame && this.animation === null)
        this.animation = requestAnimationFrame(this.frame);
    }
    // Data, state synchronization, and camera lifecycle.
    setData(result) {
      this.data = result;
      this.dataVersion++;
      this.syncOperation();
      this.dirty = true;
      this.render();
      this.wake();
    }
    setState(next) {
      const previous = this.state.expandedBlock;
      Object.assign(this.state, next);
      this.state.channel = clamp(this.state.channel | 0, 0, 7);
      this.state.row = clamp(this.state.row | 0, 0, 7);
      this.state.col = clamp(this.state.col | 0, 0, 7);
      if (next.expandedBlock !== undefined && previous !== next.expandedBlock) {
        if (next.expandedBlock >= 0) this.expandBlock(next.expandedBlock);
        else this.collapseBlock();
      }
      this.syncOperation();
      if (this.state.reducedMotion && this.operation && this.operation.progress < 1) {
        this.operation.progress = 1;
        this.emitOperationStage('complete');
      }
      this.dirty = true;
      if (this.state.reducedMotion) {
        this.expansion = this.expansionGoal;
        this.expansionClip = null;
        this.camera = { ...this.goal };
        this.render();
      }
      this.wake();
    }
    resize() {
      const rect = this.canvas.getBoundingClientRect();
      this.width = Math.max(260, rect.width || this.canvas.clientWidth || 900);
      this.height = Math.max(240, rect.height || this.canvas.clientHeight || 460);
      const dpr = Math.min(2, root.devicePixelRatio || 1);
      this.dpr = dpr;
      if (this.autoFitFocus && this.state.mode === 'focus') this.goal.zoom = this.focusZoom();
      if (this.autoFitFocus && this.state.mode === 'expanded') this.goal = this.expansionCamera();
      this.canvas.width = Math.round(this.width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.dirty = true;
      this.render();
      this.wake();
    }
    focusZoom() {
      return clamp(
        (this.height - 85) / 4.8 / Math.min(this.width / 18.5, this.height / 7.6),
        1.15,
        3.1
      );
    }
    resetCamera() {
      this.collapseBlock();
    }
    setCameraPreset(name) {
      if (name === 'focus') return this.focusBlock(Math.max(0, this.state.selectedBlock));
      if (PRESETS[name]) {
        this.autoFitFocus = false;
        this.goal = { ...PRESETS[name] };
        this.state.mode = 'overview';
        this.dirty = true;
        this.wake();
        this.options.onCameraChange?.({ ...this.goal, mode: this.state.mode });
      }
    }
    focusBlock(index) {
      index = clamp(index | 0, 0, 2);
      this.state.selectedBlock = index;
      this.state.mode = 'focus';
      this.autoFitFocus = true;
      this.goal = {
        yaw: -0.56,
        pitch: 0.21,
        zoom: this.focusZoom(),
        target: BLOCKS[index] + 1.3,
        centerY: 0.64,
      };
      this.dirty = true;
      this.wake();
      this.options.onCameraChange?.({ ...this.goal, mode: this.state.mode });
    }
    get isExpanded() {
      return this.state.expandedBlock >= 0;
    }
    expandedLayout() {
      const small = this.width < 620,
        tall = small && this.height >= 900;
      const top = Math.min(small ? 240 : this.width < 900 ? 230 : 200, this.height * 0.34),
        bottom = Math.max(top + 150, this.height - (small ? (tall ? 410 : 310) : 250));
      const minY = tall ? -5.05 : small ? -3.35 : -1.8,
        maxY = tall ? 4.4 : small ? 2.65 : 2.2;
      const scale = Math.min(
        (this.width - (small ? 30 : 65)) / (tall ? 6.8 : small ? 9.8 : 14.8),
        (bottom - top) / (maxY - minY)
      );
      const originY = (top + bottom) / 2 + ((minY + maxY) / 2) * scale;
      return { small, tall, top, bottom, scale, centerY: originY / this.height };
    }
    expansionCamera() {
      return {
        yaw: -0.14,
        pitch: 0.035,
        zoom: 1,
        target: 0,
        centerY: this.expandedLayout().centerY,
      };
    }
    expandBlock(index) {
      index = clamp(index | 0, 0, 2);
      this.hoveredBlock = -1;
      this.expansionIndex = index;
      this.startExpansion(1);
      this.state.expandedBlock = index;
      this.state.selectedBlock = index;
      this.state.mode = 'expanded';
      this.autoFitFocus = true;
      this.goal = this.expansionCamera();
      if (this.state.reducedMotion) {
        this.expansion = 1;
        this.camera = { ...this.goal };
      }
      this.syncOperation();
      this.dirty = true;
      this.render();
      this.wake();
      this.options.onCameraChange?.({ ...this.goal, mode: this.state.mode, expandedBlock: index });
    }
    collapseBlock() {
      this.operation = null;
      this.operationKey = null;
      this.hoveredBlock = -1;
      this.startExpansion(0);
      this.state.expandedBlock = -1;
      this.state.mode = 'overview';
      this.autoFitFocus = false;
      this.goal = { ...PRESETS.overview };
      if (this.state.reducedMotion) {
        this.expansion = 0;
        this.camera = { ...this.goal };
      }
      this.dirty = true;
      this.render();
      this.wake();
      this.options.onCameraChange?.({ ...this.goal, mode: this.state.mode, expandedBlock: -1 });
    }
    zoomBy(factor) {
      this.autoFitFocus = false;
      this.goal.zoom = clamp(this.goal.zoom * factor, 0.6, 3.4);
      this.dirty = true;
      this.wake();
      this.options.onCameraChange?.({ ...this.goal, mode: this.state.mode });
    }
    destroy() {
      this.alive = false;
      this.operation = null;
      this.operationKey = null;
      this.expansionClip = null;
      if (this.animation !== null) cancelAnimationFrame(this.animation);
      this.animation = null;
      this.observer?.disconnect();
      for (const [target, name, handler, options] of this.listeners)
        target.removeEventListener(name, handler, options);
      this.listeners = [];
    }

    // Projection and reusable canvas drawing primitives.
    project(x, y, z) {
      x -= this.camera.target;
      const c = Math.cos(this.camera.yaw),
        s = Math.sin(this.camera.yaw),
        cp = Math.cos(this.camera.pitch),
        sp = Math.sin(this.camera.pitch);
      const rx = c * x + s * z,
        rz = -s * x + c * z,
        ry = cp * y - sp * rz,
        depth = sp * y + cp * rz;
      const perspective = 25 / (25 + depth),
        scale = this.scale * perspective;
      return {
        x: this.width * 0.53 + rx * scale,
        y: this.height * (this.camera.centerY ?? 0.57) - ry * scale,
        depth,
        scale,
      };
    }
    path(points, stroke, width = 1, dash = [], alpha = 1) {
      const ctx = this.ctx;
      ctx.save();
      ctx.globalAlpha = alpha * this.layerAlpha;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.setLineDash(dash);
      ctx.beginPath();
      points.forEach((p, i) => {
        const a = Array.isArray(p) ? this.project(...p) : p;
        i ? ctx.lineTo(a.x, a.y) : ctx.moveTo(a.x, a.y);
      });
      ctx.stroke();
      ctx.restore();
    }
    text(value, x, y, options = {}) {
      const ctx = this.ctx;
      ctx.save();
      ctx.font = `${options.weight || 500} ${options.size || 11}px ${options.mono ? '"SFMono-Regular", Consolas, monospace' : 'Inter, ui-sans-serif, system-ui, sans-serif'}`;
      ctx.textAlign = options.align || 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = options.color || PALETTE.text;
      ctx.globalAlpha = (options.alpha ?? 1) * this.layerAlpha;
      if (options.backing) {
        const w = ctx.measureText(value).width,
          height = (options.size || 11) + 10;
        ctx.fillStyle = color('#101a27', 0.96);
        ctx.fillRect(x - w / 2 - 6, y - height / 2, w + 12, height);
        ctx.fillStyle = options.color || PALETTE.text;
      }
      ctx.fillText(value, x, y);
      ctx.restore();
    }
    segment(points, hex, active, alpha, phase = 0) {
      const screen = points.map((p) => this.project(...p));
      this.path(screen, color(hex, 0.12), active ? 6 : 3, [], alpha);
      this.path(screen, color(hex, active ? 0.92 : 0.45), active ? 1.65 : 1, [], alpha);
      if (!active || this.state.reducedMotion || !this.state.playing) return;
      const lengths = [];
      let total = 0;
      for (let i = 1; i < screen.length; i++) {
        const len = Math.hypot(screen[i].x - screen[i - 1].x, screen[i].y - screen[i - 1].y);
        lengths.push(len);
        total += len;
      }
      for (let dot = 0; dot < 3; dot++) {
        let distance = ((this.time * 0.19 + dot / 3 + phase) % 1) * total;
        for (let i = 0; i < lengths.length; i++) {
          if (distance > lengths[i]) {
            distance -= lengths[i];
            continue;
          }
          const t = distance / (lengths[i] || 1),
            p = screen[i],
            q = screen[i + 1];
          const ctx = this.ctx,
            x = lerp(p.x, q.x, t),
            y = lerp(p.y, q.y, t);
          ctx.save();
          ctx.globalAlpha = alpha * this.layerAlpha;
          ctx.fillStyle = hex;
          ctx.shadowBlur = 9;
          ctx.shadowColor = hex;
          ctx.beginPath();
          ctx.arc(x, y, 2.1, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
          break;
        }
      }
    }

    plane(list, tensor, x, options = {}) {
      if (!tensor?.length) return;
      const channels = Math.min(8, Math.max(1, Math.floor(tensor.length / 64))),
        selected = channels === 1 ? 0 : this.state.channel;
      const base = options.hex || PALETTE.branch,
        alpha = (options.alpha ?? 1) * this.layerAlpha;
      if (alpha <= 0) return;
      const limit = options.limit ?? Math.max(0.00001, ...Array.from(tensor, Math.abs)),
        gamma = options.gamma ?? 0.4;
      let cached = this.tensorColors.get(tensor);
      if (!cached || cached.base !== base || cached.limit !== limit || cached.gamma !== gamma) {
        cached = {
          base,
          limit,
          gamma,
          colors: Array.from(tensor, (value) =>
            mix(
              '#101a27',
              value < 0 ? PALETTE.negative : base,
              0.04 + Math.pow(Math.min(1, Math.abs(value) / Math.max(limit, 0.00001)), gamma) * 0.96
            )
          ),
        };
        this.tensorColors.set(tensor, cached);
      }
      const explosion = clamp(Number(this.state.explode) || 1, 0.35, 3),
        size = options.size || 1.65;
      const rotation = options.rotation || 0,
        u = [Math.sin(rotation), 0, Math.cos(rotation)],
        normal = [Math.cos(rotation), 0, -Math.sin(rotation)];
      const centerY = options.y ?? 0.045,
        centerZ = options.z || 0;
      const point = (horizontal, vertical, depth, stagger = 0) =>
        this.project(
          x + u[0] * horizontal + normal[0] * depth,
          centerY + vertical + stagger,
          centerZ + u[2] * horizontal + normal[2] * depth
        );
      const allCorners = [];
      const order = options.selectedOnly
        ? [selected]
        : Array.from({ length: channels }, (_, i) => i);
      if (options.selectedOnly && channels > 1) {
        for (let ghost = 7; ghost >= 1; ghost--) {
          const corners = [
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.5, -0.5],
          ].map(([h, v]) => point(h * size, v * size, -ghost * 0.035));
          list.push({
            depth: corners.reduce((n, p) => n + p.depth, 0) / 4 + 0.002,
            draw: () => {
              const ctx = this.ctx;
              ctx.save();
              ctx.globalAlpha = alpha * 0.18;
              ctx.strokeStyle = base;
              ctx.lineWidth = 0.6;
              ctx.beginPath();
              corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
              ctx.closePath();
              ctx.stroke();
              ctx.restore();
            },
          });
        }
      }
      for (const channel of order) {
        const slot = options.selectedOnly ? 0 : (channel - selected + channels) % channels;
        const depth = options.selectedOnly ? 0.01 : (slot - (channels - 1) / 2) * 0.074 * explosion;
        const stagger = options.selectedOnly ? 0 : slot * 0.013 * explosion;
        const corners = [
          [-0.5, -0.5],
          [-0.5, 0.5],
          [0.5, 0.5],
          [0.5, -0.5],
        ].map(([h, v]) => point(h * size, v * size, depth, stagger));
        allCorners.push(...corners);
        list.push({
          depth: corners.reduce((n, p) => n + p.depth, 0) / 4 + 0.002,
          draw: () => {
            const ctx = this.ctx;
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
            ctx.closePath();
            ctx.fillStyle = color(
              '#111b27',
              options.selectedOnly ? 0.97 : channel === selected ? 0.5 : 0.3
            );
            ctx.fill();
            ctx.strokeStyle = color(base, channel === selected ? 0.75 : 0.3);
            ctx.lineWidth = channel === selected ? 1 : 0.55;
            ctx.stroke();
            ctx.restore();
          },
        });
        const selectedAlpha = channels === 1 || channel === selected ? 1 : 0.55;
        for (let row = 0; row < 8; row++)
          for (let col = 0; col < 8; col++) {
            const cell = size / 8,
              gap = 0.015,
              left = -size / 2 + col * cell,
              top = size / 2 - row * cell;
            const points = [
              [left + gap, top - cell + gap],
              [left + gap, top - gap],
              [left + cell - gap, top - gap],
              [left + cell - gap, top - cell + gap],
            ].map(([h, v]) => point(h, v, depth + 0.003, stagger));
            const averageDepth = points.reduce((n, p) => n + p.depth, 0) / 4;
            const fill = cached.colors[channel * 64 + row * 8 + col];
            list.push({
              depth: averageDepth,
              draw: () => {
                const ctx = this.ctx;
                ctx.save();
                ctx.globalAlpha = alpha * selectedAlpha;
                ctx.fillStyle = fill;
                ctx.beginPath();
                points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
                ctx.closePath();
                ctx.fill();
                ctx.restore();
              },
            });
            if (options.pickable && channel === selected)
              this.pixelTargets.push({
                points,
                row,
                col,
                tensor: options.tensorName,
                depth: averageDepth,
              });
          }
      }
      if (Number.isInteger(options.block) && allCorners.length)
        this.hitTargets.push({
          block: options.block,
          left: Math.min(...allCorners.map((p) => p.x)) - 8,
          right: Math.max(...allCorners.map((p) => p.x)) + 8,
          top: Math.min(...allCorners.map((p) => p.y)) - 20,
          bottom: Math.max(...allCorners.map((p) => p.y)) + 30,
          distance: Math.abs(options.block - this.state.selectedBlock),
        });
    }

    // Frame composition.
    render() {
      if (!this.ctx) return;
      this.dirty = false;
      const ctx = this.ctx,
        w = this.width,
        h = this.height,
        s = this.state;
      ctx.setTransform(this.dpr || 1, 0, 0, this.dpr || 1, 0, 0);
      ctx.fillStyle = PALETTE.ink;
      ctx.fillRect(0, 0, w, h);
      this.layerAlpha = 1;
      this.scale =
        lerp(Math.min(w / 18.5, h / 7.6), this.expandedLayout().scale, this.expansion) *
        this.camera.zoom;
      const glow = ctx.createRadialGradient(w * 0.5, h * 0.52, 10, w * 0.5, h * 0.52, w * 0.65);
      glow.addColorStop(0, '#172132');
      glow.addColorStop(1, PALETTE.ink);
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, w, h);
      // A world-space floor helps the viewer read depth while orbiting.
      for (let x = -12; x <= 12; x += 2)
        this.path(
          [
            [x, -1.35, -5],
            [x, -1.35, 5],
          ],
          '#334457',
          0.45,
          [],
          0.13
        );
      for (let z = -4; z <= 4; z += 2)
        this.path(
          [
            [-12, -1.35, z],
            [12, -1.35, z],
          ],
          '#334457',
          0.45,
          [],
          0.13
        );
      this.hitTargets = [];
      this.pixelTargets = [];
      this.expandedPlanes = [];
      if (!this.data) {
        this.text('Preparing the network…', w / 2, h / 2, { color: PALETTE.muted, size: 13 });
        return;
      }
      this.layerAlpha = 1 - this.expansion;
      // Skip overview geometry once the transition has nearly faded it out.
      if (this.layerAlpha > 0.02) {
        const list = [],
          d = this.data,
          activeBlock = s.step >= 2 && s.step <= 16 ? Math.floor((s.step - 2) / 5) : -1;
        const focused =
          activeBlock >= 0 || s.mode === 'focus'
            ? s.selectedBlock >= 0 && s.selectedBlock <= 2
              ? s.selectedBlock
              : activeBlock
            : -1;
        const faded = s.mode === 'focus' ? 0.36 : 0.65;
        const alphaFor = (i) =>
          focused < 0 || focused === i || this.hoveredBlock === i ? 1 : faded;
        const tensorActive = (step) => s.step < 0 || s.step >= step;
        if (s.showConnections) {
          this.segment(
            [
              [-8.35, 0.03, 0],
              [-6.9, 0.03, 0],
            ],
            PALETTE.identity,
            s.step <= 1 && s.step >= 0,
            focused < 0 ? 1 : faded
          );
          this.segment(
            [
              [-6.7, 0.03, 0],
              [BLOCKS[0], 0.03, 0],
            ],
            PALETTE.branch,
            s.step === 2,
            alphaFor(0)
          );
          for (let i = 0; i < 3; i++) {
            const x = BLOCKS[i],
              alpha = alphaFor(i),
              blockStep = 2 + i * 5;
            const both = s.step >= blockStep && s.step <= blockStep + 4;
            const shortcutScale = d.shortcutScales?.[i] ?? 1,
              branchScale = d.branchScales?.[i] ?? 1;
            const shortcutPresent = Math.abs(shortcutScale) > 1e-9,
              branchPresent = Math.abs(branchScale) > 1e-9;
            const shortcutStrength = shortcutPresent
              ? 0.35 + 0.65 * Math.min(1, Math.abs(shortcutScale))
              : 0.1;
            const branchStrength = branchPresent
              ? 0.35 + 0.65 * Math.min(1, Math.abs(branchScale))
              : 0.1;
            const arch = [[x, 0.03, 0]];
            for (let k = 0; k <= 24; k++) {
              const t = k / 24;
              arch.push([x + 2.55 * t, 0.03 + 2.27 * Math.sin(Math.PI * t) ** 0.38, 0]);
            }
            this.segment(
              arch,
              PALETTE.identity,
              (both || this.hoveredBlock === i) && shortcutPresent,
              alpha * shortcutStrength,
              0.08
            );
            this.segment(
              [
                [x, 0.03, 0],
                [x + 0.6, 0.03, 0],
                [x + 1.5, 0.03, 0],
                [x + 2.55, 0.03, 0],
              ],
              PALETTE.branch,
              (both || this.hoveredBlock === i) && branchPresent,
              alpha * branchStrength,
              0.08
            );
            this.segment(
              [
                [x + 2.55, 0.03, 0],
                [x + 2.95, 0.03, 0],
                [i === 2 ? 6.2 : BLOCKS[i + 1], 0.03, 0],
              ],
              PALETTE.sum,
              s.step >= blockStep + 3 && s.step <= blockStep + 4,
              alpha
            );
          }
          this.segment(
            [
              [6.2, 0.03, 0],
              [7.5, 0.03, 0],
            ],
            PALETTE.sum,
            s.step >= 17,
            focused < 0 ? 1 : faded
          );
        }
        this.plane(list, d.input, -8.35, {
          hex: '#d7fff3',
          alpha: focused < 0 ? 1 : faded,
          size: 1.65,
        });
        this.plane(list, d.stem, -6.9, {
          hex: PALETTE.branch,
          alpha: (focused < 0 ? 1 : faded) * (tensorActive(1) ? 1 : 0.75),
        });
        for (let i = 0; i < 3; i++) {
          const x = BLOCKS[i],
            block = d.blocks[i],
            alpha = alphaFor(i),
            blockStep = 2 + i * 5;
          this.plane(list, block.conv1, x + 0.6, {
            block: i,
            hex: PALETTE.branch,
            alpha: alpha * (tensorActive(blockStep + 1) ? 1 : 0.75),
          });
          const branchPresent = Math.abs(d.branchScales?.[i] ?? 1) > 1e-9;
          this.plane(list, block.scaledBranch || block.branch, x + 1.5, {
            block: i,
            hex: PALETTE.branch,
            alpha: alpha * (tensorActive(blockStep + 2) ? 1 : 0.75) * (branchPresent ? 1 : 0.22),
          });
          this.plane(list, block.output, x + 2.95, {
            block: i,
            hex: PALETTE.sum,
            alpha: alpha * (tensorActive(blockStep + 4) ? 1 : 0.75),
          });
        }
        // Classifier's 512-vector is grouped for display; all 512 values are kept.
        const flat = d.flat || d.blocks[2].output,
          maxFlat = Math.max(0.00001, ...Array.from(flat, Math.abs));
        for (let k = 0; k < 64; k++) {
          let sum = 0;
          for (let c = 0; c < 8; c++) sum += Math.abs(flat[c * 64 + k] || 0);
          const strength = sum / (8 * maxFlat),
            y = -0.95 + (k / 63) * 2.1;
          const a = this.project(6.2, y, -0.28),
            b = this.project(6.2, y, 0.28);
          list.push({
            depth: a.depth,
            draw: () =>
              this.path(
                [a, b],
                mix('#233044', PALETTE.sum, strength ** 0.6),
                Math.max(1, this.scale * 0.018),
                [],
                focused < 0 ? 0.95 : faded
              ),
          });
        }
        list.sort((a, b) => b.depth - a.depth);
        for (const item of list) item.draw();
        ctx.globalAlpha = 1;
        // The sums are screen-facing markers, so their plus signs remain legible.
        for (let i = 0; i < 3; i++) {
          const x = BLOCKS[i],
            p = this.project(x + 2.55, 0.03, 0),
            r = clamp(p.scale * 0.17, 6, 14),
            alpha = alphaFor(i),
            active = s.step === 5 + i * 5 || s.step === 6 + i * 5;
          ctx.save();
          ctx.globalAlpha = alpha * this.layerAlpha;
          ctx.fillStyle = '#1e2530';
          ctx.strokeStyle = color(PALETTE.sum, active ? 1 : 0.8);
          ctx.lineWidth = active ? 2 : 1.2;
          if (active) {
            ctx.shadowColor = PALETTE.sum;
            ctx.shadowBlur = s.reducedMotion ? 0 : 12;
          }
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = PALETTE.sum;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.moveTo(p.x - r * 0.4, p.y);
          ctx.lineTo(p.x + r * 0.4, p.y);
          ctx.moveTo(p.x, p.y - r * 0.4);
          ctx.lineTo(p.x, p.y + r * 0.4);
          ctx.stroke();
          ctx.restore();
        }
        this.drawClassifier(focused < 0 ? 1 : faded);
        if (s.showLabels && this.expansion < 0.12) this.drawLabels(focused, activeBlock);
        if (this.hoveredBlock >= 0) this.drawHoverOutline(this.hoveredBlock);
      }
      if (this.expansion > 0) {
        this.layerAlpha = this.expansion;
        this.drawExpanded();
      }
      this.layerAlpha = 1;
      const vignette = ctx.createLinearGradient(0, 0, w, 0);
      vignette.addColorStop(0, color(PALETTE.ink, 0.78));
      vignette.addColorStop(0.06, color(PALETTE.ink, 0));
      vignette.addColorStop(0.94, color(PALETTE.ink, 0));
      vignette.addColorStop(1, color(PALETTE.ink, 0.78));
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, w, h);
    }

    flow(points, hex, active = true, alpha = 1) {
      this.segment(points, hex, active, alpha);
      const p = this.project(...points.at(-1)),
        q = this.project(...points.at(-2));
      const angle = Math.atan2(p.y - q.y, p.x - q.x),
        size = 6;
      this.path(
        [
          { x: p.x - Math.cos(angle - 0.45) * size, y: p.y - Math.sin(angle - 0.45) * size },
          p,
          { x: p.x - Math.cos(angle + 0.45) * size, y: p.y - Math.sin(angle + 0.45) * size },
        ],
        hex,
        1.4,
        [],
        alpha
      );
    }

    expandedNode(symbol, position, options = {}) {
      const p = this.project(...position),
        ctx = this.ctx;
      ctx.save();
      ctx.globalAlpha = this.layerAlpha * (options.alpha ?? 1);
      ctx.fillStyle = '#152231';
      ctx.strokeStyle = PALETTE.sum;
      ctx.lineWidth = options.active ? 2 : 1.3;
      if (options.active && !this.state.reducedMotion) {
        ctx.shadowColor = PALETTE.sum;
        ctx.shadowBlur = 10;
      }
      if (symbol === '+') {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(p.x - 5, p.y);
        ctx.lineTo(p.x + 5, p.y);
        ctx.moveTo(p.x, p.y - 5);
        ctx.lineTo(p.x, p.y + 5);
        ctx.stroke();
      } else {
        ctx.fillRect(p.x - 24, p.y - 13, 48, 26);
        ctx.strokeRect(p.x - 24, p.y - 13, 48, 26);
      }
      ctx.restore();
      if (symbol !== '+')
        this.text(symbol, p.x, p.y, {
          size: this.width < 620 ? 10 : 11,
          color: PALETTE.sum,
          weight: 650,
          alpha: Math.max(0.75, options.alpha ?? 1),
        });
    }

    routePoint(points, progress) {
      const screen = points.map((point) => this.project(...point)),
        lengths = [];
      let total = 0;
      for (let i = 1; i < screen.length; i++) {
        const length = Math.hypot(screen[i].x - screen[i - 1].x, screen[i].y - screen[i - 1].y);
        lengths.push(length);
        total += length;
      }
      let distance = clamp(progress, 0, 1) * total;
      for (let i = 0; i < lengths.length; i++) {
        if (distance > lengths[i] && i < lengths.length - 1) {
          distance -= lengths[i];
          continue;
        }
        const fraction = lengths[i] ? clamp(distance / lengths[i], 0, 1) : 0;
        return {
          x: lerp(screen[i].x, screen[i + 1].x, fraction),
          y: lerp(screen[i].y, screen[i + 1].y, fraction),
        };
      }
      return screen[0];
    }
    drawValuePill(label, position, hex, kind, value) {
      const ctx = this.ctx,
        small = this.width < 620,
        size = small ? 11 : 13,
        layout = this.expandedLayout();
      ctx.save();
      ctx.font = `600 ${size}px Consolas, monospace`;
      const width = ctx.measureText(label).width + 16,
        height = small ? 26 : 30;
      const x = clamp(position.x, width / 2 + 8, this.width - width / 2 - 8),
        y = clamp(position.y, layout.top + 16, layout.bottom - 16);
      ctx.globalAlpha = this.layerAlpha;
      ctx.fillStyle = '#0b1723';
      ctx.strokeStyle = hex;
      ctx.lineWidth = 1.2;
      ctx.fillRect(x - width / 2, y - height / 2, width, height);
      ctx.strokeRect(x - width / 2, y - height / 2, width, height);
      ctx.restore();
      this.text(label, x, y, { size, weight: 650, mono: true, color: hex });
      this.operationTokens.push({ kind, value, label, x, y });
    }
    drawOperation(stages, plus, relu, paths) {
      const operation = this.operation;
      if (!operation || operation.phase !== this.currentPhase()) return;
      const progress = operation.progress,
        small = this.width < 620,
        number = (value) => (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3);
      const cellPoint = (stage) => {
        const rotation = (Math.PI / 2) * this.expansion,
          horizontal = -stage.size / 2 + ((this.state.col + 0.5) * stage.size) / 8;
        const vertical = stage.size / 2 - ((this.state.row + 0.5) * stage.size) / 8;
        return [
          stage.position[0] + Math.sin(rotation) * horizontal + Math.cos(rotation) * 0.014,
          stage.position[1] + vertical,
          Math.cos(rotation) * horizontal - Math.sin(rotation) * 0.014,
        ];
      };
      if (operation.phase === 'add') {
        if (progress < 0.68) {
          const fraction = progress / 0.68;
          this.drawValuePill(
            `x ${number(operation.x)}`,
            this.routePoint([cellPoint(stages[0]), ...paths.identity], fraction),
            PALETTE.identity,
            'input',
            operation.x
          );
          this.drawValuePill(
            `F ${number(operation.branch)}`,
            this.routePoint([cellPoint(stages[2]), ...paths.branch], fraction),
            operation.branch < 0 ? PALETTE.negative : PALETTE.branch,
            'branch',
            operation.branch
          );
        } else {
          const point =
            progress < 0.95
              ? this.routePoint(
                  [plus, stages[3].position, cellPoint(stages[3])],
                  (progress - 0.68) / 0.27
                )
              : this.project(...plus);
          if (progress >= 0.95) point.y -= small ? 24 : 30;
          this.drawValuePill(
            `Σ ${number(operation.sum)}`,
            point,
            operation.sum < 0 ? PALETTE.negative : PALETTE.sum,
            'sum',
            operation.sum
          );
        }
      } else {
        if (progress < 0.58) {
          this.drawValuePill(
            `z ${number(operation.sum)}`,
            this.routePoint([cellPoint(stages[3]), stages[3].position, relu], progress / 0.58),
            operation.sum < 0 ? PALETTE.negative : PALETTE.sum,
            'sum',
            operation.sum
          );
        } else {
          const point =
            progress < 1
              ? this.routePoint(
                  [relu, stages[4].position, cellPoint(stages[4])],
                  (progress - 0.58) / 0.42
                )
              : this.project(...relu);
          if (progress >= 1) point.y -= small ? 25 : 30;
          this.drawValuePill(
            `y ${number(operation.output)}`,
            point,
            PALETTE.sum,
            'output',
            operation.output
          );
          const gate = this.project(...relu);
          this.text(
            operation.sum < 0 ? 'negative → 0' : 'keep the value',
            gate.x,
            gate.y + (small ? 31 : 35),
            {
              size: small ? 10 : 11,
              color: operation.sum < 0 ? PALETTE.negative : PALETTE.sum,
              backing: true,
            }
          );
        }
      }
    }
    drawExpanded() {
      const layout = this.expandedLayout(),
        small = layout.small,
        tall = layout.tall,
        index = this.expansionIndex,
        block = this.data.blocks[index],
        source = BLOCKS[index],
        t = this.expansion;
      const a = block.shortcut || block.input,
        b = block.scaledBranch || block.branch;
      const values = [a, b, block.sum, block.output],
        channel = this.state.channel,
        pixel = channel * 64 + this.state.row * 8 + this.state.col;
      const sharedLimit = Math.max(
        0.00001,
        ...values.flatMap((tensor) =>
          Array.from(tensor.slice(channel * 64, channel * 64 + 64), Math.abs)
        )
      );
      const positions = tall
        ? [
            [-1.75, 3.1],
            [1.75, 3.1],
            [1.75, 0],
            [-1.75, -3.1],
            [1.75, -3.1],
          ]
        : small
          ? [
              [-3.4, 1.45],
              [0, 1.45],
              [3.4, 1.45],
              [-0.2, -1.5],
              [3.4, -1.5],
            ]
          : [
              [-6.2, 0],
              [-3.4, 0],
              [-0.6, 0],
              [2.6, 0],
              [5.6, 0],
            ];
      const names = ['input', 'conv1', 'branch', 'sum', 'output'],
        tensors = [a, block.conv1, b, block.sum, block.output];
      const titles = small
        ? ['Input x', 'Conv 1', 'Change F(x)', 'Sum x + F', 'Output y']
        : ['Input · x', 'Conv 1 + ReLU', 'Learned change · F(x)', 'Sum · x + F(x)', 'Output · y'];
      const hexes = [PALETTE.identity, PALETTE.branch, PALETTE.branch, PALETTE.sum, PALETTE.sum],
        offsets = [0, 0.6, 1.5, 2.55, 2.95];
      const world = (target, offset) => [
        lerp(source + offset, target[0], t),
        lerp(0.045, target[1], t),
        0,
      ];
      const stages = positions.map((position, i) => ({
        name: names[i],
        tensor: tensors[i],
        title: titles[i],
        hex: hexes[i],
        position: world(position, offsets[i]),
        size: lerp(1.65, small && !tall && i === 1 ? 1.25 : 1.85, t),
      }));
      const plus = world(tall ? [-1.75, 0] : small ? [-2.25, -1.5] : [1.1, 0], 2.55),
        relu = world(tall ? [0, -3.1] : small ? [1.55, -1.5] : [4.15, 0], 2.85);
      const phase = this.currentPhase(),
        rank = phase ? ['split', 'conv1', 'branch', 'add', 'relu'].indexOf(phase) : 4;
      const paths = {
        branch: tall
          ? [stages[2].position, plus]
          : small
            ? [
                stages[2].position,
                world([4.65, 1.45], 1.8),
                world([4.65, 0.02], 2.0),
                world([-2.25, 0.02], 2.3),
                plus,
              ]
            : [stages[2].position, plus],
        identity: [],
      };
      if (tall)
        paths.identity = [stages[0].position, world([-3.0, 3.1], 0), world([-3.0, 0], 2.4), plus];
      else if (small)
        paths.identity = [
          stages[0].position,
          world([-4.65, 1.45], 0),
          world([-4.65, -1.5], 2.4),
          plus,
        ];
      else
        for (let k = 0; k <= 32; k++) {
          const q = k / 32;
          paths.identity.push([
            lerp(stages[0].position[0], plus[0], q),
            0.045 + 1.65 * Math.sin(Math.PI * q) ** 0.5,
            0,
          ]);
        }
      if (this.state.showConnections) {
        this.flow(
          [stages[0].position, stages[1].position],
          PALETTE.branch,
          phase === 'split' || phase === 'conv1',
          rank >= 0 ? 0.9 : 0.2
        );
        this.flow(
          [stages[1].position, stages[2].position],
          PALETTE.branch,
          phase === 'branch',
          rank >= 1 ? 0.85 : 0.15
        );
        this.flow(paths.branch, PALETTE.branch, phase === 'add', rank >= 2 ? 1 : 0.15);
        this.flow(paths.identity, PALETTE.identity, phase === 'split' || phase === 'add', 1);
        this.flow([plus, stages[3].position], PALETTE.sum, phase === 'add', rank >= 3 ? 1 : 0.15);
        this.flow(
          [stages[3].position, relu, stages[4].position],
          PALETTE.sum,
          phase === 'relu',
          rank >= 4 ? 1 : 0.15
        );
      }
      const list = [];
      for (const stage of stages) {
        const localLimit =
          stage.name === 'conv1'
            ? Math.max(
                0.00001,
                ...Array.from(stage.tensor.slice(channel * 64, channel * 64 + 64), Math.abs)
              )
            : sharedLimit;
        const alpha = this.phaseAlpha(stage.name);
        this.expandedPlanes.push({ ...stage, limit: localLimit, alpha });
        this.plane(list, stage.tensor, stage.position[0], {
          y: stage.position[1],
          z: 0,
          size: stage.size,
          rotation: (Math.PI / 2) * t,
          hex: stage.hex,
          alpha,
          selectedOnly: true,
          pickable: true,
          tensorName: stage.name,
          limit: localLimit,
          gamma: 1,
        });
      }
      list.sort((a, b) => b.depth - a.depth);
      for (const item of list) item.draw();
      this.expandedNode('+', plus, { alpha: rank >= 3 ? 1 : 0.3, active: phase === 'add' });
      this.expandedNode('ReLU', relu, { alpha: rank >= 4 ? 1 : 0.2, active: phase === 'relu' });
      this.operationTokens = [];
      this.drawOperation(stages, plus, relu, paths);
      // Keep the selected-cell annotation readable from both faces of every map.
      const ctx = this.ctx;
      ctx.save();
      for (const cell of this.pixelTargets)
        if (cell.row === this.state.row && cell.col === this.state.col) {
          const alpha = this.phaseAlpha(cell.tensor);
          ctx.globalAlpha = this.layerAlpha * alpha;
          ctx.beginPath();
          cell.points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
          ctx.closePath();
          if (alpha >= 0.3) {
            ctx.strokeStyle = '#07101b';
            ctx.lineWidth = 4.6;
            ctx.stroke();
          }
          ctx.strokeStyle = '#f4f7ff';
          ctx.lineWidth = 1.7;
          ctx.stroke();
        }
      ctx.restore();
      if (this.state.showLabels) {
        for (const stage of stages) {
          const p = this.project(stage.position[0], stage.position[1] - stage.size / 2 - 0.32, 0),
            alpha = this.phaseAlpha(stage.name),
            future = rank < names.indexOf(stage.name);
          const title =
            stage.name === 'conv1'
              ? small
                ? 'Conv 1'
                : 'Conv 1 + ReLU'
              : stage.name === 'branch'
                ? 'Change F(x)'
                : stage.title;
          this.text(title, p.x, p.y, {
            size: small ? 10.5 : 12,
            color: future ? PALETTE.muted : stage.hex,
            weight: 650,
            backing: true,
            alpha: Math.max(0.85, alpha),
          });
          this.text(
            future ? 'Later' : alpha >= 0.3 ? stage.tensor[pixel].toFixed(3) : '…',
            p.x,
            p.y + 18,
            {
              size: small ? 10.5 : 11,
              color: alpha >= 0.3 ? PALETTE.text : PALETTE.muted,
              mono: true,
              backing: true,
              alpha: 0.95,
            }
          );
        }
        const identity = this.project(
          ...(tall
            ? world([-3.0, 1.6], 1.1)
            : small
              ? world([-3.65, -0.35], 1.1)
              : world([-2.55, 1.92], 1.2))
        );
        this.text(small ? 'Keep x' : 'Shortcut keeps x', identity.x, identity.y, {
          size: small ? 10 : 12,
          color: PALETTE.identity,
          backing: true,
        });
      }
    }

    drawClassifier(alpha) {
      const ctx = this.ctx,
        probs = this.data.probabilities,
        top = this.data.prediction,
        points = [];
      for (let i = 0; i < 10; i++) points.push(this.project(7.5, 1.35 - i * 0.29, 0));
      for (let i = 0; i < 10; i++) {
        const p = points[i],
          probability = probs[i] || 0,
          active = i === top && (this.state.step >= 18 || this.state.step < 0),
          r = clamp(p.scale * 0.06, 2, 5);
        ctx.save();
        ctx.globalAlpha = alpha * this.layerAlpha;
        ctx.fillStyle = active ? PALETTE.sum : color(PALETTE.sum, 0.16 + probability * 0.75);
        ctx.beginPath();
        ctx.arc(p.x, p.y, active ? r + 1 : r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        if (this.width >= 560 || active)
          this.text(String(i), p.x + 13, p.y, {
            size: 11,
            mono: true,
            color: active ? PALETTE.sum : PALETTE.muted,
            alpha: Math.max(0.8, alpha),
          });
      }
      if (this.state.step >= 18 || this.state.step < 0) {
        const p = this.project(8.4, 0.02, 0);
        this.text(String(top), p.x + 10, p.y - 8, {
          size: clamp(p.scale * 0.8, 22, 46),
          weight: 650,
          color: PALETTE.sum,
          alpha,
        });
        if (this.width >= 620)
          this.text(`${((probs[top] || 0) * 100).toFixed(1)}%`, p.x + 10, p.y + 24, {
            size: 11,
            mono: true,
            color: PALETTE.muted,
            alpha: Math.max(0.8, alpha),
          });
      }
    }

    drawLabels(focused, activeBlock) {
      const small = this.width < 900,
        detail = !small || this.camera.zoom > 1.5;
      const stage = (x, name, sub, hex, alpha = 1) => {
        const p = this.project(x, -1.65, 0);
        if (p.x < -30 || p.x > this.width + 30) return;
        const y = p.y + (this.width < 560 && (x === -6.9 || x === 6.2) ? 20 : 0);
        this.text(name, p.x, y, {
          size: small ? 11 : 12,
          color: hex,
          alpha: Math.max(0.75, alpha),
          weight: 650,
          backing: true,
        });
        if (!small)
          this.text(sub, p.x, y + 21, {
            size: 11,
            color: PALETTE.muted,
            mono: true,
            alpha: Math.max(0.8, alpha),
          });
      };
      const faded = this.state.mode === 'focus' ? 0.36 : 0.65;
      stage(-8.35, 'Input', '1 × 8 × 8', '#d7fff3', focused < 0 ? 1 : faded);
      stage(-6.9, 'Stem', '8 feature maps', PALETTE.branch, focused < 0 ? 1 : faded);
      for (let i = 0; i < 3; i++) {
        const x = BLOCKS[i],
          alpha = focused < 0 || focused === i ? 1 : faded,
          middle = this.project(x + 1.3, 3.02, 0),
          selected = focused === i || this.hoveredBlock === i;
        this.text(
          this.width < 650 ? `Block ${i + 1}` : `Residual block ${i + 1}`,
          middle.x,
          middle.y,
          {
            color: selected ? PALETTE.text : PALETTE.muted,
            size: small ? 11 : 13,
            weight: selected ? 750 : 650,
            alpha: Math.max(0.8, alpha),
            backing: true,
          }
        );
        if (detail || (selected && this.width >= 650)) {
          const arch = this.project(x + 1.3, 2.47, 0);
          this.text('Keep x', arch.x, arch.y, {
            color: PALETTE.identity,
            size: 11,
            alpha: Math.max(0.8, alpha),
            backing: true,
          });
          stage(x + 0.6, 'Conv 1', 'ReLU', PALETTE.branch, alpha);
          stage(x + 1.5, 'Conv 2', 'F(x)', PALETTE.branch, alpha);
          stage(x + 2.95, 'Σ → ReLU', '8 × 8 × 8', PALETTE.sum, alpha);
        } else if (this.width >= 650)
          stage(x + 1.5, `Block ${i + 1}`, '', selected ? PALETTE.text : PALETTE.muted, alpha);
        // An interaction zone includes the block's heading and arch.
        const left = this.project(x, 0.8, 0),
          right = this.project(x + 3.2, 0.8, 0),
          top = this.project(x + 1.3, 3.2, 0),
          bottom = this.project(x + 1.3, -1.6, 0);
        this.hitTargets.push({
          block: i,
          left: Math.min(left.x, right.x),
          right: Math.max(left.x, right.x),
          top: top.y - 6,
          bottom: bottom.y + 12,
          distance: Math.abs(middle.x - this.width * 0.5) + 10,
        });
        if (activeBlock === i && this.state.mode !== 'focus' && this.hoveredBlock !== i) {
          const p = this.project(x + 1.3, 3.48, 0);
          this.text('Current block', p.x, p.y, {
            size: 11,
            color: PALETTE.identity,
            weight: 650,
            backing: true,
          });
        }
      }
      if (this.width < 1300)
        stage(7.0, 'Classifier', '512 → 10 scores', PALETTE.sum, focused < 0 ? 1 : faded);
      else {
        stage(6.2, 'Flatten', '512 values', PALETTE.sum, focused < 0 ? 1 : faded);
        stage(7.65, 'Predict', '10 classes', PALETTE.sum, focused < 0 ? 1 : faded);
      }
    }
  }
  root.ResNetScene = ResNetScene;
  if (typeof module !== 'undefined' && module.exports) module.exports = { ResNetScene, PALETTE };
})(typeof window !== 'undefined' ? window : globalThis);
