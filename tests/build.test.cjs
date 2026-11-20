const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const read = (name) => fs.readFileSync(path.join(root, name), 'utf8');

test('the website and offline edition ship the same authored runtime in loading order', () => {
  const source = read('src/index.html');
  const offline = read('ResNet_3D_Playground.html');
  const scripts = [...source.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*>\s*<\/script>/g)].map(
    (match) => match[1].split('?')[0]
  );
  assert.deepEqual(scripts, [
    'model/model-data.js',
    'model/inference.js',
    'app/lesson.js',
    'rendering/scene.js',
    'app/controller.js',
  ]);
  assert.equal(read('dist/index.html'), source);
  const embedded = [
    ...offline.matchAll(/<script>\n\/\* Embedded ([^\n]+) \*\/\n([\s\S]*?)\n<\/script>/g),
  ];
  assert.deepEqual(
    embedded.map((match) => match[1]),
    scripts
  );
  for (const [index, name] of scripts.entries()) {
    const javascript = read('src/' + name);
    assert.equal(read('dist/' + name), javascript);
    assert.equal(embedded[index][2], javascript.replace(/<\/script/gi, '<\\/script'));
  }
  const css = read('src/styles/playground.css');
  assert.equal(read('dist/styles/playground.css'), css);
  const offlineCss = [...offline.matchAll(/<style>\n([\s\S]*?)\n<\/style>/g)];
  assert.equal(offlineCss.length, 1);
  assert.equal(
    offlineCss[0][1],
    css.replace(/@import\s+(?:url\([^)]*\)|["'][^"']*["'])\s*;\s*/gi, '')
  );
  assert.doesNotMatch(offline, /<script\b[^>]*\bsrc=|<link\b[^>]*\brel="stylesheet"/i);
  assert.doesNotMatch(offlineCss[0][1], /https?:\/\//);
  assert.ok(!fs.existsSync(path.join(root, 'dist/kernel-inspector.js')));
});

test('the runtime loads its model and lesson APIs as browser scripts without CommonJS', () => {
  const window = {};
  const context = vm.createContext({ window });
  for (const name of ['model/model-data.js', 'model/inference.js', 'app/lesson.js']) {
    vm.runInContext(read('dist/' + name), context, { filename: name });
  }
  const sample = window.ResNetData.samples.find((value) => value.label === 6);
  const result = window.ResNetEngine.forward(sample.pixels);
  assert.equal(result.prediction, 6);
  assert.equal(window.ResNetLesson.steps.length, 19);
  assert.equal(window.ResNetLesson.blockStep(2, 'relu'), 16);
  assert.equal(window.ResNetLesson.nextDestination(16), 'classifier');
  assert.equal(window.ResNetLesson.nextDestination(18), null);
});
