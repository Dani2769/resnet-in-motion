/** Check authored JavaScript syntax and local runtime references without a browser. */
import { readdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = fileURLToPath(new URL('../', import.meta.url));

async function javascriptFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) return javascriptFiles(target);
      return /\.(?:js|mjs|cjs)$/.test(entry.name) ? [target] : [];
    })
  );
  return nested.flat();
}

const files = (
  await Promise.all(
    ['src', 'tools', 'tests', 'scripts'].map((directory) =>
      javascriptFiles(path.join(root, directory))
    )
  )
).flat();

for (const file of files) {
  const result = spawnSync(process.execPath, ['--check', file], { stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status || 1);
}

const html = await readFile(path.join(root, 'src/index.html'), 'utf8');
const references = [...html.matchAll(/(?:src|href)="([^"#]+)"/g)]
  .map((match) => match[1])
  .filter((value) => !/^(?:https?:|data:)/.test(value));
for (const reference of references) {
  await readFile(path.join(root, 'src', reference.split('?')[0]));
}

console.log(`Checked ${files.length} JavaScript files and ${references.length} local HTML assets.`);
