"""Build the static website and portable offline HTML from the same src/ tree.

Run: python scripts/build-playground.py
Only the generated dist/ directory is replaced. Model weights are never trained.
"""

from html.parser import HTMLParser
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
DIST = ROOT / 'dist'
OUTPUT = ROOT / 'ResNet_3D_Playground.html'


class RuntimeAssets(HTMLParser):
    """Read runtime dependencies in their declared HTML loading order."""

    def __init__(self):
        super().__init__()
        self.stylesheets = []
        self.scripts = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == 'link' and attributes.get('rel') == 'stylesheet':
            self.stylesheets.append(attributes['href'])
        if tag == 'script' and attributes.get('src'):
            self.scripts.append(attributes['src'])


def read_asset(reference):
    path = (SOURCE / reference.split('?')[0]).resolve()
    if not path.is_relative_to(SOURCE.resolve()) or not path.is_file():
        raise ValueError(f'Runtime asset must be a file within src/: {reference}')
    return path.read_text(encoding='utf-8')


def replace_once(source, pattern, replacement):
    source, count = re.subn(pattern, lambda _: replacement, source, flags=re.IGNORECASE)
    if count != 1:
        raise ValueError(f'Expected exactly one match for {pattern}, found {count}.')
    return source


def bundle_offline(html, assets):
    for reference in assets.stylesheets:
        css = read_asset(reference)
        # The offline edition uses system fonts and makes no font requests.
        css = re.sub(r'@import\s+(?:url\([^)]*\)|["\'][^"\']*["\'])\s*;\s*', '', css, flags=re.IGNORECASE)
        if re.search(r'url\(\s*["\']?https?://', css, re.IGNORECASE):
            raise ValueError('An external CSS asset remains in the offline bundle.')
        pattern = r'<link\b(?=[^>]*\brel=["\']stylesheet["\'])(?=[^>]*\bhref=["\']' + re.escape(reference) + r'["\'])[^>]*>'
        html = replace_once(html, pattern, '<style>\n' + css + '\n</style>')

    embedded = []
    for reference in assets.scripts:
        pattern = r'<script\b(?=[^>]*\bsrc=["\']' + re.escape(reference) + r'["\'])[^>]*>\s*</script\s*>'
        html = replace_once(html, pattern, '')
        javascript = re.sub(r'</script', r'<\\/script', read_asset(reference), flags=re.IGNORECASE)
        name = reference.split('?')[0]
        embedded.append(f'<script>\n/* Embedded {name} */\n{javascript}\n</script>')

    if re.search(r'<script\b[^>]*\bsrc=|<link\b[^>]*\brel=["\']stylesheet', html, re.IGNORECASE):
        raise ValueError('An external runtime dependency remains in the offline bundle.')
    return replace_once(html, r'</body\s*>', '\n' + '\n'.join(embedded) + '\n</body>')


def build():
    html = (SOURCE / 'index.html').read_text(encoding='utf-8')
    assets = RuntimeAssets()
    assets.feed(html)
    if not assets.stylesheets or not assets.scripts:
        raise ValueError('src/index.html must declare the runtime stylesheet and scripts.')
    offline = bundle_offline(html, assets)

    # Refuse recursive replacement if dist resolves outside this project.
    if DIST.is_symlink() or DIST.resolve() != ROOT.resolve() / 'dist':
        raise ValueError('Refusing to replace an unexpected build directory.')
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(SOURCE, DIST)
    OUTPUT.write_text(offline, encoding='utf-8', newline='\n')
    print(f'Built dist/ and {OUTPUT.name}: {OUTPUT.stat().st_size:,} bytes; '
          f'{len(assets.stylesheets)} stylesheet and {len(assets.scripts)} scripts embedded.')


if __name__ == '__main__':
    build()
