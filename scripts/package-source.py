"""Build and package the reproducible source, excluding caches and credentials."""

from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'ResNet_Playground_Source.zip'
DIRECTORIES = ('src', 'dist', 'tools', 'scripts', 'tests', 'reports')
ROOT_FILES = ('README.md', 'package.json', 'package-lock.json', '.editorconfig',
              '.prettierrc.json', '.prettierignore', '.gitignore', 'ResNet_3D_Playground.html')


def package():
    subprocess.run([sys.executable, str(ROOT / 'scripts/build-playground.py')], check=True)
    files = [ROOT / name for name in ROOT_FILES]
    for name in DIRECTORIES:
        files.extend(path for path in (ROOT / name).rglob('*')
                     if path.is_file() and not any(part.startswith('.') or part == '__pycache__'
                                                  for part in path.relative_to(ROOT / name).parts)
                     and path.suffix not in ('.pyc', '.pyo'))
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED) as archive:
        for file in sorted(files):
            archive.write(file, Path('resnet-shortcuts') / file.relative_to(ROOT))
    with ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError('Source archive integrity check failed.')
        print(f'Packaged {len(archive.namelist())} files into {OUTPUT.name}.')


if __name__ == '__main__':
    package()
