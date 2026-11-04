#!/usr/bin/env python3
"""Install the pinned torch wheel dependencies one at a time to limit disk peaks."""
from importlib.metadata import requires
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


def main():
    seen = set()
    for package in ('torch', 'torchvision'):
        for text in requires(package) or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            requirement.marker = None
            text = str(requirement)
            if text in seen:
                continue
            seen.add(text)
            print('INSTALL', text, flush=True)
            command = [sys.executable, '-I', '-m', 'pip', 'install', '--no-cache-dir']
            constraints = Path(__file__).resolve().parents[1] / 'environment/visual-local.constraints.txt'
            if constraints.is_file():
                command.extend(['-c', str(constraints)])
            # torch declares the complete pinned CUDA closure. Prevent cudnn's
            # unpinned cublas dependency from pulling a second, different build.
            if requirement.name.startswith('nvidia-') or requirement.name in {'triton', 'torch', 'torchvision'}:
                command.extend(['--no-deps', '--index-url', 'https://download.pytorch.org/whl/cu128'])
            subprocess.run([*command, text], check=True)


if __name__ == '__main__':
    main()
