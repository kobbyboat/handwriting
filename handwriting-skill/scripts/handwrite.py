#!/usr/bin/env python3
"""
handwrite.py - one entry point for the handwriting-replica skill.

    # 1. teach the skill a hand from a sample page (once per handwriting style)
    python3 handwrite.py learn --pdf page.pdf [--out ../assets]

    # 2. write anything in that hand
    python3 handwrite.py write "Any text you like" -o out.png
    python3 handwrite.py write --file lesson.txt -o lesson.pdf --paper ruled

    # 3. check what was learned (contact sheet of every letter)
    python3 handwrite.py inspect

Run `python3 handwrite.py <cmd> -h` for the full option list.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, '..', 'assets'))


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print('$', ' '.join(os.path.basename(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description='Handwriting replica skill')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('learn', help='extract letters + metrics from a sample page')
    p.add_argument('--pdf'); p.add_argument('--png')
    p.add_argument('--out', default=ASSETS)

    p = sub.add_parser('write', help='render text in the learned hand')
    p.add_argument('text', nargs='?', default='')
    p.add_argument('--file'); p.add_argument('-o', '--out', default='writing.png')
    p.add_argument('--assets', default=ASSETS)
    p.add_argument('--x-height', type=float, default=None)
    p.add_argument('--line-pitch', type=float, default=None)
    p.add_argument('--paper', default='white', choices=['white', 'ruled'])
    p.add_argument('--color', default=None)
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--max-width', type=int, default=1500)
    p.add_argument('--jitter', type=float, default=1.0)
    p.add_argument('--noise', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--slant', type=float, default=None)

    p = sub.add_parser('inspect', help='print the learned metrics')
    p.add_argument('--assets', default=ASSETS)

    a = ap.parse_args()
    if a.cmd == 'learn':
        src = ['--pdf', a.pdf] if a.pdf else ['--png', a.png]
        run('01_extract_glyphs.py', *src, '--out', a.out)
        run('02_calibrate.py', '--assets', a.out, *src)
        print(f'\nlearned -> {a.out}/metrics.json  (see atlas.png for the letters)')
    elif a.cmd == 'write':
        args = [a.text, '-o', a.out, '--assets', a.assets,
                '--paper', a.paper, '--max-width', str(a.max_width),
                '--jitter', str(a.jitter), '--noise', str(a.noise)]
        if a.file:
            args += ['--file', a.file]
        for flag, val in [('--x-height', a.x_height), ('--line-pitch', a.line_pitch),
                          ('--width', a.width), ('--seed', a.seed),
                          ('--slant', a.slant), ('--color', a.color)]:
            if val is not None:
                args += [flag, str(val)]
        run('03_write.py', *args)
    else:
        m = json.load(open(os.path.join(a.assets, 'metrics.json')))
        print(json.dumps({k: v for k, v in m.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
