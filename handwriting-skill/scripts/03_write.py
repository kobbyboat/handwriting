#!/usr/bin/env python3
"""
Step 3 - Write: render arbitrary text in the calibrated handwriting.

    python3 03_write.py "The intention is to create a 95% minimum replica" -o out.png
    python3 03_write.py --file lesson.txt -o lesson.pdf --paper ruled
    python3 03_write.py "Hello" -o big.png --x-height 90

Every letter is the writer's own bitmap (step 1), placed on a baseline grid with
the metrics measured in step 2, plus the variation a real hand produces:
per-letter size / rotation / vertical-offset jitter, letter-spacing jitter,
per-line baseline drift and a global slant.
"""
import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(HERE, '..', 'assets')

# characters we substitute when the writer's sheet has no sample for them
FALLBACK = {'\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
            '\u2014': '-', '\u2013': '-', '\u2026': '.'}


class Hand:
    def __init__(self, assets=DEFAULT_ASSETS, jitter=1.0, seed=0, slant=None,
                 upsample=2.0):
        self.assets = assets
        self.metrics = json.load(open(os.path.join(assets, 'metrics.json')))
        self.glyphs = json.load(open(os.path.join(assets, 'glyphs.json')))
        self.jitter = jitter
        self.rng = random.Random(seed)
        self.upsample = upsample                 # work at 2x, downsample for AA
        self.slant = math.radians(slant if slant is not None
                                 else self.metrics['paragraph'].get('slant_deg', 0.0))
        self._cache = {}

    # ------------------------------------------------------------------ glyphs
    def glyph(self, ch):
        """(alpha image, top_rel, bot_rel) for one character."""
        if ch in self._cache:
            return self._cache[ch]
        lst = self.glyphs.get(ch)
        if not lst:
            self._cache[ch] = None
            return None
        xh = self.metrics['paragraph']['x_height'] or 0
        m = min(lst, key=lambda g: abs((g.get('x_height') or xh) - xh))
        im = Image.open(os.path.join(self.assets, m['file'])).convert('L')
        im = im.point(lambda v: v if v > 110 else 0)
        box = im.getbbox()
        if box:
            im = im.crop(box)
        self._cache[ch] = (im, m['top_rel'], m['bot_rel'])
        return self._cache[ch]

    # ------------------------------------------------------------------ layout
    def _advance(self, ch, sx):
        g = self.glyph(ch)
        pad = 2 * self.metrics.get('pad', 4) * sx
        gap = max(0.5, self.metrics['paragraph']['letter_gap'] * sx - pad)
        if not g:
            return gap, gap
        w = g[0].width * sx
        return w, w + gap

    def layout(self, text, x_height, line_pitch, max_width=None):
        """[(char, x, baseline_y, sx)] laid out line by line."""
        met = self.metrics
        sx = x_height / met['letters']['x_height']
        # glyph crops carry a small transparent pad (metrics['pad'] px at sheet
        # scale), which would otherwise inflate every gap we add
        pad = 2 * met.get('pad', 4) * sx
        letter_gap = max(0.5, met['paragraph']['letter_gap'] * sx - pad)
        word_gap = max(letter_gap, met['paragraph']['word_gap'] * sx - pad)
        placed = []
        # first baseline: leave room for the tallest ascender plus a little air
        asc = met['letters']['ascender'] * (x_height / met['letters']['x_height'])
        y = asc * 1.05 + 0.25 * x_height
        x = 0.0
        for token in re.split(r'(\s+)', text):
            if token == '':
                continue
            if token.isspace():
                if '\n' * 2 in token or token.count('\n') >= 2:
                    y += line_pitch * (token.count('\n'))      # blank line(s)
                    x = 0.0
                elif '\n' in token:                            # hard line break
                    y += line_pitch
                    x = 0.0
                elif x > 0:                                     # word space
                    x += word_gap
                continue
            if max_width and x > 0 and x + self._word_width(token, sx) > max_width:
                y += line_pitch
                x = 0.0
            for ch in token:
                ch = FALLBACK.get(ch, ch)
                g = self.glyph(ch)
                if not g:
                    x += letter_gap
                    continue
                placed.append((ch, x, y, sx))
                x += g[0].width * sx + letter_gap * self.rng.uniform(0.75, 1.35)
        return placed, y

    def _word_width(self, word, sx):
        w = 0.0
        for ch in word:
            _, adv = self._advance(FALLBACK.get(ch, ch), sx)
            w += adv
        return w

    # ------------------------------------------------------------------ drawing
    def render(self, text, x_height=None, paper='white', color=None,
               width=None, margin=None, line_pitch=None, max_width=1500,
               noise=0.0):
        met = self.metrics
        x_height = x_height or met['letters']['x_height'] * met['scale']
        line_pitch = line_pitch or x_height * met['units']['line_pitch_over_x']
        color = tuple(color) if color else tuple(met['pen']['color'])
        missing = sorted({c for c in text if not c.isspace() and not self.glyph(FALLBACK.get(c, c))})
        if missing:
            print(f'[handwrite] no sample for: {" ".join(missing)} '
                  f'- left as a gap (add them to the sample sheet to render them)',
                  file=sys.stderr)
        place, last_y = self.layout(text, x_height, line_pitch, max_width)
        margin = margin if margin is not None else int(x_height * 1.5)

        text_w = max([x + self.glyph(c)[0].width * sx for c, x, _, sx in place] or [0])
        W = width or int(text_w + 2 * margin)
        top_pad = met['letters']['ascender'] * x_height / met['letters']['x_height']
        H = int(last_y + x_height * 1.2 + top_pad)
        W, H = max(W, 60), max(H, 60)

        paper_rgb = (253, 252, 250) if paper != 'white' else (255, 255, 255)
        canvas = Image.new('RGB', (W, H), paper_rgb)
        if paper == 'ruled':
            d = ImageDraw.Draw(canvas)
            y = margin * 0.35 + line_pitch
            while y < H - margin * 0.15:
                d.line([(margin * 0.25, y), (W - margin * 0.25, y)],
                       fill=(196, 208, 224), width=max(1, int(x_height * 0.035)))
                y += line_pitch

        up = self.upsample
        ink = Image.new('L', (int(W * up), int(H * up)), 0)
        rng = self.rng
        cur_y, drift, drift_x, drift_amp = None, 0.0, 0.0, 0.0

        for ch, x, y, sx in place:
            g = self.glyph(ch)
            if not g:
                continue
            im, top_rel, bot_rel = g
            if cur_y is None or abs(y - cur_y) > 1e-6:
                cur_y = y
                drift = rng.gauss(0, 0.012) * x_height           # whole-line wobble
                drift_amp = rng.uniform(0.0, 0.025) * x_height    # slow wave
                drift_x = x
            j = self.jitter
            s = sx * up * (1 + rng.gauss(0, 0.028) * j)
            rot = rng.gauss(0, 1.2) * j
            dy = rng.gauss(0, 0.018) * x_height * j
            wave = drift_amp * math.sin((x - drift_x) / (x_height * 9.0) + 0.6)

            w = max(1, int(round(im.width * s)))
            h = max(1, int(round(im.height * s)))
            gi = im.resize((w, h), Image.LANCZOS)
            # slant: shear the glyph about the baseline
            shear = math.tan(self.slant)
            if abs(shear) > 0.01:
                extra = int(abs(shear) * h)
                wide = Image.new('L', (w + extra, h), 0)
                wide.paste(gi, (extra if shear > 0 else 0, 0))
                gi = wide.transform((w + extra, h), Image.AFFINE,
                                    (1, shear, -shear * h, 0, 1, 0),
                                    resample=Image.BICUBIC)
                if shear < 0:
                    gi = gi.crop((0, 0, w + extra, h))
            if abs(rot) > 0.06:
                before = gi
                gi = gi.rotate(rot, resample=Image.BICUBIC, expand=False,
                               center=(gi.width / 2, gi.height * 0.72))
            gw, gh = gi.size
            gx = (margin + x) * up + (w - gw) / 2
            gy = (y + top_rel * sx + dy + drift + wave) * up + (h - gh) / 2
            ink.paste(gi, (int(round(gx)), int(round(gy))), gi)

        # ink texture: soften, then modulate with a fine grain so it reads as pen
        ink = ink.filter(ImageFilter.GaussianBlur(0.7 * up / 2))
        arr = np.asarray(ink).astype(np.float32) / 255.0
        if self.jitter > 0:
            grain = np.random.default_rng(7).normal(1.0, 0.10, arr.shape)
            arr = np.clip(arr * grain, 0, 1)
        alpha = Image.fromarray((arr * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        solid = Image.new('RGB', (W, H), color)
        out = Image.composite(solid, canvas, alpha)

        if noise > 0:                                  # optional scan-like noise
            a = np.asarray(out).astype(np.int16)
            a += np.random.default_rng(11).normal(0, noise, a.shape).astype(np.int16)
            out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text', nargs='?', default='')
    ap.add_argument('--file', help='read the text from a file')
    ap.add_argument('-o', '--out', default='writing.png')
    ap.add_argument('--assets', default=DEFAULT_ASSETS)
    ap.add_argument('--x-height', type=float, default=None,
                    help="target x-height in px (default: the writer's natural size)")
    ap.add_argument('--line-pitch', type=float, default=None)
    ap.add_argument('--paper', default='white', choices=['white', 'ruled'])
    ap.add_argument('--color', default=None, help='ink colour, e.g. #14142c')
    ap.add_argument('--width', type=int, default=None, help='page width in px')
    ap.add_argument('--max-width', type=int, default=1500, help='wrap after this many px')
    ap.add_argument('--jitter', type=float, default=1.0, help='0 = rigid, 1 = natural')
    ap.add_argument('--noise', type=float, default=0.0, help='scan-like grain (0-12)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--slant', type=float, default=None, help='override slant in degrees')
    a = ap.parse_args()

    text = a.text
    if a.file:
        text = open(a.file).read()
    if not text.strip():
        ap.error('no text given (pass it as an argument or with --file)')

    color = tuple(int(a.color[i:i + 2], 16) for i in (1, 3, 5)) if a.color else None
    hand = Hand(a.assets, jitter=a.jitter,
                seed=a.seed if a.seed is not None else random.randrange(1 << 30),
                slant=a.slant)
    img = hand.render(text, x_height=a.x_height, paper=a.paper, color=color,
                      width=a.width, max_width=a.max_width, line_pitch=a.line_pitch,
                      noise=a.noise)
    if a.out.lower().endswith('.pdf'):
        img.save(a.out, 'PDF', resolution=300.0)
    else:
        img.save(a.out)
    print(f'{a.out}  ({img.width}x{img.height})')


if __name__ == '__main__':
    main()
