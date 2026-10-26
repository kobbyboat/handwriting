#!/usr/bin/env python3
"""
Step 2 - Calibrate: measure the writer's metrics from the sample page.

Reads assets/glyphs.json + assets/ink_mask.png (+ the raw page for colour) and
writes assets/metrics.json with everything the renderer needs:

    pen        ink colour + median stroke thickness
    letters    x-height, cap-height, ascender, descender (px, measured on the
               spaced-out alphabet block == the writer's "natural" size)
    glyphs     per-character top/bottom offsets relative to the baseline
    rhythm     line pitch, letter gap, word gap, slant  (measured on the free
               paragraph, i.e. the writer's natural flow)
    scale      paragraph x-height / alphabet x-height - how much smaller the
               writer writes when writing normally instead of demonstrating

Usage:
    python3 02_calibrate.py --assets ../assets --pdf ../../uploads/myhandwriting.pdf
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

NO_DESCENDER = set('ACEIMNORSUVWXZacemnorsuvwxz')   # sit on the baseline
NO_ASCENDER = set('acemnorsuvwxz')                  # define the x-height


def load_rgb(pdf=None, png=None):
    if pdf:
        import pymupdf
        doc = pymupdf.open(pdf)
        page = doc[0]
        best, pix = -1, None
        for img in page.get_images(full=True):
            if img[2] * img[3] > best:
                best, pix = img[2] * img[3], pymupdf.Pixmap(doc, img[0])
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return arr[:, :, :3]
    return np.array(Image.open(png).convert('RGB'))


def row_metrics(glyphs, row):
    """baseline / x-height / cap-height of one alphabet row."""
    items = [(ch, m) for ch, lst in glyphs.items() for m in lst if m['row'] == row]
    baselines = [m['baseline'] for ch, m in items if ch in NO_DESCENDER]
    baseline = float(np.median(baselines)) if baselines else float(np.median([m['baseline'] for _, m in items]))
    xs = [m['h'] for ch, m in items if ch in NO_ASCENDER]
    caps = [m['h'] for ch, m in items if ch.isupper()]
    return dict(baseline=baseline,
                x_height=float(np.median(xs)) if xs else None,
                cap_height=float(np.median(caps)) if caps else None)


def paragraph_metrics(ink, bands, skip=4):
    """Letter gap, word gap, x-height and line pitch from the free writing."""
    gaps_all, pitches, xh = [], [], []
    baselines = []
    for y0, y1 in bands[skip:]:
        line = ink[y0:y1]
        if line.sum() == 0:
            continue
        prof = line.sum(0).astype(float)
        on = prof > 0
        # runs of empty columns
        runs, st = [], None
        for x, v in enumerate(on):
            if not v and st is None:
                st = x
            elif v and st is not None:
                runs.append((st, x)); st = None
        inner = [r[1] - r[0] for r in runs if r[0] > 0 and r[1] < len(on)]
        # ink blobs -> representative letter heights
        n, lab, stats, _ = cv2.connectedComponentsWithStats(line, 8)
        hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 200]
        if hs:
            xh += [h for h in hs if h < np.median(hs) * 1.15]
        ys = np.where(line.sum(1) > 0)[0]
        if len(ys):
            baselines.append(y0 + int(ys.max()))
        gaps_all += inner
    # the gap distribution is bimodal: gaps inside a word vs gaps between words
    gaps_small, gaps_words = [], []
    if len(gaps_all) >= 6:
        g = np.sort(np.array(gaps_all, float))
        # 1-D 2-means split
        best_t, best_sse = g[0], None
        for t in g[:-1]:
            low, high = g[g <= t], g[g > t]
            sse = ((low - low.mean()) ** 2).sum() + ((high - high.mean()) ** 2).sum()
            if best_sse is None or sse < best_sse:
                best_sse, best_t = sse, t
        gaps_small = [v for v in g if v <= best_t]
        gaps_words = [v for v in g if v > best_t]
    if len(baselines) > 1:
        pitches = list(np.diff(sorted(baselines)))
    return dict(
        x_height=float(np.median(xh)) if xh else None,
        line_pitch=float(np.median(pitches)) if pitches else None,
        letter_gap=float(np.median(gaps_small)) if gaps_small else None,
        word_gap=float(np.median(gaps_words)) if gaps_words else None,
    )


def slant_deg(mask):
    """Median slant of near-vertical strokes (positive = leaning right)."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=30,
                           minLineLength=25, maxLineGap=4)
    angs = []
    if segs is not None:
        for x1, y1, x2, y2 in segs[:, 0]:
            a = np.degrees(np.arctan2(x2 - x1, y1 - y2))   # 0 = vertical
            if abs(a) < 35:
                angs.append(a)
    return float(np.median(angs)) if angs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', default='../assets')
    ap.add_argument('--pdf'); ap.add_argument('--png')
    a = ap.parse_args()

    assets = a.assets
    glyphs = json.load(open(os.path.join(assets, 'glyphs.json')))
    meta = json.load(open(os.path.join(assets, 'lines.json')))
    ink = (np.array(Image.open(os.path.join(assets, 'ink_mask.png'))) > 0).astype(np.uint8)
    rgb = load_rgb(a.pdf, a.png)

    rows = [row_metrics(glyphs, r) for r in range(4)]
    x_h = float(np.median([r['x_height'] for r in rows if r['x_height']]))
    cap_h = float(np.median([r['cap_height'] for r in rows if r['cap_height']]))

    # per glyph vertical offsets relative to its own line's baseline
    para_x = float(np.median([m['h'] for ch, l in glyphs.items() for m in l
                              if m['row'] == 9 and ch in NO_ASCENDER] or [x_h]))
    for ch, lst in glyphs.items():
        for m in lst:
            if m['row'] < 4:
                base = rows[m['row']]['baseline']
                m['x_height'] = rows[m['row']]['x_height']
            else:                               # harvested from the paragraph
                base = m.get('para_baseline') or m['baseline']
                m['x_height'] = para_x
            m['top_rel'] = round((m['baseline'] - m['h']) - base, 1)
            m['bot_rel'] = round(m['baseline'] - base, 1)

    # pen colour and thickness, from the ink pixels themselves
    pix = rgb[ink.astype(bool)]
    pen = np.median(pix, 0).astype(int).tolist() if len(pix) else [20, 20, 60]
    thick = 2 * cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    t = float(np.median(thick[ink.astype(bool)]))

    bands = [(int(b['y0']), int(b['y1'])) for b in meta['bands']]
    para = paragraph_metrics(ink, bands)
    para['slant_deg'] = slant_deg(ink)

    scale = (para['x_height'] / x_h) if para.get('x_height') else 1.0
    out = dict(
        pen=dict(color=pen, thickness=t),
        letters=dict(x_height=x_h, cap_height=cap_h,
                     ascender=float(np.median([abs(m['top_rel']) for ch, l in glyphs.items()
                                               if ch in 'bdfhklt' for m in l])),
                     descender=float(np.median([m['bot_rel'] for ch, l in glyphs.items()
                                                if ch in 'gjpqy' for m in l]))),
        rows=rows,
        paragraph=para,
        pad=4,
        scale=round(scale, 3),
        units=dict(line_pitch_over_x=round(para['line_pitch'] / para['x_height'], 3)
                   if para.get('line_pitch') and para.get('x_height') else 2.9),
    )
    json.dump(out, open(os.path.join(assets, 'metrics.json'), 'w'), indent=1)
    json.dump(glyphs, open(os.path.join(assets, 'glyphs.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
