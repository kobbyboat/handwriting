#!/usr/bin/env python3
"""
Step 1 - Calibrate: extract glyph bitmaps from a handwriting sample sheet.

The sample page must contain (a) a handwritten alphabet block (A a  B b  ...)
near the top and (b) free paragraph writing below it.  The alphabet block is
what teaches us what every letter looks like; the paragraph is used later to
measure letter-spacing, x-height, slant and line rhythm.

Outputs (into --out):
    ink_mask.png        clean ink-only binarisation (ruled lines removed)
    lines.json          text-line bands + line size
    glyphs.json         per-character crop index (file, w, h, baseline, row)
    glyphs/<char>/*.png one grayscale crop per occurrence
    atlas.png           contact sheet, for eyeballing the calibration

Usage:
    python3 01_extract_glyphs.py --pdf ../../uploads/myhandwriting.pdf --out ../assets
    python3 01_extract_glyphs.py --png scan.png --out ../assets
"""
import argparse
import json
import os
import sys

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

# the sample sheet writes the alphabet as upper/lower pairs, 7-9 pairs per row
ALPHABET = [["Aa", "Bb", "Cc", "Dd", "Ee", "Ff", "Gg"],
            ["Hh", "Ii", "Jj", "Kk", "Ll", "Mm", "Nn"],
            ["Oo", "Pp", "Qq", "Rr", "Ss", "Tt", "Uu"],
            ["Vv", "Ww", "Xx", "Yy", "Zz"]]
ALPHABET_CHARS = ''.join(''.join(r) for r in ALPHABET)

# Transcript of the free-writing paragraph on the sample page, used to harvest
# characters that the alphabet block does not contain (digits, punctuation).
# One entry per detected text line, in page order, after the alphabet rows.
# Pass --transcript FILE (one line of text per paragraph line) to override, or
# --no-transcript to skip this step.
TRANSCRIPT = [
    "The intention is to create a 95%",
    "minimum replica of my handwriting -",
    "a reusable tool for writing lesson plans.",
    "Lesson objectives include the following:",
    "a. State the four steps used to change an",
    "active sentence into the passive.",
    "b. Convert simple active sentences into the",
    "passive while keeping the correct tense.",
]


# --------------------------------------------------------------------- page + ink

def load_page(path):
    """Return an HxWx3 uint8 RGB array of the page at native resolution."""
    if path.lower().endswith('.pdf'):
        import pymupdf
        doc = pymupdf.open(path)
        page = doc[0]
        best, bestpix = -1, None
        for img in page.get_images(full=True):          # prefer the embedded scan
            w, h = img[2], img[3]
            if w * h > best:
                best, bestpix = w * h, pymupdf.Pixmap(doc, img[0])
        pix = bestpix if best > 200000 else page.get_pixmap(dpi=400)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return arr[:, :, :3]
    return np.array(Image.open(path).convert('RGB'))


def remove_dashed_rules(binary, max_thick=6.5, min_len=40, row_frac=0.15):
    """Remove leftover fragments of dashed notebook rules.

    A dashed rule leaves short thin horizontal islands. A horizontal run is
    deleted when it is thin AND lies on a row whose total thin-horizontal ink
    spans a good part of the page width (i.e. that row IS a rule row, even if
    the rule is broken into dashes). Crossbars of letters sit on rows whose thin
    ink is short, so they survive.
    """
    h, w = binary.shape
    thick = 2 * cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thin = (thick <= max_thick)
    # thin-ink mass per row
    mass = (binary & thin).sum(1)
    rule_rows = mass > row_frac * w
    if not rule_rows.any():
        return binary, 0
    out = binary.copy()
    removed = 0
    for y in np.where(rule_rows)[0]:
        x = 0
        row = out[y]
        while x < w:
            if row[x] and thin[y, x]:
                x2 = x
                while x2 + 1 < w and row[x2 + 1]:
                    x2 += 1
                if x2 - x + 1 >= min_len:
                    removed += int(out[y, x:x2 + 1].sum())
                    out[y, x:x2 + 1] = 0
                x = x2 + 1
            else:
                x += 1
    return out, removed


def remove_rules(binary):
    """Delete printed notebook rules (horizontal) and margin rules (vertical).

    They are found with a probabilistic Hough transform: anything long and
    almost perfectly horizontal/vertical is a rule, never handwriting. Only
    thin parts of the mask are erased, so a letter that happens to sit on a
    rule survives.
    """
    H, W = binary.shape
    thick = 2 * cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thin = thick <= 6.5
    out = binary.copy()
    segs = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=120,
                           minLineLength=int(0.22 * W), maxLineGap=25)
    n_removed = 0
    if segs is not None:
        for x1, y1, x2, y2 in segs[:, 0]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy <= max(2, 0.01 * dx):                      # horizontal rule
                yy = int(round((y1 + y2) / 2))
                band = thin[max(0, yy - 2):yy + 3, min(x1, x2):max(x1, x2) + 1]
                m = np.zeros_like(out)
                m[max(0, yy - 2):yy + 3, min(x1, x2):max(x1, x2) + 1] = band
                n_removed += int((out & m).sum())
                out &= ~m.astype(np.uint8)
            elif dx <= max(2, 0.01 * dy) and dy > 0.18 * H:   # vertical margin rule
                xx = int(round((x1 + x2) / 2))
                band = thin[min(y1, y2):max(y1, y2) + 1, max(0, xx - 2):xx + 3]
                m = np.zeros_like(out)
                m[min(y1, y2):max(y1, y2) + 1, max(0, xx - 2):xx + 3] = band
                n_removed += int((out & m).sum())
                out &= ~m.astype(np.uint8)
    return out, n_removed


def ink_mask(rgb):
    """Binarise the pen ink and remove the printed rules from the page."""
    lum = rgb.astype(float).mean(2)
    m = (lum < 178).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m, _ = remove_rules(m)
    m, _ = remove_dashed_rules(m)
    ink = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    for i in range(1, n):                                # despkle + rule leftovers
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] < 40 or (h <= 9 and w >= 22):
            ink[lab == i] = 0
    # ignore the outer margins of the page (binder edge / scanner border)
    H, W = ink.shape
    ink[:, :int(0.055 * W)] = 0
    ink[:, int(0.97 * W):] = 0
    ink[:int(0.01 * H)] = 0
    return ink


# ------------------------------------------------------------------- line finding

def _pitch(prof, lo=60, hi=280):
    """Dominant line pitch of a row profile, via autocorrelation."""
    p = prof - prof.mean()
    if p.std() < 1e-6 or len(p) < hi * 2:
        return None
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    ac /= ac[0]
    best = None
    for i in range(lo, min(hi, len(ac) - 1)):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]:
            if best is None or ac[i] > ac[best]:
                best = i
    return best


def text_lines(ink, min_gap=8, min_mass=4000):
    """Split the page into horizontal text lines.

    1. dense row bands are cut at profile valleys (a spaced-out alphabet block
       always has clear whitespace between rows);
    2. a band much taller than the page's line pitch (dense paragraph writing
       where ascenders and descenders interlock) is split at every prominent
       interior valley, at the pitch found by autocorrelation;
    3. each line is grown to the midpoints of the gaps between neighbours.
    """
    from scipy.signal import find_peaks
    prof = np.convolve(ink.sum(1).astype(float), np.ones(5) / 5, 'same')

    bands, st = [], None
    for y, v in enumerate(prof):
        if v > min_gap and st is None:
            st = y
        elif v <= min_gap and st is not None:
            if y - st > 25 and ink[st:y].sum() > min_mass:
                bands.append([st, y])
            st = None
    if st is not None and ink[st:].sum() > min_mass:
        bands.append([st, len(prof)])

    def fallback_pitch(mx=300):
        hs = [b[1] - b[0] for b in bands if b[1] - b[0] < mx]
        return float(np.median(hs)) if hs else 150.0

    lines = []
    for b in bands:
        sub = ink[b[0]:b[1]]
        ys = np.where(sub.sum(1) > 0)[0]
        if not len(ys):
            continue
        b = [b[0] + int(ys.min()), b[0] + int(ys.max()) + 1]
        h = b[1] - b[0]
        pitch = _pitch(prof[b[0]:b[1]]) or fallback_pitch()
        k = max(1, int(round(h / pitch)))
        mins = []
        if k > 1:
            mins, _ = find_peaks(-prof[b[0]:b[1]], distance=int(0.55 * pitch),
                                 prominence=5)
            mins = [b[0] + int(m) for m in mins if 0.35 * pitch < m < h - 0.35 * pitch]
        if not mins:
            lines.append(b)
            continue
        edges = [b[0]]
        for i in range(1, k):
            target = b[0] + i * h / k
            edges.append(min(mins, key=lambda m: abs(m - target)))
        edges = sorted(set(edges + [b[1]]))
        for i in range(len(edges) - 1):
            if edges[i + 1] - edges[i] > 20:
                lines.append([edges[i], edges[i + 1]])

    pitch_use = fallback_pitch()
    out = []
    for i, b in enumerate(lines):
        a = (lines[i - 1][1] + b[0]) // 2 if i > 0 else b[0] - pitch_use // 3
        z = (b[1] + lines[i + 1][0]) // 2 if i < len(lines) - 1 else b[1] + pitch_use // 3
        out.append((max(0, int(a)), int(z)))
    return out


# ------------------------------------------------------------------ glyph cutting

def line_column_profile(ink_line):
    """Per-column ink mass of a text line, normalised to 0..1."""
    p = ink_line.sum(0).astype(float)
    mx = p.max()
    return p / mx if mx > 0 else p


def whitespace_runs(prof, tol=0.0):
    """Maximal runs of (nearly) empty columns: [(start, end)].

    `tol` lets a column count as empty even if a few speck pixels cross it,
    which keeps the gap-finding robust to scan noise.
    """
    runs, st = [], None
    for x, v in enumerate(prof):
        if v <= tol and st is None:
            st = x
        elif v > tol and st is not None:
            runs.append((st, x)); st = None
    if st is not None:
        runs.append((st, len(prof)))
    return runs


# rough relative letter widths for cursive/print handwriting; only used as a
# prior to keep the dynamic-programming alignment honest
WIDTH_PRIOR = {
    'i': .45, 'l': .45, 'j': .55, 'I': .45, 'f': .60, 't': .60, 'r': .65,
    'J': .75, 'L': .75, 'T': .80, 'E': .80, 'F': .80, 'z': .80, 's': .80,
    'c': .80, 'e': .85, 'a': .90, 'o': .90, 'b': .90, 'd': .90, 'g': .90,
    'h': .95, 'k': .95, 'n': .95, 'p': .90, 'q': .90, 'u': .95, 'v': .90,
    'x': .90, 'y': .90, 'A': 1.05, 'B': 1.00, 'C': 1.00, 'D': 1.05,
    'G': 1.00, 'H': 1.10, 'K': 1.00, 'N': 1.10, 'O': 1.05, 'P': .95,
    'Q': 1.05, 'R': 1.00, 'S': .95, 'U': 1.05, 'V': 1.05, 'X': 1.05,
    'Y': 1.05, 'Z': 1.00, 'M': 1.35, 'W': 1.35, 'm': 1.30, 'w': 1.25,
}


def row_components(ink_line, min_area=100):
    """Solid components of a line, left to right: [(x0, x1, y0, y1)].

    Small but real marks (the dot of an i, a detached serif) are kept; specks and
    leftover rule fragments (thin and wide) are dropped.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink_line, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, ar = stats[i]
        # specks, and rule fragments: long, low, extremely wide components
        if ar < min_area or (h <= 25 and w >= 50 and w >= 6 * h):
            continue
        out.append((int(x), int(x + w), int(y), int(y + h)))
    out.sort()
    return _merge_overlapping(out)


def _merge_overlapping(comps):
    """Merge pieces of one letter that the threshold broke apart.

    A thin crossbar ("A"), the top dash of a "T" or the dot of an "i" often
    comes out as a separate component sitting inside, or overlapping, the main
    stroke's x-range. Two different letters practically never overlap in x, so
    components whose x-spans overlap strongly are fused into one bounding box.
    """
    merged = True
    comps = list(comps)
    while merged:
        merged = False
        for i in range(len(comps) - 1):
            a, b = comps[i], comps[i + 1]
            ov = min(a[1], b[1]) - max(a[0], b[0])
            if ov > 0.55 * min(a[1] - a[0], b[1] - b[0]):
                comps[i] = (min(a[0], b[0]), max(a[1], b[1]),
                            min(a[2], b[2]), max(a[3], b[3]))
                del comps[i + 1]
                merged = True
                break
    return comps


def boundary_cost(prev_c, next_c, unit):
    """Penalty for cutting between two components: cheap across a wide gap."""
    gap = max(0, next_c[0] - prev_c[1])
    return 1.2 * math.exp(-gap / max(unit * 0.30, 1.0))


def assign_letters(comps, text, xs_lo, xs_hi):
    """Assign components to the letters of `text` by dynamic programming.

    A letter owns a contiguous run of components. Its cost is the squared
    relative error between the span it covers and its expected width (a
    hand-tuned prior), plus a penalty for a letter that ends up with no ink.
    Touching letters are therefore separated at their most plausible
    proportion, and an "i" dot that merged with a neighbour is kept.
    """
    n, m = len(text), len(comps)
    if m == 0 or n == 0:
        return []
    span = xs_hi - xs_lo
    pri = np.array([WIDTH_PRIOR.get(c, 1.0) for c in text], float)
    target = pri / pri.sum() * span
    INF = 1e18
    cost = np.full((n + 1, m + 1), INF)
    back = np.zeros((n + 1, m + 1), np.int32)
    cost[0, 0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            best, bi = INF, -1
            for i0 in range(0, i):
                if cost[j - 1, i0] >= INF:
                    continue
                w = comps[i - 1][1] - comps[i0][0]
                c = ((w - target[j - 1]) / max(target[j - 1], 1.0)) ** 2
                if i0 > 0:
                    c += 0.5 * boundary_cost(comps[i0 - 1], comps[i0], span / n)
                if cost[j - 1, i0] + c < best:
                    best, bi = cost[j - 1, i0] + c, i0
            # a letter with no ink at all (an empty cell) is allowed but costs
            if cost[j - 1, i] + 6.0 < best:
                best, bi = cost[j - 1, i] + 6.0, i
            cost[j, i] = best
            back[j, i] = bi
    groups, j, i = [], n, m
    while j > 0:
        i0 = int(back[j, i])
        if i0 < i:
            groups.append((j - 1, comps[i0][0], comps[i - 1][1]))
        else:
            groups.append((j - 1, None, None))
        j, i = j - 1, i0
    groups.reverse()
    return [(text[j], a, b) for j, a, b in groups]


def segment_line(ink_line, text, n_pairs):
    """Return [(char, x0, x1)] for an alphabet row of known pairs.

    1. keep the row's real components (rule fragments and specks dropped);
    2. find the gaps between pairs - they are several times wider than the gap
       between the two letters of a pair (and the DP below tolerates rows where
       the writer left a pair out);
    3. assign each pair's components to its letters with a DP.
    """
    comps = row_components(ink_line)
    if not comps or n_pairs < 1:
        return []
    hs = np.median([c[3] - c[2] for c in comps])
    core = [c for c in comps if (c[3] - c[2]) >= 0.35 * hs]
    if core:
        lo_c, hi_c = core[0][0] - 8, core[-1][1] + 8
        comps = [c for c in comps if lo_c <= (c[0] + c[1]) / 2 <= hi_c]
    if not comps:
        return []
    xs_lo, xs_hi = comps[0][0], comps[-1][1]

    gaps = [(comps[i + 1][0] - comps[i][1], i) for i in range(len(comps) - 1)]
    med_w = np.median([c[1] - c[0] for c in comps])
    med_gap = np.median([g for g, _ in gaps]) if gaps else 0
    thr = max(0.85 * med_w, 1.6 * med_gap)
    cuts = [i for g, i in gaps if g >= thr]
    if len(cuts) > n_pairs - 1:                      # too many: keep the widest
        cuts = sorted(i for g, i in sorted(gaps, reverse=True)[:n_pairs - 1])
    step = len(text) // n_pairs
    groups = [text[g * step:(g + 1) * step] for g in range(len(cuts) + 1)]
    segs = assign_groups(comps, groups, xs_lo, xs_hi, cuts)
    out = []
    for (i0, i1), grp in zip(segs, groups):
        gc = comps[i0:i1]
        if not gc:
            continue
        out += assign_letters(gc, grp, gc[0][0], gc[-1][1])
    return out


def assign_groups(comps, groups, xs_lo, xs_hi, cuts=None):
    """Split the components into len(groups) contiguous pair groups.

    When the pair boundaries were already detected (`cuts`) they are used as
    hard cuts, otherwise a DP places them by width prior.
    """
    if cuts:
        edges = [0] + [c + 1 for c in cuts] + [len(comps)]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    n, m = len(groups), len(comps)
    span = xs_hi - xs_lo
    pri = np.array([sum(WIDTH_PRIOR.get(c, 1.0) for c in g) for g in groups], float)
    target = pri / pri.sum() * span
    INF = 1e18
    cost = np.full((n + 1, m + 1), INF)
    back = np.zeros((n + 1, m + 1), np.int32)
    cost[0, 0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            best, bi = INF, -1
            for i0 in range(j - 1, i):
                if cost[j - 1, i0] >= INF:
                    continue
                w = comps[i - 1][1] - comps[i0][0]
                c = ((w - target[j - 1]) / max(target[j - 1], 1.0)) ** 2
                if i0 > 0:                       # cost of the cut before this group
                    c += boundary_cost(comps[i0 - 1], comps[i0], span / len(groups))
                if cost[j - 1, i0] + c < best:
                    best, bi = cost[j - 1, i0] + c, i0
            cost[j, i] = best
            back[j, i] = bi
    segs, j, i = [], n, m
    while j > 0:
        i0 = int(back[j, i])
        segs.append((i0, i))
        j, i = j - 1, i0
    return list(reversed(segs))


def inkbox(ink_line, a, b, y0=0):
    """Tight ink bounding box of columns [a, b)."""
    sub = ink_line[:, a:b]
    xs = np.where(sub.sum(0) > 0)[0]
    if not len(xs):
        return None
    ys = np.where(sub.sum(1) > 0)[0]
    return dict(x=int(a + xs.min()), x2=int(a + xs.max()) + 1,
                y=int(ys.min()) + y0, y2=int(ys.max()) + 1 + y0)


def cut_glyphs(ink, band, pairs, overrides=None):
    """Return [(char, box)] for one alphabet row, given its letter pairs."""
    y0, y1 = band
    line = ink[y0:y1]
    text = ''.join(pairs)
    comps = row_components(line)
    out = []
    overrides = overrides or {}
    for ch, a, b in segment_line(line, text, len(pairs)):
        if ch in overrides:                     # manual fix from calibration
            a, b = overrides[ch]
        if a is None:
            continue
        # keep only the components that mostly live inside this letter's slice,
        # so a wide DP slice cannot drag in a neighbour's strokes
        sel = [(cx0, cx1, cy0, cy1) for (cx0, cx1, cy0, cy1) in comps
               if (min(cx1, b) - max(cx0, a)) > 0.5 * (cx1 - cx0)]
        if not sel:
            bx = inkbox(line, a, b, y0)
            if bx:
                out.append((ch, bx))
            continue
        bx = dict(x=min(c[0] for c in sel), x2=max(c[1] for c in sel),
                  y=min(c[2] for c in sel) + y0, y2=max(c[3] for c in sel) + y0)
        out.append((ch, bx))
    return _unshare_components(out, line, y0)


def _unshare_components(boxes, line, y0):
    """Separate letters that were assigned the same ink (fully merged strokes).

    When two letters of a pair are written as one continuous flourish the DP
    hands both the same component; split their common span at the proportion
    predicted by the width prior, then trim each half to its own ink.
    """
    boxes = [(ch, dict(bx)) for ch, bx in boxes]
    for i in range(1, len(boxes)):
        pch, pbx = boxes[i - 1]
        ch, bx = boxes[i]
        overlap = pbx['x2'] - bx['x']
        if overlap > 0.35 * min(pbx['x2'] - pbx['x'], bx['x2'] - bx['x']):
            a, b = min(pbx['x'], bx['x']), max(pbx['x2'], bx['x2'])
            w1 = WIDTH_PRIOR.get(pch, 1.0)
            w2 = WIDTH_PRIOR.get(ch, 1.0)
            mid = int(a + (b - a) * w1 / (w1 + w2))
            n1 = inkbox(line, a, mid, y0) or dict(pbx, x2=mid)
            n2 = inkbox(line, mid, b, y0) or dict(bx, x=mid)
            boxes[i - 1] = (pch, n1)
            boxes[i] = (ch, n2)
    return boxes


def line_baseline(comps, x_height):
    """Estimate a line's baseline from the bottoms of x-height components."""
    hs = [c[3] - c[2] for c in comps]
    if not hs:
        return None
    xh = np.median(hs)
    bots = [c[3] for c in comps if 0.55 * xh <= (c[3] - c[2]) <= 1.2 * xh]
    return float(np.median(bots)) if bots else float(np.median([c[3] for c in comps]))


def segment_words(comps, x_height):
    """Split a line's components into words at its widest gaps."""
    if len(comps) < 2:
        return [comps]
    gaps = [comps[i + 1][0] - comps[i][1] for i in range(len(comps) - 1)]
    med = np.median(gaps)
    thr = max(0.55 * x_height, 2.2 * med)
    words, cur = [], [comps[0]]
    for g, c in zip(gaps, comps[1:]):
        if g > thr:
            words.append(cur); cur = [c]
        else:
            cur.append(c)
    words.append(cur)
    return words


def harvest_line(ink_line, text, y0, x_height, wanted):
    """Return [(char, box)] for the `wanted` characters of one transcript line."""
    comps = row_components(ink_line)
    if not comps:
        return []
    words = segment_words(comps, x_height)
    tw = [w for w in text.split() if w]
    pairs = list(zip(tw, words)) if len(tw) == len(words) else [(text, comps)]
    out = []
    for wtxt, wcomps in pairs:
        for ch, a, b in assign_letters(wcomps, wtxt, wcomps[0][0], wcomps[-1][1]):
            if a is None or ch not in wanted:
                continue
            # punctuation frequently has no component of its own (it merges
            # with a neighbour or is written on the same stroke). Keep the cell
            # only when it actually holds a mark, and never the ink of a letter.
            cand = inkbox(ink_line, a, b, y0)
            if not cand:
                continue
            if (cand['x2'] - cand['x']) > 2.2 * (cand['y2'] - cand['y']) and \
                    ch in '.,:;':
                continue
            out.append((ch, cand))
    # a punctuation mark must not steal a letter's/number's ink: when two
    # harvested boxes overlap, keep the alphanumeric one
    out.sort(key=lambda t: (t[0].isalpha() or t[0].isdigit()), reverse=True)
    kept = []
    for ch, bx in out:
        clash = any(min(bx['x2'], k['x2']) - max(bx['x'], k['x']) > 0.5 *
                    min(bx['x2'] - bx['x'], k['x2'] - k['x']) for _, k in kept)
        if not clash:
            kept.append((ch, bx))
    return kept


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf'); ap.add_argument('--png')
    ap.add_argument('--out', default='../assets')
    ap.add_argument('--pad', type=int, default=4)
    ap.add_argument('--overrides', default=None,
                    help='JSON {char: [row, x0, x1]} fixing a bad automatic cut')
    ap.add_argument('--transcript', default=None,
                    help='text file, one line per paragraph line (or "none")')
    a = ap.parse_args()

    rgb = load_page(a.pdf or a.png)
    ink = ink_mask(rgb)
    lines = text_lines(ink)
    os.makedirs(a.out, exist_ok=True)
    gdir = os.path.join(a.out, 'glyphs')
    if os.path.isdir(gdir):                  # keep re-runs reproducible
        import shutil
        shutil.rmtree(gdir)
    Image.fromarray((ink * 255)).save(os.path.join(a.out, 'ink_mask.png'))

    ov_path = a.overrides or os.path.join(a.out, 'overrides.json')
    overrides = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
    size = float(np.median([z - y for y, z in lines[:4]]))
    meta, seen = {}, {}
    for row, (band, pairs) in enumerate(zip(lines[:4], ALPHABET)):
        ov = {ch: v[1:] for ch, v in overrides.items() if len(v) == 3 and v[0] == row}
        for ch, box in cut_glyphs(ink, band, pairs, ov):
            seen[ch] = seen.get(ch, 0) + 1
            x0, y0, x1, y1 = box['x'], box['y'], box['x2'], box['y2']
            p = a.pad
            crop = ink[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p] * 255
            d = os.path.join(gdir, f'{ord(ch):03d}_{ch}')
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f'{seen[ch]:02d}_row{row}.png')
            Image.fromarray(crop.astype(np.uint8)).save(fn)
            meta.setdefault(ch, []).append(
                dict(file=os.path.relpath(fn, a.out), w=int(x1 - x0), h=int(y1 - y0),
                     baseline=int(y1), left=int(x0), row=row))

    # ---- harvest extra characters (digits, punctuation) from the paragraph
    # digits and light punctuation are harvested from the paragraph
    from_wanted = set('0123456789%.,:;?!')
    transcript = TRANSCRIPT
    if a.transcript == 'none':
        transcript = []
    elif a.transcript:
        transcript = [l.rstrip('\n') for l in open(a.transcript)]
    if transcript:
        import re as _re
        for band, line_text in zip(lines[4:], transcript):
            wanted = {c for c in line_text if c in from_wanted}
            if not wanted:
                continue
            y0, y1 = int(band[0]), int(band[1])
            got = harvest_line(ink[y0:y1], line_text, y0, size, wanted)
            base = line_baseline(row_components(ink[y0:y1]), size)
            for ch, box in got:
                seen[ch] = seen.get(ch, 0) + 1
                x0, y0b, x1, y1b = box['x'], box['y'], box['x2'], box['y2']
                p = a.pad
                crop = ink[max(0, y0b - p):y1b + p, max(0, x0 - p):x1 + p] * 255
                d = os.path.join(gdir, f'{ord(ch):03d}_{ch}')
                os.makedirs(d, exist_ok=True)
                fn = os.path.join(d, f'{seen[ch]:02d}_para.png')
                Image.fromarray(crop.astype(np.uint8)).save(fn)
                meta.setdefault(ch, []).append(
                    dict(file=os.path.relpath(fn, a.out), w=int(x1 - x0),
                         h=int(y1b - y0b), baseline=int(y1b), left=int(x0),
                         row=9, para_baseline=None if base is None else int(base + y0)))

    json.dump(meta, open(os.path.join(a.out, 'glyphs.json'), 'w'), indent=1)
    json.dump(dict(bands=[dict(y0=int(b[0]), y1=int(b[1])) for b in lines], size=size),
              open(os.path.join(a.out, 'lines.json'), 'w'), indent=1)

    # contact sheet for eyeballing the calibration
    cell, cols = 150, 13
    chars = sorted(meta, key=lambda c: (c.islower(), c))
    sheet = Image.new('RGB', (cols * cell, ((len(chars) + cols - 1) // cols) * (cell + 18)), 'white')
    dr = ImageDraw.Draw(sheet)
    for i, ch in enumerate(chars):
        cx, cy = (i % cols) * cell, (i // cols) * (cell + 18)
        dr.text((cx + 4, cy + 2), f'{ch} x{len(meta[ch])}', fill='black')
        for j, m in enumerate(sorted(meta[ch], key=lambda m: m['row'])):
            im = Image.open(os.path.join(a.out, m['file']))
            sc = min((cell - 10) / max(im.width, im.height), 1.7)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            sheet.paste(Image.merge('RGB', [im] * 3), (cx + 4 + j * (cell // 3), cy + 18 + cell - im.height))
    sheet.save(os.path.join(a.out, 'atlas.png'))

    missing = [c for c in ALPHABET_CHARS if c not in meta]
    print(f'{sum(seen.values())} glyphs, {len(seen)} distinct chars')
    if missing:
        print('WARNING missing glyphs:', ' '.join(missing))
    print(f'-> {a.out}/glyphs  (contact sheet: {a.out}/atlas.png)')


if __name__ == '__main__':
    main()
