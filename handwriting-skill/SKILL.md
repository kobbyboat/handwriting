# handwriting-replica

Turn one photographed page of handwriting into a **reusable writing engine**:
it reproduces the same hand, at its natural size and rhythm, for any text you
give it.

Built and calibrated on the sample page in `uploads/myhandwriting.pdf`
(a notebook page with the alphabet written as `Aa Bb Cc ...` above a paragraph
of free writing). The learned model lives in `assets/` and is already usable:

```bash
python3 scripts/handwrite.py write "Any text you like" -o out.png
```

---

## 1. When to use this skill

Use it when someone wants text *written* (not typeset) in a specific
handwriting: lesson plans, notes, letters, worksheets, "handwritten" copies of
a document, or a font-like asset made from their own hand.

## 2. How the skill works

The skill is a **calibration + rendering** pipeline, not a font generator with
made-up shapes. Both steps matter for fidelity:

| stage | script | what it produces |
|---|---|---|
| 1. extract | `scripts/01_extract_glyphs.py` | the writer's real letter bitmaps |
| 2. calibrate | `scripts/02_calibrate.py` | the writer's metrics (size, slant, gaps, rhythm, pen) |
| 3. write | `scripts/03_write.py` | rendered text in that hand |
| all-in-one | `scripts/handwrite.py` | `learn` / `write` / `inspect` CLI |

### Stage 1 - extract (what the script actually does)

1. **Page load** - if the input is a PDF, the *largest embedded raster* is used
   (the real scan) instead of a re-render, so stroke detail is kept.
2. **Ink binarisation** - luminance threshold, then the printed notebook is
   removed: long thin horizontal rules are found with a Hough transform, dashed
   rule fragments are removed row-wise (a row whose thin horizontal ink spans
   much of the page is a rule row), vertical margin rules are detected as tall
   thin components. The writer's strokes survive because they are either thicker
   or short (an "A" crossbar is ~60 px, a rule is ~1000 px).
3. **Text lines** - dense row bands are cut at profile valleys; any band much
   taller than the page's line pitch (measured by autocorrelation, so dense
   paragraph writing where ascenders and descenders interlock still splits
   correctly) is subdivided at its deepest interior valleys.
4. **Letters** - alphabet rows are segmented with two DPs:
   * pair boundaries (`Aa|Bb|...`) go at gaps several times wider than the
     gap inside a pair;
   * within a pair, components are assigned to the two letters by a DP whose
     cost is the squared relative error between the span a letter covers and a
     hand-tuned width prior - so letters that touch ("Gg" written as one
     flourish) are split at their most plausible proportion, and an "i" dot
     that merged with a neighbour stays with its letter;
   * components that overlap in x (a crossbar, a detached dash) are fused, and
     letters that ended up sharing ink are separated afterwards.
5. **Output** - one tight crop per letter in `assets/glyphs/<code>_<char>/`,
   plus `glyphs.json` (crop file, width, height, baseline, left edge, row) and
   `atlas.png`, a contact sheet to eyeball.

### Stage 2 - calibrate

Measured from the sample and stored in `assets/metrics.json`:

* `pen` - median ink RGB and median stroke thickness (this page: near-black
  navy `#0d0c2c`, ~4 px stroke);
* `letters` - x-height 53 px, cap-height 98 px, ascender 92 px, descender 54 px
  (measured on the spaced-out alphabet, i.e. the writer's *demonstration* size);
* `glyphs` - per-letter top/bottom offset relative to its row's baseline;
* `paragraph` - from the free writing: x-height 39 px, line pitch 119 px,
  letter gap 12 px, word gap 65 px, slant +5.8°;
* `scale` 0.73 - how much smaller the writer writes normally than when writing
  a demonstration alphabet. **This is why output looks like the page and not
  like the oversized alphabet.**

### Stage 3 - write

Letters are placed on a baseline grid at the calibrated size and rhythm, then
made human: per-letter size (±3%), rotation (±1.2°), vertical offset, random
letter-spacing (0.75-1.35×), per-line baseline drift and a slow within-line
wave, plus a global slant shear, sub-pixel antialiasing and a fine grain in the
ink so it reads as pen rather than as a pasted bitmap.

---

## 3. Usage

```bash
cd scripts

# teach it a (new) handwriting sample
python3 handwrite.py learn --pdf ../../uploads/myhandwriting.pdf

# write anything
python3 handwrite.py write "The intention is to create a 95% minimum replica." -o out.png
python3 handwrite.py write --file lesson.txt -o lesson.pdf --paper ruled

# options worth knowing
#   --paper white|ruled      ruled adds a faint printed rule grid on the pitch
#   --x-height N             size in px (default: the writer's natural size)
#   --max-width N            wrap width in px
#   --jitter 0..2            rigidity vs. natural variation (0 = mechanical)
#   --slant DEG              override the measured slant
#   --color '#14142c'        override the measured ink colour
#   --noise N                add scan-like grain (0-12)
#   --seed N                 reproducible randomness
```

The same commands are available directly on the individual scripts
(`01_extract_glyphs.py`, `02_calibrate.py`, `03_write.py`).

## 4. Package layout

```
handwriting-skill/
├── SKILL.md
├── scripts/
│   ├── handwrite.py            # learn / write / inspect
│   ├── 01_extract_glyphs.py    # sample page -> letter crops
│   ├── 02_calibrate.py         # crops + page -> metrics.json
│   └── 03_write.py             # text -> rendered handwriting
├── assets/                     # the learned model (this person's hand)
│   ├── glyphs/<code>_<char>/   # 52 letter bitmaps
│   ├── glyphs.json             # crop index + baselines
│   ├── metrics.json            # pen, sizes, rhythm, slant
│   ├── ink_mask.png            # clean binarisation of the sample
│   ├── lines.json              # text-line bands of the sample
│   └── atlas.png               # contact sheet of every letter
└── examples/                   # sample outputs
```

## 5. Quality notes and limits

* Everything on the page is the writer's own ink; nothing is a look-alike font.
* The hand is *upright print with joins only inside a letter* - so no cursive
  joining is synthesised between letters. Output is best read as the writer
  writing carefully on a line.
* Only the glyphs present on the sample sheet exist. If a page is written with
  a different alphabet, add letters by re-running `learn` on a sheet that
  includes them (or extend the `ALPHABET` constant in `01_extract_glyphs.py`).
* Character variants: one sample per letter is stored per row (upper and lower
  case). Ask for more natural variation by varying `--seed`; a given seed
  always produces the same page.
* Not suitable for creating a font file directly - it renders images/PDFs. If a
  real font file is needed, the crops in `assets/glyphs/` can be traced and
  compiled with `fontTools`/`potrace`, but that removes the paper-like rendering
  and the variation model.

## 6. Verifying fidelity

Open `assets/atlas.png` to check that every letter was cut correctly, and
compare an output page against the sample paragraph:

```bash
python3 handwrite.py write --file ../../text.txt -o check.pdf --paper ruled
```

The letterforms, size, spacing, slant and line rhythm should read as the same
hand; if a letter looks wrong, its crop is the thing to fix (see
`--overrides` in `01_extract_glyphs.py`, a JSON map `{char: [row, x0, x1]}` that
forces a manual cut).
