#!/usr/bin/env python3
"""Slice the Run-game sprite sheets into clean, transparent, registered frames.

  run.png   -> sprites/run_0..4.png      (black bg + labels removed)
  jump.png  -> sprites/jump_0..6.png     (black bg + title + labels removed)
  rocks.jpeg-> sprites/rock_0..N.png     (baked-in checkerboard removed)

Run + jump frames share ONE canvas size so the character never changes
scale between animations. The black background is keyed with a low
threshold and interior holes are re-filled, so dark hair stays solid.
"""
import os
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "sprites")
os.makedirs(OUT, exist_ok=True)

A_THRESH = 24          # alpha above this counts as content


# --------------------------------------------------------------- utilities
def flood_from_border(passable):
    """Boolean flood: cells of `passable` reachable from the image border."""
    h, w = passable.shape
    out = np.zeros((h, w), bool)
    dq = deque()
    border = ([(0, x) for x in range(w)] + [(h - 1, x) for x in range(w)] +
              [(y, 0) for y in range(h)] + [(y, w - 1) for y in range(h)])
    for y, x in border:
        if passable[y, x] and not out[y, x]:
            out[y, x] = True; dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and passable[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True; dq.append((ny, nx))
    return out


def dilate(m, it):
    """4-connected (diamond) binary dilation, `it` iterations."""
    for _ in range(it):
        d = m.copy()
        d[1:, :]  |= m[:-1, :]
        d[:-1, :] |= m[1:, :]
        d[:, 1:]  |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
    return m


def erode(m, it):
    return ~dilate(~m, it)


def remove_black_bg(path, thresh=14):
    """Flood-fill the connected pure-black background to transparent, then
    re-fill any enclosed holes so dark hair / outlines stay 100% opaque."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    flood = img.copy()
    marker = (255, 0, 255)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for x in range(0, w, 25):
        seeds += [(x, 0), (x, h - 1)]
    for y in range(0, h, 25):
        seeds += [(0, y), (w - 1, y)]
    for s in seeds:
        if flood.getpixel(s) != marker and sum(img.getpixel(s)) < thresh:
            ImageDraw.floodfill(flood, s, marker, thresh=thresh)
    bg = np.all(np.asarray(flood) == np.array(marker), axis=2)
    # holes = transparent pixels NOT connected to the border -> make opaque
    bg &= flood_from_border(bg)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([np.asarray(img), alpha]), "RGBA")


def runs_1d(flags, gap, min_len):
    """Group True indices into (start, end) runs, merging gaps <= `gap`."""
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return []
    segs, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - p > gap:
            segs.append((s, p)); s = v
        p = v
    segs.append((s, p))
    return [(a, b) for a, b in segs if b - a + 1 >= min_len]


def label_components(mask):
    """4-connected connected-component labelling (no scipy needed)."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    cur = 0
    for y0, x0 in zip(*np.where(mask)):
        if lab[y0, x0]:
            continue
        cur += 1
        dq = deque([(y0, x0)])
        lab[y0, x0] = cur
        while dq:
            y, x = dq.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    dq.append((ny, nx))
    return lab, cur


# ----------------------------------------------------- sprite-sheet slicing
def analyze_sheet(path, name, n_expected):
    """Locate every frame on a black-bg sheet. Returns (rgba, frame-boxes)."""
    rgba = remove_black_bg(path)
    mask = np.asarray(rgba)[:, :, 3] > A_THRESH
    h, w = mask.shape

    # the characters form the tallest horizontal band; any title/label band
    # is much shorter, so pick the tallest run of content rows.
    rbands = runs_1d(mask.any(axis=1), gap=14, min_len=10)
    ry0, ry1 = max(rbands, key=lambda b: b[1] - b[0])
    band = mask[ry0:ry1 + 1, :]

    col_runs = runs_1d(band.any(axis=0), gap=8, min_len=18)
    if len(col_runs) == n_expected:
        bounds = [0]
        for i in range(n_expected - 1):
            bounds.append((col_runs[i][1] + col_runs[i + 1][0]) // 2)
        bounds.append(w)
    else:                                       # fall back to an even split
        cols = np.where(band.any(axis=0))[0]
        step = (cols[-1] - cols[0] + 1) / n_expected
        bounds = [int(round(cols[0] + i * step)) for i in range(n_expected + 1)]
    print(f"[{name}] rows {ry0}-{ry1}  detected {len(col_runs)} cols -> {n_expected}")

    frames = []
    for i in range(n_expected):
        cl, cr = bounds[i], bounds[i + 1] - 1
        cell_c = (bounds[i] + bounds[i + 1]) / 2.0
        win = mask[ry0:ry1 + 1, cl:cr + 1]
        rsub = runs_1d(win.any(axis=1), gap=10, min_len=12)
        ca, cb = max(rsub, key=lambda b: b[1] - b[0])   # tallest = the body
        ra, rb = ry0 + ca, ry0 + cb
        cell = mask[ra:rb + 1, cl:cr + 1]
        cin = np.where(cell.any(axis=0))[0]
        rin = np.where(cell.any(axis=1))[0]
        frames.append(dict(
            rgba=rgba, cell_c=cell_c,
            bx0=cl + cin[0], bx1=cl + cin[-1],
            by0=ra + rin[0],  by1=ra + rin[-1],
        ))
    return frames


def render_frames(frames, name, W, H):
    out = []
    for f in frames:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        crop = f["rgba"].crop((f["bx0"], f["by0"], f["bx1"] + 1, f["by1"] + 1))
        dx = int(round(W / 2 + (f["bx0"] - f["cell_c"])))
        dy = H - crop.height - 3                  # feet pinned near bottom
        canvas.paste(crop, (dx, dy), crop)
        out.append(canvas)
    for i, im in enumerate(out):
        im.save(os.path.join(OUT, f"{name}_{i}.png"))
    montage(out, os.path.join(OUT, f"_montage_{name}.png"), bg=(255, 0, 255))
    print(f"[{name}] {len(out)} frames @ {W}x{H}")


# ------------------------------------------------------------ rock slicing
def extract_rocks():
    """Isolate rocks by connectivity: the checkerboard greys (~53 / ~131)
    overlap rock shadows, so flood the checker-coloured pixels from the
    border; whatever is left is a rock. Edges are eroded to kill JPEG halo."""
    img = Image.open(os.path.join(HERE, "rocks.jpeg")).convert("RGB")
    arr = np.asarray(img).astype(int)
    br = arr.mean(2)
    sat = arr.max(2) - arr.min(2)
    checker_col = (sat < 18) & (((br >= 38) & (br <= 68)) |
                                ((br >= 116) & (br <= 146)))
    closed = erode(dilate(checker_col, 3), 3)    # bridge JPEG-ringing gaps
    fg = ~flood_from_border(closed)

    lab, n = label_components(fg)
    rocks = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if len(ys) < 1500:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if x1 - x0 < 26 or y1 - y0 < 20:
            continue
        comp = erode(lab[y0:y1 + 1, x0:x1 + 1] == k, 2)   # trim halo ring
        if comp.sum() < 800:
            continue
        sub = np.asarray(img)[y0:y1 + 1, x0:x1 + 1]
        alpha = np.where(comp, 255, 0).astype(np.uint8)
        rocks.append((y0, x0, Image.fromarray(np.dstack([sub, alpha]), "RGBA")))

    rocks.sort(key=lambda t: (t[0] // 90, t[1]))      # top-to-bottom, l-to-r
    imgs = [r[2] for r in rocks]
    for i, im in enumerate(imgs):
        im.save(os.path.join(OUT, f"rock_{i}.png"))
    montage(imgs, os.path.join(OUT, "_montage_rocks.png"), cell=210, cols=6)
    print(f"[rocks] {len(imgs)} rocks, widths {sorted(im.width for im in imgs)}")


# ------------------------------------------------------------------ montage
def montage(imgs, path, cell=None, cols=None, bg=(110, 110, 120)):
    if not imgs:
        return
    cols = cols or len(imgs)
    cell = cell or max(max(i.width, i.height) for i in imgs) + 10
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), bg)
    for k, im in enumerate(imgs):
        scale = min(1.0, (cell - 12) / max(im.width, im.height))
        s = im.resize((max(1, int(im.width * scale)),
                       max(1, int(im.height * scale))), Image.LANCZOS)
        cx = (k % cols) * cell + (cell - s.width) // 2
        cy = (k // cols) * cell + (cell - s.height) // 2
        sheet.paste(s, (cx, cy), s)
    sheet.save(path)


if __name__ == "__main__":
    run_frames  = analyze_sheet(os.path.join(HERE, "run.png"),  "run",  5)
    jump_frames = analyze_sheet(os.path.join(HERE, "jump.png"), "jump", 7)

    # ONE shared canvas size so the character keeps a constant scale
    allf = run_frames + jump_frames
    half_w = max(max(f["cell_c"] - f["bx0"], f["bx1"] - f["cell_c"]) for f in allf)
    W = int(np.ceil(half_w * 2)) + 6
    H = max(f["by1"] - f["by0"] + 1 for f in allf) + 6

    render_frames(run_frames,  "run",  W, H)
    render_frames(jump_frames, "jump", W, H)
    extract_rocks()
    print(f"shared frame size: {W}x{H}")
    print("done -> run/sprites/")
