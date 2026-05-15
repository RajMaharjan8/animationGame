from PIL import Image
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(src_dir, "expressions.png")).convert("RGBA")

W, H = img.size
n = 5
frame_w = W // n

names = [
    "boy_neutral.png",
    "boy_hit_right.png",
    "boy_hit_left.png",
    "boy_hit_center.png",
    "boy_cry.png",
]


def clean_low_alpha(im, threshold=40):
    """Erase pixels below alpha threshold so dark halo doesn't show."""
    px = im.load()
    w, h = im.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a < threshold:
                px[x, y] = (0, 0, 0, 0)


def strong_alpha_bbox(im, threshold=60):
    alpha = im.split()[-1]
    binarized = alpha.point(lambda v: 255 if v >= threshold else 0)
    return binarized.getbbox()


# pass 1: clean and find each frame's bbox
crops = []
bboxes = []
for i in range(n):
    crop = img.crop((i * frame_w, 0, (i + 1) * frame_w, H))
    clean_low_alpha(crop, threshold=40)
    bb = strong_alpha_bbox(crop, threshold=60)
    crops.append(crop)
    bboxes.append(bb)

# unify bottom (feet) so the boy stands at the same baseline.
# unify horizontal so head x is roughly centered.
# we use the union top/bottom and the max width.
tops    = [b[1] for b in bboxes]
bots    = [b[3] for b in bboxes]
widths  = [b[2] - b[0] for b in bboxes]

unified_top = min(tops) - 8
unified_bot = max(bots) + 8
target_h = unified_bot - unified_top
target_w = max(widths) + 24   # padding around widest frame

print("Unified canvas:", target_w, "x", target_h)

for i, (crop, bb, name) in enumerate(zip(crops, bboxes, names)):
    l, t, r, b = bb
    cw = r - l

    # crop horizontally tight, vertically use unified top/bottom
    box = (l, max(0, unified_top), r, min(crop.size[1], unified_bot))
    sub = crop.crop(box)

    # paste centered horizontally onto target canvas, bottom-aligned
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    x_off = (target_w - sub.size[0]) // 2
    y_off = target_h - sub.size[1]
    canvas.paste(sub, (x_off, y_off), sub)
    canvas.save(os.path.join(src_dir, name))
    print(name, "->", canvas.size)
