from PIL import Image
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
files = ["boy.png", "boyexpression.png", "girl.png", "girlexpression.png"]

trimmed = {}
for f in files:
    img = Image.open(os.path.join(src_dir, f)).convert("RGBA")
    bbox = img.split()[-1].getbbox()
    if bbox is None:
        raise SystemExit(f"{f}: no opaque pixels")
    cropped = img.crop(bbox)
    trimmed[f] = cropped
    print(f, "raw size:", img.size, "bbox:", bbox, "trimmed size:", cropped.size)

target_h = max(t.height for t in trimmed.values())
target_w = max(t.width for t in trimmed.values())

canvas_w = int(target_w * 1.1)
canvas_h = int(target_h * 1.02)

for f, t in trimmed.items():
    scale = target_h / t.height
    new_w = int(t.width * scale)
    new_h = target_h
    resized = t.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x = (canvas_w - new_w) // 2
    y = canvas_h - new_h
    canvas.paste(resized, (x, y), resized)
    canvas.save(os.path.join(src_dir, f))
    print(f, "->", canvas.size)
