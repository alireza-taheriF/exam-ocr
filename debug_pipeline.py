import cv2
import os
import sys

os.environ["DEBUG_LAYOUT"] = "1"

from pipeline.layout import detect_blocks

image_path = sys.argv[1]

original = cv2.imread(image_path)
print(f"Original image size: {original.shape}")

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

blocks = detect_blocks(original_rgb)
print(f"\nTotal blocks detected: {len(blocks)}")
for i, b in enumerate(blocks):
    print(f"  Block {i}: type={b['type']} confidence={b['confidence']:.2f} bbox={b['bbox']}")

debug_img = original.copy()
colors = {
    "text": (0, 255, 0),
    "list": (0, 200, 100),
    "table": (255, 0, 0),
    "figure": (0, 0, 255),
}
for b in blocks:
    x1, y1, x2, y2 = b["bbox"]
    color = colors.get(b["type"], (128, 128, 128))
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
    label = f"{b['type']} {b['confidence']:.1f}"
    cv2.putText(debug_img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite("debug_blocks.png", debug_img)
print("\nSaved debug_blocks.png — open this to see detected blocks visually")
