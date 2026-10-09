from PIL import Image

from labeling.image_ops import resize_for_vlm


def test_resize_wide_crop():
    img = Image.new("RGB", (4000, 80), color=(255, 255, 255))
    out = resize_for_vlm(img, max_long_side=1024)
    assert max(out.size) == 1024
    assert out.size[0] < 4000


def test_no_upscale():
    img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    out = resize_for_vlm(img, max_long_side=1024)
    assert out.size == (200, 50)
