import cv2
import numpy as np

from pipeline.segment import _absorb_fragments, _drop_binding_holes, _opencv_line_boxes


def _ruled_page(lines=6):
    page = np.full((1400, 1000, 3), 255, np.uint8)
    for y in range(150, 1300, 80):  # faint ruling every 80 px
        cv2.line(page, (40, y), (960, y), (190, 190, 230), 2)
    for i in range(lines):
        y = 140 + 160 * i
        cv2.putText(page, "handwriting line text", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 160), 3)
    return page


def test_ruled_page_gives_one_row_per_written_line():
    boxes = _opencv_line_boxes(_ruled_page(6))
    rows = {round((b[1] + b[3]) / 2 / 40) for b in boxes}
    assert len(rows) == 6  # ruling neither adds rows nor fuses written lines
    assert all((b[2] - b[0]) < 900 for b in boxes)  # no full-width ruling remnants


def test_binding_holes_column_is_dropped():
    width = 1000
    holes = [[960, y, 985, y + 25] for y in range(100, 700, 100)]
    text = [[200, 100, 700, 140]]
    assert _drop_binding_holes(holes + text, width) == text


def test_fragment_is_absorbed_into_line_below():
    line = [100, 200, 600, 260]
    stroke = [300, 170, 330, 205]  # ascender cut off above the line
    assert _absorb_fragments([line, stroke], line_h=60) == [[100, 170, 600, 260]]
