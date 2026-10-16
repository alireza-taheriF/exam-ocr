#!/bin/bash
# usage: rec_sheet.sh SHEETDIR SHEETNUM TODO OUT   (labels JSON on stdin: {"1":"text"|null,...})
cd "$(dirname "$0")/.."
.venv/bin/python scripts/add_labels.py --sheet-json "$1/sheet_$(printf %03d $2).json" --todo "$3" --out "$4" --labeler "${LABELER:-claude-code-r2}"
