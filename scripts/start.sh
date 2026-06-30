#!/usr/bin/env bash
# راه‌اندازی سامانه خواندن برگه امتحان
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if [[ ! -d ".venv" ]]; then
  echo "ساخت محیط مجازی Python..."
  python3.12 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import fastapi" 2>/dev/null; then
  echo "نصب وابستگی‌ها..."
  pip install -r requirements.txt
fi

if [[ ! -f ".env" ]]; then
  echo "فایل .env یافت نشد — از .env.example کپی می‌شود."
  cp .env.example .env
  echo ""
  echo "⚠️  کلید API vision را در .env تنظیم کنید (OPENROUTER_API_KEY یا bash scripts/setup_freellmapi.sh)"
  echo ""
fi

echo ""
echo "سامانه Exam OCR"
echo "  رابط کاربری:  http://${HOST}:${PORT}/"
echo "  مستندات API:  http://${HOST}:${PORT}/docs"
echo ""
echo "برای توقف: Ctrl+C"
echo ""

exec uvicorn main:app --host "$HOST" --port "$PORT" --reload
