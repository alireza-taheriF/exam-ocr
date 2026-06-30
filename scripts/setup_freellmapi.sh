#!/usr/bin/env bash
# Bootstrap FreeLLMAPI for exam-ocr (no paid API keys — free tiers only).
# Requires: Docker, curl, python3, freellmapi running on PORT (default 3002).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FREELLMAPI_DIR="${FREELLMAPI_DIR:-$ROOT/freellmapi}"
PORT="${PORT:-3002}"
BASE="http://127.0.0.1:${PORT}"
EMAIL="${FREELLMAPI_EMAIL:-exam-ocr@local.dev}"
PASSWORD="${FREELLMAPI_PASSWORD:-exam-ocr-local}"

if [[ ! -d "$FREELLMAPI_DIR" ]]; then
  echo "Clone FreeLLMAPI first:"
  echo "  git clone https://github.com/tashfeenahmed/freellmapi.git $FREELLMAPI_DIR"
  exit 1
fi

if ! curl -sf "$BASE/api/ping" >/dev/null 2>&1; then
  echo "Starting FreeLLMAPI on port $PORT..."
  (cd "$FREELLMAPI_DIR" && printf 'PORT=%s\n' "$PORT" >> .env 2>/dev/null || true)
  grep -q "^PORT=$PORT" "$FREELLMAPI_DIR/.env" 2>/dev/null || echo "PORT=$PORT" >> "$FREELLMAPI_DIR/.env"
  (cd "$FREELLMAPI_DIR" && docker compose up -d)
  for _ in $(seq 1 30); do
    curl -sf "$BASE/api/ping" >/dev/null 2>&1 && break
    sleep 1
  done
fi

status="$(curl -sf "$BASE/api/auth/status")"
needs_setup="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['needsSetup'])" "$status")"

if [[ "$needs_setup" == "True" ]]; then
  echo "Creating dashboard account ($EMAIL)..."
  TOKEN="$(curl -sf -X POST "$BASE/api/auth/setup" \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")"
else
  echo "Logging in to dashboard..."
  TOKEN="$(curl -sf -X POST "$BASE/api/auth/login" \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")"
fi

auth_hdr=(-H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json')

add_keyless() {
  local platform="$1"
  curl -sf -X POST "$BASE/api/keys" "${auth_hdr[@]}" -d "{\"platform\":\"$platform\"}" >/dev/null \
    && echo "  + $platform (no key needed)" \
    || echo "  ~ $platform (already configured or skipped)"
}

echo "Adding anonymous free providers..."
for p in ovh kilo pollinations; do
  add_keyless "$p"
done

# Optional free provider keys from exam-ocr .env (all free signups, no payment).
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ROOT/.env"; set +a
  add_provider_key() {
    local platform="$1" key="$2" label="$3"
    [[ -z "$key" || "$key" == "your_api_key_here" || "$key" == "changeme" ]] && return 0
    curl -sf -X POST "$BASE/api/keys" "${auth_hdr[@]}" \
      -d "{\"platform\":\"$platform\",\"key\":\"$key\",\"label\":\"$label\"}" >/dev/null \
      && echo "  + $platform ($label)" \
      || echo "  ~ $platform ($label) skipped"
  }
  if [[ -n "${OPENROUTER_API_KEY:-}" && "$OPENROUTER_API_KEY" != "your_api_key_here" ]]; then
    add_provider_key openrouter "${OPENROUTER_API_KEY}" "exam-ocr-openrouter"
    # Broken Google keys (location/quota) sit first in fallback and waste minutes.
    curl -sf -X PATCH "$BASE/api/keys/platform/google" "${auth_hdr[@]}" \
      -d '{"enabled":false}' >/dev/null \
      && echo "  - google disabled (use OpenRouter vision instead)"
  else
    add_provider_key google "${GEMINI_API_KEY:-}" "exam-ocr-gemini"
  fi
  add_provider_key groq     "${GROQ_API_KEY:-}"        "exam-ocr-groq"
  add_provider_key github   "${GITHUB_TOKEN:-}"        "exam-ocr-github"
fi

UNIFIED="$(curl -sf "$BASE/api/settings/api-key" "${auth_hdr[@]}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['apiKey'])")"

echo ""
echo "Unified API key: $UNIFIED"
echo "Dashboard:     $BASE"
echo ""

# Patch exam-ocr .env
ENV_FILE="$ROOT/.env"
touch "$ENV_FILE"
python3 - "$ENV_FILE" "$UNIFIED" "$PORT" <<'PY'
import pathlib, re, sys
path, key, port = sys.argv[1:4]
text = pathlib.Path(path).read_text(encoding="utf-8")
updates = {
    "VISION_API_BASE": f"http://localhost:{port}/v1",
    "VISION_API_KEY": key,
    "VISION_MODEL": "nvidia/nemotron-nano-12b-v2-vl:free",
    "VLM_API_BASE": f"http://localhost:{port}/v1",
    "VLM_API_KEY": key,
    "VLM_MODEL": "nvidia/nemotron-nano-12b-v2-vl:free",
    "LABEL_PROVIDER": "openai",
    "LABEL_DIRECT_OPENROUTER": "true",
}
for k, v in updates.items():
    pat = re.compile(rf"^{re.escape(k)}=.*$", re.M)
    line = f"{k}={v}"
    text = pat.sub(line, text) if pat.search(text) else text.rstrip() + "\n" + line + "\n"
pathlib.Path(path).write_text(text, encoding="utf-8")
print(f"Updated {path}")
PY

cat <<'NOTE'

برای لیبل‌گذاری تصویر (vision) حداقل یک کلید رایگان لازم است:

  ۱) OpenRouter (پیشنهادی — معمولاً از ایران هم کار می‌کند)
     https://openrouter.ai/settings/keys
     ثبت‌نام رایگان → Create Key → در .env:
       OPENROUTER_API_KEY=sk-or-v1-...

  ۲) GitHub Models (اگر اکانت GitHub دارید)
     https://github.com/settings/tokens → Fine-grained token با Models
       GITHUB_TOKEN=ghp_...

  ۳) Groq — اگر console.groq.com باز نشد یا {"error":"Not Found"} داد،
     احتمالاً از منطقه شما بلاک است؛ از OpenRouter استفاده کنید.

سپس:  bash scripts/setup_freellmapi.sh
لیبل:  python auto_labeler.py --provider openai --limit 5

NOTE
