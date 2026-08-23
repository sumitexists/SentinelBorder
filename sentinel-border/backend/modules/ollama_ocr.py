"""Local vision-based structured document extraction through Ollama."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request

from config import OLLAMA_HOST, OLLAMA_MODEL
from modules.gemini_ocr import _PROMPT, _normalise_response
from utils.helpers import get_logger

log = get_logger("ollama_ocr")


def run_ollama_ocr(
    raw_bytes: bytes,
    content_type: str = "image/jpeg",
    viz_text: str = "",
) -> tuple[dict, str]:
    """Ask a local vision-capable Ollama model to extract document fields."""
    supporting_text = (
        "\n\nSupporting OCR transcript (it can contain mistakes; inspect the image "
        "and treat it as the source of truth):\n" + viz_text[:4000]
        if viz_text else ""
    )
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [{
            "role": "user",
            "content": _PROMPT + supporting_text,
            "images": [base64.b64encode(raw_bytes).decode("ascii")],
        }],
    }
    request = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        log.info("Calling local Ollama model='%s'", OLLAMA_MODEL)
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw_text = str(result.get("message", {}).get("content", "")).strip()
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text).strip()
        if not raw_text:
            return {}, "Ollama returned an empty extraction response."
        return _normalise_response(json.loads(raw_text)), ""
    except json.JSONDecodeError:
        return {}, "Ollama returned invalid JSON."
    except urllib.error.URLError as exc:
        return {}, f"Ollama is unavailable at {OLLAMA_HOST}: {exc.reason}"
    except Exception as exc:
        log.warning("Ollama OCR failed: %s", exc)
        return {}, f"Ollama extraction failed: {exc}"
