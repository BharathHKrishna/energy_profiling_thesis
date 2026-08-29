"""
KIT KI-Toolbox Caption Generator

The pipeline's only caption backend (factual 3-5 sentence energy-focused
description per coordinate). Built 2026-08-24 after finding Groq's
account-level daily token cap (200,000 tokens/day) made captioning all
10,000 coordinates take ~38 days regardless of in-process pacing -- Groq
was fully removed 2026-08-27, this is what replaced it. KIT's service is a
KIT-account-holder benefit (central token procurement, "usual extent" of
research use, no documented hard daily cap), reached via an
OpenAI-compatible endpoint -- see
https://www.zml.kit.edu/downloads/KI.Toolbox_API-AS-v2_ENG.pdf.

Uses build_prompt() from caption_prompt.py (no API client, no key, no
network call in that file -- pure string template).

Setup (the user does this part, not Claude -- API key never touches this repo
in plaintext beyond the .env file, which is already gitignored):
    1. Log in to https://ki-toolbox.scc.kit.edu/ with your KIT account.
    2. Settings (gear icon) -> Account -> API keys -> Create API Key.
    3. Add to /srv/THESIS/energy_profiling_thesis/.env:
           KIT_API_KEY=<paste the key here>
    4. (Optional) override the model, default is kit.gpt-oss-120b:
           KIT_MODEL=kit.gpt-oss-120b

Data-sensitivity note (from KIT's own 3-level classification): this pipeline's
per-coordinate feature data is real research data, not public web content --
use a `kit.*` (on-prem, data stays inside KIT) model, not an `azure.*` one.

Fair-use note: KIT's docs explicitly ask that "larger, automated projects --
e.g. processing complete research datasets with many thousands of API calls"
be coordinated in advance via ki-toolbox@scc.kit.edu. A 10,000-coordinate run
is exactly that case -- flagged to the user, not something this script decides
on its own.
"""
import os, sys, time
from dotenv import load_dotenv

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")
load_dotenv("/srv/THESIS/energy_profiling_thesis/.env")

from openai import OpenAI
from scripts.utils.logger import get_logger
from scripts.captions.caption_prompt import build_prompt  # shared, single source of truth

logger = get_logger("kit_caption")

KIT_BASE_URL = "https://ki-toolbox.scc.kit.edu/api/v1"
DEFAULT_MODEL = "kit.gpt-oss-120b"


def generate_caption(features: dict, bbox_size_m: int = 512) -> str:
    """Returns the caption string, raises on failure (never a silent
    '[Error...]' placeholder that could pass as a real caption downstream)."""
    api_key = os.environ.get("KIT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "KIT_API_KEY not set -- add it to .env (see kit_caption.py docstring "
            "for how to generate one at https://ki-toolbox.scc.kit.edu/)")

    model = os.environ.get("KIT_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key, base_url=KIT_BASE_URL)
    prompt = build_prompt(features, bbox_size_m)

    last_err = None
    n_attempts = 5
    for attempt in range(n_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant at KIT."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1200,
                # gpt-oss-120b spends real completion-token budget on an internal
                # reasoning pass before writing the visible answer (confirmed live
                # 2026-08-24: max_tokens=500 -> finish_reason='length' at 500/500
                # tokens, cut off mid-sentence; 900 -> completes but only ~160
                # tokens of margin; 1200 -> completes with real headroom, ~740
                # tokens actually used). No reasoning_effort param here (that's a
                # Groq-specific extension, not standard OpenAI) -- a bigger budget
                # is the only lever on this OpenAI-compatible endpoint.
                temperature=0.3,
            )
            caption = (resp.choices[0].message.content or "").strip()
            if not caption or resp.choices[0].finish_reason == "length":
                raise RuntimeError(f"incomplete caption (finish_reason={resp.choices[0].finish_reason}, "
                                   f"completion_tokens={resp.usage.completion_tokens if resp.usage else '?'})")
            logger.info(f"Caption generated via {model} ({len(caption)} chars)")
            return caption
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            rate_limited = "429" in msg or "rate" in msg or "quota" in msg
            wait = (10 * (attempt + 1)) if rate_limited else (2 * (attempt + 1))
            logger.warning(f"KIT attempt {attempt + 1}/{n_attempts} failed "
                           f"({'rate-limited' if rate_limited else 'error'}): {e}")
            if attempt < n_attempts - 1:
                time.sleep(wait)

    raise RuntimeError(f"caption generation failed after {n_attempts} attempts: {last_err}")
