"""OpenRouter image generation backend for Hermes.

Uses OpenRouter's Chat Completions image-output flow and saves data-URL
responses under ``$HERMES_HOME/cache/images/`` when possible.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemini-2.5-flash-image"

_MODELS: Dict[str, Dict[str, Any]] = {
    "google/gemini-2.5-flash-image": {
        "display": "Nano Banana (Gemini 2.5 Flash Image)",
        "speed": "fast",
        "strengths": "Good low-cost trial/default image model",
        "modalities": ["image", "text"],
    },
    "google/gemini-3.1-flash-image-preview": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image Preview)",
        "speed": "fast",
        "strengths": "Newer Gemini image preview; supports extended image_config",
        "modalities": ["image", "text"],
    },
    "google/gemini-3-pro-image-preview": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image Preview)",
        "speed": "medium",
        "strengths": "Higher-quality Gemini image generation",
        "modalities": ["image", "text"],
    },
    "openai/gpt-5-image-mini": {
        "display": "GPT-5 Image Mini",
        "speed": "medium",
        "strengths": "OpenAI image generation, cheaper/mini tier",
        "modalities": ["image", "text"],
    },
    "openai/gpt-5-image": {
        "display": "GPT-5 Image",
        "speed": "medium",
        "strengths": "OpenAI image generation",
        "modalities": ["image", "text"],
    },
    "openai/gpt-5.4-image-2": {
        "display": "GPT-5.4 Image 2",
        "speed": "medium",
        "strengths": "OpenAI's newer image model on OpenRouter",
        "modalities": ["image", "text"],
    },
    "black-forest-labs/flux.2-klein-4b": {
        "display": "FLUX.2 Klein 4B",
        "speed": "fast",
        "strengths": "Fast FLUX trial model",
        "modalities": ["image"],
    },
    "black-forest-labs/flux.2-pro": {
        "display": "FLUX.2 Pro",
        "speed": "medium",
        "strengths": "High-quality FLUX generation",
        "modalities": ["image"],
    },
    "black-forest-labs/flux.2-flex": {
        "display": "FLUX.2 Flex",
        "speed": "medium",
        "strengths": "Flexible FLUX generation/editing",
        "modalities": ["image"],
    },
    "recraft/recraft-v4.1": {
        "display": "Recraft V4.1",
        "speed": "medium",
        "strengths": "Design/illustration-oriented generation",
        "modalities": ["image"],
    },
    "recraft/recraft-v4.1-pro": {
        "display": "Recraft V4.1 Pro",
        "speed": "medium",
        "strengths": "Higher-quality Recraft design generation",
        "modalities": ["image"],
    },
    "sourceful/riverflow-v2-fast": {
        "display": "Riverflow V2 Fast",
        "speed": "fast",
        "strengths": "Fast Sourceful image generation",
        "modalities": ["image"],
    },
    "sourceful/riverflow-v2-pro": {
        "display": "Riverflow V2 Pro",
        "speed": "medium",
        "strengths": "Higher-quality Sourceful image generation",
        "modalities": ["image"],
    },
    "x-ai/grok-imagine-image-quality": {
        "display": "Grok Imagine Image Quality",
        "speed": "medium",
        "strengths": "xAI image generation model",
        "modalities": ["image"],
    },
}

_ASPECT_TO_OPENROUTER = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def _load_image_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive config read
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model(explicit: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    candidates = [
        explicit,
        os.environ.get("OPENROUTER_IMAGE_MODEL"),
    ]
    cfg = _load_image_config()
    openrouter_cfg = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}
    if isinstance(openrouter_cfg, dict):
        candidates.append(openrouter_cfg.get("model"))
    candidates.append(cfg.get("model"))
    candidates.append(DEFAULT_MODEL)

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            model_id = candidate.strip()
            return model_id, _MODELS.get(model_id, {"display": model_id, "modalities": ["image", "text"]})
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _extract_image_ref(message: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    images = message.get("images") or []
    if images:
        image_obj = images[0] or {}
        image_url = image_obj.get("image_url") or image_obj.get("imageUrl") or {}
        if isinstance(image_url, dict):
            url = image_url.get("url")
        else:
            url = image_url
        if isinstance(url, str) and url:
            return url, {"image_count": len(images)}

    # Some providers may return mixed content parts instead of message.images.
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image_url") or part.get("imageUrl")
            if isinstance(image_url, dict) and image_url.get("url"):
                return image_url["url"], {"image_count": 1}
            if isinstance(image_url, str) and image_url:
                return image_url, {"image_count": 1}
    return None, {}


def _save_data_url(data_url: str, model_id: str) -> str:
    # Expected format: data:image/png;base64,<payload>
    header, _, b64 = data_url.partition(",")
    if not b64 or ";base64" not in header:
        return data_url
    extension = "png"
    if header.startswith("data:image/"):
        extension = header.split("data:image/", 1)[1].split(";", 1)[0] or "png"
        if extension == "jpeg":
            extension = "jpg"
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model_id)[-80:]
    return str(save_b64_image(b64, prefix=f"openrouter_{safe_model}", extension=extension))


class OpenRouterImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", "varies"),
                "strengths": meta.get("strengths", "OpenRouter image-output model"),
                "price": "OpenRouter pricing",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter",
            "badge": "paid",
            "tag": "Use OpenRouter image-output models with OPENROUTER_API_KEY",
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/keys",
                }
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id, meta = _resolve_model(kwargs.get("model"))

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openrouter",
                model=model_id,
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return error_response(
                error="OPENROUTER_API_KEY not set. Add it to ~/.hermes/.env and restart Hermes.",
                error_type="auth_required",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        modalities = meta.get("modalities") or ["image", "text"]
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": modalities,
            "stream": False,
            "image_config": {
                "aspect_ratio": _ASPECT_TO_OPENROUTER.get(aspect, "16:9"),
            },
        }

        # Allow advanced callers/config to pass OpenRouter image_config values.
        cfg = _load_image_config()
        openrouter_cfg = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}
        configured_image_config = openrouter_cfg.get("image_config") if isinstance(openrouter_cfg, dict) else None
        if isinstance(configured_image_config, dict):
            payload["image_config"].update(configured_image_config)

        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://hermes-agent.local",
                "X-Title": "Hermes Agent Image Generation",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            try:
                details = exc.read().decode("utf-8")
            except Exception:
                details = str(exc)
            return error_response(
                error=f"OpenRouter image generation failed (HTTP {exc.code}): {details[:1000]}",
                error_type="api_error",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenRouter image generation failed", exc_info=True)
            return error_response(
                error=f"OpenRouter image generation failed: {exc}",
                error_type="api_error",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        choices = data.get("choices") or []
        if not choices:
            return error_response(
                error=f"OpenRouter returned no choices: {str(data)[:1000]}",
                error_type="empty_response",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        message = choices[0].get("message") or {}
        image_ref, extra = _extract_image_ref(message)
        if not image_ref:
            return error_response(
                error=f"OpenRouter response contained no image: {str(message)[:1000]}",
                error_type="empty_response",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if image_ref.startswith("data:image/"):
            try:
                image_ref = _save_data_url(image_ref, model_id)
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not save OpenRouter image to cache: {exc}",
                    error_type="io_error",
                    provider="openrouter",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            extra["content"] = content.strip()
        extra["openrouter_aspect_ratio"] = payload["image_config"].get("aspect_ratio")

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openrouter",
            extra=extra,
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(OpenRouterImageGenProvider())
