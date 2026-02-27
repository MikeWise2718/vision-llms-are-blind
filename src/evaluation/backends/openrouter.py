"""OpenRouter API backend for VLM evaluation."""

import base64
import json
import os
import time

import urllib.request
import urllib.error

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class OpenRouterClient:
    def __init__(self, api_key: str = "", base_url: str = OPENROUTER_BASE_URL):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it as an environment variable."
            )
        self.base_url = base_url
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def query(
        self,
        model: str,
        prompt: str,
        image_path: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        max_retries: int = 3,
    ) -> dict:
        """Send an image + prompt to a VLM via OpenRouter.

        Returns dict with keys: response, tokens_used, model, error
        """
        image_b64 = self._encode_image(image_path)
        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    self.base_url, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                response_text = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0) or (input_tokens + output_tokens)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_requests += 1

                return {
                    "response": response_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_used": total_tokens,
                    "model": model,
                    "error": None,
                }
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code == 429 or e.code >= 500:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retrying in {wait}s (HTTP {e.code})...")
                    time.sleep(wait)
                    continue
                return {
                    "response": "",
                    "input_tokens": 0, "output_tokens": 0, "tokens_used": 0,
                    "model": model,
                    "error": f"HTTP {e.code}: {body}",
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {
                    "response": "",
                    "input_tokens": 0, "output_tokens": 0, "tokens_used": 0,
                    "model": model,
                    "error": str(e),
                }

        return {"response": "", "input_tokens": 0, "output_tokens": 0, "tokens_used": 0, "model": model, "error": "Max retries exceeded"}

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
