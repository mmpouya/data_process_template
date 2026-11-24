"""Decision model client (System-1 reasoning: H2O-Lightning-4B / Jev-1.13).

Provides fast, token-free, calibrated classification for:
- Task A: Statutory grounding & reflection
- Task C: Normative PKG relation classification
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from src.core.config import get_settings
from src.utils.rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)

RELATION_CRITERIA: dict[str, str] = {
    "condition_of": "شرط و مقدمه تحقق گزاره دیگر",
    "exception_to": "استثنا و خروج موضوعی از گزاره دیگر",
    "elaborates": "توضیح، تفصیل یا بسط معنایی گزاره دیگر",
    "procedurally_followed_by": "گام و مرحله پسینی اجرایی یا اداری",
    "presupposes": "پیش‌فرض موضوعی، مفهومی یا صلاحیتی",
    "qualifies": "تحدید، مقیدسازی یا قید شمول گزاره دیگر",
    "alternative_to": "حکم یا گزینه جایگزین و تخییری",
    "none": "هیچ رابطه هنجاری مستقیمی میان این دو گزاره وجود ندارد",
}


class DecisionManager:
    """Manages connections and requests to the System-1 decision model (H2O/Jev)."""

    _client: httpx.AsyncClient | None = None
    _limiter: AsyncRateLimiter | None = None
    _relation_cache: dict[tuple[str, str], dict[str, Any]] = {}
    _MAX_CACHE_SIZE: int = 2000

    @classmethod
    def is_enabled(cls) -> bool:
        settings = get_settings()
        return settings.DECISION_MODEL_PROVIDER.lower() not in (
            "",
            "none",
            "disabled",
            "false",
        )

    @classmethod
    async def init_client(cls) -> httpx.AsyncClient | None:
        if not cls.is_enabled():
            return None
        if cls._client is None:
            settings = get_settings()
            cls._limiter = AsyncRateLimiter(rpm=settings.DECISION_RPM, name="decision")
            cls._client = httpx.AsyncClient(
                base_url=settings.DECISION_BASE_URL.rstrip("/"),
                timeout=httpx.Timeout(120.0, connect=15.0),
                trust_env=False,  # Bypass Windows system proxy for localhost
            )
            # Non-fatal connection check
            try:
                # Send a lightweight ping payload
                ping_payload = {
                    "model": settings.DECISION_MODEL_NAME,
                    "state": "تست ارتباط",
                    "questions": {
                        "ping": {
                            "type": "noul",
                            "instructions": "تست",
                            "criteria": {"true": "بله", "false": "خیر"},
                        }
                    },
                }
                headers = {"Content-Type": "application/json"}
                if settings.DECISION_API_KEY and settings.DECISION_API_KEY != "none":
                    headers["Authorization"] = f"Bearer {settings.DECISION_API_KEY}"
                else:
                    headers["Authorization"] = "Bearer none"
                resp = await cls._client.post(
                    "/systemone", json=ping_payload, headers=headers
                )
                if resp.status_code == 200:
                    logger.info(
                        "Decision model connection established (%s)",
                        settings.DECISION_MODEL_NAME,
                    )
                else:
                    logger.warning(
                        "Decision model ping returned status %d: %s",
                        resp.status_code,
                        resp.text[:150],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Decision model connection test failed (non-fatal): %s", exc
                )
        return cls._client

    @classmethod
    async def close_client(cls) -> None:
        if cls._client is not None:
            try:
                await cls._client.aclose()
            except Exception:
                pass
            cls._client = None
        cls._relation_cache.clear()

    @classmethod
    async def _post_systemone(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a System-1 query with retry and rate-limiting."""
        if not cls.is_enabled():
            raise RuntimeError("Decision model is disabled in settings")
        if cls._client is None:
            await cls.init_client()
        if cls._client is None:
            raise RuntimeError("Decision client could not be initialized")

        settings = get_settings()
        if cls._limiter is None:
            cls._limiter = AsyncRateLimiter(rpm=settings.DECISION_RPM, name="decision")

        body = dict(payload)
        if "model" not in body:
            body["model"] = settings.DECISION_MODEL_NAME

        headers = {"Content-Type": "application/json"}
        if settings.DECISION_API_KEY and settings.DECISION_API_KEY != "none":
            headers["Authorization"] = f"Bearer {settings.DECISION_API_KEY}"
        else:
            headers["Authorization"] = "Bearer none"

        max_attempts = 4
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            await cls._limiter.acquire()
            try:
                resp = await cls._client.post("/systemone", json=body, headers=headers)
                if resp.status_code in (429, 502, 503, 504):
                    cls._limiter.pause(3.0)
                    await asyncio.sleep(2.0 + attempt * 1.5)
                    continue
                resp.raise_for_status()
                raw_text = resp.text
                if "data: [DONE]" in raw_text:
                    raw_text = raw_text.split("data: [DONE]")[0].strip()
                return json.loads(raw_text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await asyncio.sleep(1.0 + attempt * 1.5)

        err_detail = str(last_exc) or repr(last_exc)
        raise RuntimeError(
            f"Decision model request failed after {max_attempts} attempts: {err_detail}"
        )

    @classmethod
    async def classify_grounding(
        cls, chunk_text: str, statement_text: str
    ) -> dict[str, Any]:
        """Evaluate statutory grounding (Task A) of a proposition against source chunks.

        Returns:
            dict containing:
            - is_grounded: bool (P >= 0.5)
            - p_grounded: float (0.0 to 1.0)
            - confidence: float (distance from uncertainty)
            - decision: "accept" (>= accept_threshold) | "reject" (< reject_threshold) | "uncertain"
            - raw: raw response dictionary
        """
        settings = get_settings()
        if not chunk_text.strip() or not statement_text.strip():
            return {
                "is_grounded": False,
                "p_grounded": 0.0,
                "confidence": 1.0,
                "decision": "reject",
                "raw": {},
            }

        payload = {
            "state": chunk_text,
            "questions": {
                "grounded": {
                    "type": "noul",
                    "instructions": f"آیا این گزاره مستند به متن است و از آن نتیجه می‌شود؟\nگزاره: {statement_text}",
                    "criteria": {
                        "true": "مستند و ناشی از متن قانون",
                        "false": "نامستند یا خلاف متن",
                    },
                }
            },
        }

        t0 = time.time()
        resp = await cls._post_systemone(payload)
        latency_ms = (time.time() - t0) * 1000.0

        ans = resp.get("answers", {}).get("grounded", {})
        if isinstance(ans, (int, float)):
            prob_true = float(ans)
        elif isinstance(ans, dict):
            prob_true = float(ans.get("noul", ans.get("score", ans.get("value", 0.5))))
        else:
            prob_true = 0.5
        is_grounded = prob_true >= 0.5
        confidence = prob_true if is_grounded else (1.0 - prob_true)

        if prob_true >= settings.DECISION_GROUNDING_ACCEPT_THRESHOLD:
            decision = "accept"
        elif prob_true < settings.DECISION_GROUNDING_REJECT_THRESHOLD:
            decision = "reject"
        else:
            decision = "uncertain"

        return {
            "is_grounded": is_grounded,
            "p_grounded": prob_true,
            "confidence": confidence,
            "decision": decision,
            "latency_ms": latency_ms,
            "usage": resp.get("usage", {}),
            "raw": ans,
        }

    @classmethod
    async def classify_grounding_batch(
        cls, chunk_text: str, statement_texts: list[str]
    ) -> list[dict[str, Any]]:
        """Evaluate statutory grounding for a batch of candidate statements against chunk_text.

        Bundles all statements into a single /systemone payload sharing the chunk state,
        per System-1 integration guidance.
        """
        settings = get_settings()
        if not chunk_text.strip() or not statement_texts:
            return [
                {
                    "is_grounded": False,
                    "p_grounded": 0.0,
                    "confidence": 1.0,
                    "decision": "reject",
                    "raw": {},
                }
                for _ in statement_texts
            ]

        # Build parallel questions under the shared chunk state
        questions = {
            f"q_{idx}": {
                "type": "noul",
                "instructions": f"آیا این گزاره مستند به متن است و از آن نتیجه می‌شود؟\nگزاره: {text.strip()}",
                "criteria": {
                    "true": "مستند و ناشی از متن قانون",
                    "false": "نامستند یا خلاف متن",
                },
            }
            for idx, text in enumerate(statement_texts)
            if text.strip()
        }

        payload = {
            "state": chunk_text,
            "questions": questions,
        }

        t0 = time.time()
        resp = await cls._post_systemone(payload)
        latency_ms = (time.time() - t0) * 1000.0

        answers = resp.get("answers", {})
        results: list[dict[str, Any]] = []

        for idx, text in enumerate(statement_texts):
            if not text.strip():
                results.append(
                    {
                        "is_grounded": False,
                        "p_grounded": 0.0,
                        "confidence": 1.0,
                        "decision": "reject",
                        "raw": {},
                    }
                )
                continue

            ans = answers.get(f"q_{idx}", {})
            if isinstance(ans, (int, float)):
                prob_true = float(ans)
            elif isinstance(ans, dict):
                prob_true = float(
                    ans.get("noul", ans.get("score", ans.get("value", 0.5)))
                )
            else:
                prob_true = 0.5

            is_grounded = prob_true >= 0.5
            confidence = prob_true if is_grounded else (1.0 - prob_true)
            if prob_true >= settings.DECISION_GROUNDING_ACCEPT_THRESHOLD:
                decision = "accept"
            elif prob_true < settings.DECISION_GROUNDING_REJECT_THRESHOLD:
                decision = "reject"
            else:
                decision = "uncertain"

            results.append(
                {
                    "is_grounded": is_grounded,
                    "p_grounded": prob_true,
                    "confidence": confidence,
                    "decision": decision,
                    "latency_ms": latency_ms,
                    "raw": ans,
                }
            )

        return results

    @classmethod
    async def classify_relation(
        cls, stmt_a_text: str, stmt_b_text: str
    ) -> dict[str, Any]:
        """Evaluate normative coupling / relation (Task C) between two statements.

        Returns:
            dict containing:
            - prediction: str (e.g. "condition_of", "none")
            - p_none: float (probability statements are completely unrelated)
            - is_uncoupled: bool (p_none >= relation_guard_threshold)
            - confidence: float
            - probabilities: dict[str, float]
        """
        settings = get_settings()
        cache_key = (stmt_a_text.strip(), stmt_b_text.strip())
        if cache_key in cls._relation_cache:
            return cls._relation_cache[cache_key]

        state_str = (
            f"گزاره اول (الف): {stmt_a_text.strip()}\n"
            f"گزاره دوم (ب): {stmt_b_text.strip()}"
        )
        payload = {
            "state": state_str,
            "questions": {
                "relation": {
                    "type": "choice",
                    "instructions": "چه رابطه هنجاری میان گزاره اول و گزاره دوم برقرار است؟",
                    "criteria": RELATION_CRITERIA,
                }
            },
        }

        t0 = time.time()
        resp = await cls._post_systemone(payload)
        latency_ms = (time.time() - t0) * 1000.0

        ans = resp.get("answers", {}).get("relation", {})
        if isinstance(ans, dict):
            choice = str(ans.get("choice", ans.get("prediction", "none")))
            probs = ans.get("probabilities", {})
            if not probs and choice:
                probs = {choice: float(ans.get("confidence", 0.5))}
            p_none = float(probs.get("none", 1.0 if choice == "none" else 0.0))
            confidence = float(ans.get("confidence", probs.get(choice, 0.5)))
        else:
            choice = "none"
            probs = {}
            p_none = 1.0
            confidence = 0.5

        is_uncoupled = p_none >= settings.DECISION_RELATION_GUARD_THRESHOLD

        result = {
            "prediction": choice,
            "p_none": p_none,
            "is_uncoupled": is_uncoupled,
            "confidence": confidence,
            "probabilities": probs,
            "latency_ms": latency_ms,
            "usage": resp.get("usage", {}),
            "raw": ans,
        }

        if len(cls._relation_cache) >= cls._MAX_CACHE_SIZE:
            cls._relation_cache.clear()
        cls._relation_cache[cache_key] = result
        return result
