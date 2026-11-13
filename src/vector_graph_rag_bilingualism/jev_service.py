"""Connection presets and accounting for System One-compatible services.

Scoring, payload construction and retry behavior remain in pinned upstream code.
The observer receives metadata only, never credentials or source texts.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import threading
import time
from .config import ModelService

PRESETS = {
    "typesafe": ("https://api.typesafe.ai/v1", "TYPESAFE_API_KEY", 3),
    "jevhub": ("https://api.jevgo.cn/v1", "JEVHUB_API_KEY", 2),
}


def resolve_service(*, service="typesafe", model="jev-1.13.0", base_url=None,
                    key_env=None, timeout=60, max_concurrency=None):
    if service not in PRESETS:
        raise ValueError("Unknown Jev service; use typesafe or jevhub")
    endpoint, env, concurrency = PRESETS[service]
    base_url, key_env = (base_url or endpoint).rstrip("/"), key_env or env
    max_concurrency = concurrency if max_concurrency is None else max_concurrency
    ModelService(base_url, model, key_env, timeout)
    if not model.startswith("jev-") or model in {"jev-latest", "jev-preview"}:
        raise ValueError("Pin a released Jev model")
    if not isinstance(max_concurrency, int) or not 1 <= max_concurrency <= 16:
        raise ValueError("Invalid Jev concurrency")
    if service == "jevhub" and max_concurrency > 4:
        raise ValueError("JevHub permits at most 4 concurrent requests per key")
    return dict(service=service, model=model, base_url=base_url, key_env=key_env,
                timeout=timeout, max_concurrency=max_concurrency)


def model_connection(options):
    key = os.getenv(options["key_env"])
    if not key:
        raise ValueError(f"Missing Jev credential environment variable: {options['key_env']}")
    return dict(model=options["model"], base_url=options["base_url"], api_key=key)


class JevAccounting:
    """Observe each HTTP attempt, including billed responses rejected by validation.

    Absence of usage/charge is unknown, not zero. Amounts are reported by the
    gateway, not independently verified invoices. No assumed provider price.
    One instance limits all calls through its attached native client.
    """
    def __init__(self, options, observer=None):
        self.options, self.observer = options, observer
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(options["max_concurrency"])
        self.next_request = 0.0
        self.counts = Counter()
        self.usage = Counter()
        self.charge = Decimal(0)

    def snapshot(self):
        with self.lock:
            return {"service": self.options["service"], "counts": dict(self.counts),
                    "reported_usage": dict(self.usage), "reported_charge_usd": str(self.charge),
                    "charge_source": "X-JevHub-Charge-USD", "invoice_verified": False}

    def attach(self, native):
        original = native._request
        accounting = self

        class Client:
            def __init__(self, client):
                self.client = client

            def post(self, url, *, json):
                return accounting.post(self.client, url, json)

        native._request = lambda client, payload: original(Client(client), payload)
        return self

    def post(self, client, url, payload):
        event = {"service": self.options["service"], "requested_model": payload.get("model"),
                 "stage": "relations" if "candidate_graph" in payload.get("state", {}) else "passages",
                 "questions": len(payload.get("questions", {})),
                 "request_hash": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                 "usage": None, "charge_usd": None}
        started = time.monotonic()
        try:
            with self.slots:
                if self.options["service"] == "jevhub":
                    with self.lock:
                        scheduled = max(time.monotonic(), self.next_request)
                        self.next_request = scheduled + 0.25
                    time.sleep(max(0, scheduled-time.monotonic()))
                response = client.post(url, json=payload)
            event["http_status"] = response.status_code
            raw_charge = response.headers.get("X-JevHub-Charge-USD")
            if raw_charge is not None:
                try:
                    charge = Decimal(raw_charge)
                    if not charge.is_finite() or charge < 0:
                        raise ValueError("Invalid charge")
                    event["charge_usd"] = str(charge)
                except (InvalidOperation, ValueError):
                    event["invalid_charge_header"] = True
            try:
                data = response.json()
                if isinstance(data, dict):
                    if isinstance(data.get("model"), str):
                        event["response_model"] = data["model"]
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        event["usage"] = {k:v for k,v in usage.items()
                                          if k in ("input_tokens", "output_tokens") and type(v) is int and v >= 0}
            except (ValueError, TypeError):
                pass
            return response
        except Exception as error:
            event["error"] = type(error).__name__
            raise
        finally:
            event["elapsed_ms"] = round((time.monotonic()-started)*1000)
            with self.lock:
                self.counts["http_attempts"] += 1
                self.counts[str(event.get("http_status", "transport_error"))] += 1
                self.counts["charge_reported" if event["charge_usd"] is not None else "charge_unknown"] += 1
                self.counts["usage_reported" if event["usage"] else "usage_unknown"] += 1
                if event["charge_usd"] is not None:
                    self.charge += Decimal(event["charge_usd"])
                self.usage.update(event["usage"] or {})
            if self.observer:
                self.observer(event)
