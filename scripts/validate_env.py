"""Phase 0 gate: validate Gemini API access before any app code depends on it.

Deliberately dependency-free (stdlib only) so it can run before `pip install`.

Usage:
    python scripts/validate_env.py              # full live validation
    python scripts/validate_env.py --offline    # presence/format checks only (CI)
    python scripts/validate_env.py --json       # machine-readable report

Exit code 0 = every check passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILENAMES = (".env",)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    @property
    def label(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.ok else "FAIL"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", skipped: bool = False) -> None:
        self.checks.append(Check(name, ok, detail, skipped))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.skipped]

    def as_dict(self) -> dict:
        return {
            "ok": not self.failed,
            "checks": [
                {"name": c.name, "status": c.label, "detail": c.detail}
                for c in self.checks
            ],
        }


# --------------------------------------------------------------------------- #
# .env loading (no python-dotenv needed)
# --------------------------------------------------------------------------- #
def candidate_env_files() -> list[Path]:
    files: list[Path] = []
    for directory in (REPO_ROOT, REPO_ROOT.parent):
        for name in ENV_FILENAMES:
            path = directory / name
            if path.is_file():
                files.append(path)
    return files


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_settings() -> tuple[dict[str, str], Path | None]:
    """Real environment wins; .env only fills gaps."""
    env: dict[str, str] = {}
    source: Path | None = None
    for path in candidate_env_files():
        for key, value in parse_env_file(path).items():
            if key not in env:
                env[key] = value
                source = source or path
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL"):
        if os.environ.get(key):
            env[key] = os.environ[key]
            source = source or Path("<process environment>")
    return env, source


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def request(
    url: str, api_key: str, *, payload: dict | None = None, timeout: float = 30.0
) -> tuple[int, dict]:
    data = None
    headers = {"x-goog-api-key": api_key, "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed
    except urllib.error.URLError as exc:
        return 0, {"error": {"message": f"network error: {exc.reason}"}}
    except TimeoutError:
        return 0, {"error": {"message": f"timeout after {timeout}s"}}


def error_message(payload: dict) -> str:
    err = payload.get("error")
    if isinstance(err, dict):
        return str(err.get("message", err))[:300]
    if err:
        return str(err)[:300]
    return "unknown error"


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def run_checks(env: dict[str, str], *, offline: bool, timeout: float) -> Report:
    report = Report()
    api_key = env.get("GEMINI_API_KEY", "")
    model = env.get("GEMINI_MODEL", "").strip()
    embedding_model = env.get("GEMINI_EMBEDDING_MODEL", "").strip() or "gemini-embedding-001"

    report.add("API key present", bool(api_key), "GEMINI_API_KEY is empty" if not api_key else "")
    if not api_key:
        return report

    report.add(
        "API key format",
        len(api_key) >= 20 and not any(ch.isspace() for ch in api_key),
        f"length={len(api_key)}"
        + (" (note: 'AIza' prefix is typical; non-standard prefixes can still work)"
           if not api_key.startswith("AIza") else ""),
    )

    if not model:
        report.add("GEMINI_MODEL set", False, "GEMINI_MODEL is empty")
        return report
    report.add("GEMINI_MODEL set", True, model)

    if offline:
        report.add("API reachable", True, "skipped (--offline)", skipped=True)
        report.add("Model available", True, "skipped (--offline)", skipped=True)
        report.add("Inference works", True, "skipped (--offline)", skipped=True)
        report.add("Embedding model available", True, "skipped (--offline)", skipped=True)
        return report

    status, payload = request(f"{API_ROOT}/models?pageSize=1000", api_key, timeout=timeout)
    if status != 200:
        report.add("API reachable", False, f"HTTP {status}: {error_message(payload)}")
        return report

    names = {m.get("name", "").removeprefix("models/") for m in payload.get("models", [])}
    report.add("API reachable", True, f"{len(names)} models visible for this key")

    report.add(
        "Model available",
        model in names,
        model if model in names else f"'{model}' not in key's model list",
    )

    status, payload = request(
        f"{API_ROOT}/models/{urllib.parse.quote(model)}:generateContent",
        api_key,
        payload={
            "contents": [{"parts": [{"text": "Reply with exactly: OK"}]}],
            "generationConfig": {"maxOutputTokens": 8, "temperature": 0},
        },
        timeout=timeout,
    )
    if status != 200:
        report.add("Inference works", False, f"HTTP {status}: {error_message(payload)}")
    else:
        finish = (payload.get("candidates") or [{}])[0].get("finishReason", "?")
        usage = payload.get("usageMetadata", {})
        report.add(
            "Inference works",
            finish not in ("SAFETY", "RECITATION"),
            f"finishReason={finish}, promptTokens={usage.get('promptTokenCount', '?')}",
        )

    report.add(
        "Embedding model available",
        embedding_model in names,
        embedding_model if embedding_model in names else f"'{embedding_model}' not in model list",
    )
    return report


def print_report(report: Report, env_source: Path | None) -> None:
    print()
    print("Gemini environment validation")
    print("=" * 62)
    print(f"env source : {env_source if env_source else 'none found (no .env, no process env)'}")
    print("-" * 62)
    for check in report.checks:
        print(f"[{check.label:4}] {check.name:<28} {check.detail}")
    print("-" * 62)
    if report.failed:
        print(f"RESULT: FAIL ({len(report.failed)} check(s) failed)")
    else:
        print("RESULT: PASS - environment is ready")
    print()
    if report.failed:
        print("Next steps:")
        print("  1. Copy .env.example to .env and set GEMINI_API_KEY.")
        print("  2. Create/replace the key at https://aistudio.google.com/app/apikey")
        print("  3. Re-run: python scripts/validate_env.py")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    parser.add_argument("--json", action="store_true", help="emit JSON report only")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout seconds")
    args = parser.parse_args(argv)

    env, source = load_settings()
    report = run_checks(env, offline=args.offline, timeout=args.timeout)

    if args.json:
        print(json.dumps(report.as_dict() | {"env_source": str(source) if source else None}, indent=2))
    else:
        print_report(report, source)

    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
