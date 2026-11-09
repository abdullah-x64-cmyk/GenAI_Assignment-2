"""Local browser app for criterion-level trial screening.

The application intentionally keeps inference local. It serves a small vanilla
JS interface and exposes only health, benchmark, and screening endpoints.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
import webbrowser
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT / "src"))
CONFIG_PATH = ROOT / "configs" / "qwen_lora.json"
ARTIFACT_URL = "https://github.com/abdullah-x64-cmyk/GenAI_Assignment-2/raw/refs/heads/main/results/task1_submission_artifacts.zip"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def ensure_adapter(config):
    adapter_dir = ROOT / config["output_dir"] / "best_adapter"
    if adapter_dir.exists():
        return adapter_dir
    archive = ROOT / "results" / "task1_submission_artifacts.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(ARTIFACT_URL, timeout=45) as response, archive.open("wb") as dest:
                dest.write(response.read())
        except Exception as exc:
            raise RuntimeError(
                "The trained adapter is missing. Place task1_submission_artifacts.zip in results/ "
                "or restore it from the project README download link, then restart the app."
            ) from exc
    wanted_prefix = (Path(config["output_dir"]) / "best_adapter").as_posix() + "/"
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            member_path = Path(member)
            if not member.startswith(wanted_prefix) or member_path.is_absolute() or ".." in member_path.parts:
                continue
            if not member.endswith("/"):
                bundle.extract(member, ROOT)
    if not adapter_dir.exists():
        raise RuntimeError(f"The adapter archive did not contain {adapter_dir.relative_to(ROOT)}")
    return adapter_dir


class ScreeningService:
    def __init__(self):
        self.config = load_config()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.draft = None
        self.model_lock = threading.RLock()
        self.spec_lock = threading.RLock()
        self.inference_lock = threading.Lock()

    def status(self):
        adapter = ROOT / self.config["output_dir"] / "best_adapter"
        import importlib.util
        dependencies_available = all(importlib.util.find_spec(name) is not None
                                     for name in ("torch", "transformers", "peft"))
        return {
            "model": self.config["model_name"],
            "adapter_available": adapter.exists() or (ROOT / "results" / "task1_submission_artifacts.zip").exists(),
            "dependencies_available": dependencies_available,
            "loaded": self.model is not None,
            "device": str(self.device) if self.device is not None else ("cuda" if self._cuda_available() else "cpu"),
            "max_input_tokens": self.config["max_length"] - self.config["max_new_tokens"],
            "max_new_tokens": self.config["max_new_tokens"],
        }

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def load(self):
        if self.model is not None:
            return
        with self.model_lock:
            if self.model is not None:
                return
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Inference packages are not installed. Double-click setup_and_run.bat once, "
                    "then use the browser page it opens."
                ) from exc

            adapter_dir = ensure_adapter(self.config)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            # Preserve the end of the prompt (criterion and recent note text) if
            # the configured model input window is exceeded. The UI reports it.
            self.tokenizer.truncation_side = "left"
            base = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"], torch_dtype=dtype, attn_implementation="eager"
            )
            self.model = PeftModel.from_pretrained(base, str(adapter_dir)).to(self.device)
            self.model.eval()

    def _inputs(self, prompt):
        import torch
        token_limit = self.config["max_length"] - self.config["max_new_tokens"]
        full_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=token_limit
        )
        return {k: v.to(self.device) for k, v in encoded.items()}, len(full_ids), token_limit

    def screen(self, note, criteria, decoding):
        import torch
        from common import prompt_for
        from evaluate import explanation_for, output_fields, sentence_map

        self.load()
        results = []
        max_input = self.config["max_length"] - self.config["max_new_tokens"]
        for criterion in criteria:
            row = {
                "note": note,
                "criterion_type": criterion["type"],
                "criterion_text": criterion["text"],
            }
            prompt = prompt_for(row)
            encoded, original_tokens, token_limit = self._inputs(prompt)
            input_ids = encoded["input_ids"]
            truncated = original_tokens > token_limit
            started = time.perf_counter()
            speculative_stats = None
            if decoding == "speculative":
                with self.spec_lock:
                    if self.draft is None:
                        from benchmark_speculative import make_draft
                        self.model, self.draft = make_draft(self.model, layers=2)
                        self.model.eval().to(self.device)
                        self.draft.eval().to(self.device)
                    from speculative_decode import speculative_greedy
                    generated = speculative_greedy(
                        self.model, self.draft, input_ids,
                        self.config["max_new_tokens"], draft_tokens=4,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    token_ids = generated.generated_token_ids
                    speculative_stats = {
                        "drafted_tokens": generated.drafted_tokens,
                        "accepted_draft_tokens": generated.accepted_draft_tokens,
                        "acceptance_rate": generated.acceptance_rate,
                        "rounds": generated.rounds,
                    }
            else:
                with torch.inference_mode():
                    output = self.model.generate(
                        **encoded,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                token_ids = output[0, input_ids.shape[-1]:].tolist()
            elapsed = time.perf_counter() - started
            raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            label, evidence_ids = output_fields(raw)
            sentences = sentence_map(note)
            cited = [
                {"sentence_id": int(item), "text": sentences[int(item)]}
                for item in evidence_ids if int(item) in sentences
            ]
            invalid = [int(item) for item in evidence_ids if int(item) not in sentences]
            results.append({
                "criterion_type": criterion["type"],
                "criterion": criterion["text"],
                "predicted_category": label,
                "cited_sentence_ids": [item["sentence_id"] for item in cited],
                "evidence_text": cited,
                "invalid_sentence_ids": invalid,
                "explanation": explanation_for(label, criterion["type"], criterion["text"], cited, invalid),
                "decode_mode": decoding,
                "elapsed_seconds": elapsed,
                "generated_tokens": len(token_ids),
                "truncated": truncated,
                "original_prompt_tokens": original_tokens,
                "used_prompt_tokens": min(original_tokens, max_input),
                "speculative_stats": speculative_stats,
            })
        return {
            "model": self.config["model_name"],
            "mode": "fine-tuned LoRA adapter",
            "device": str(self.device),
            "decoding": decoding,
            "notice": "Educational screening prototype only; not a clinical enrollment decision.",
            "results": results,
        }


SERVICE = ScreeningService()


def benchmark_payload():
    paths = {
        "cache": ROOT / "results" / "cache_benchmark_final.json",
        "speculative": ROOT / "results" / "speculative_benchmark_t4.json",
    }
    payload = {}
    for name, path in paths.items():
        if path.exists():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "TrialCriterionScreen/1.0"

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/status":
            return self._send(200, json.dumps(SERVICE.status()))
        if self.path == "/api/benchmarks":
            return self._send(200, json.dumps(benchmark_payload()))
        assets = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
        if self.path not in assets:
            return self._send(404, "Not found", "text/plain; charset=utf-8")
        file_path = WEB / assets[self.path]
        content_type = "text/html; charset=utf-8" if self.path == "/" else (
            "text/javascript; charset=utf-8" if self.path.endswith(".js") else "text/css; charset=utf-8"
        )
        return self._send(200, file_path.read_bytes(), content_type)

    def do_POST(self):
        if self.path != "/api/screen":
            return self._send(404, json.dumps({"error": "Not found"}))
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 1_000_000:
                raise ValueError("Request must be between 1 byte and 1 MB.")
            request = json.loads(self.rfile.read(size))
            note = str(request.get("patient_note", "")).strip()
            criteria = request.get("criteria", [])
            decoding = request.get("decoding", "cached")
            if not note:
                raise ValueError("Enter a patient record before screening.")
            if len(note) > 100_000:
                raise ValueError("Patient record is too long (maximum 100,000 characters).")
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= 20:
                raise ValueError("Add between 1 and 20 criteria.")
            if decoding not in {"cached", "speculative"}:
                raise ValueError("Choose a supported decoding method.")
            cleaned = []
            for item in criteria:
                kind = str(item.get("type", "")).lower().strip()
                text = str(item.get("text", "")).strip()
                if kind not in {"inclusion", "exclusion"} or not text:
                    raise ValueError("Each criterion needs a type and criterion text.")
                if len(text) > 5_000:
                    raise ValueError("Each criterion must be at most 5,000 characters.")
                cleaned.append({"type": kind, "text": text})
            with SERVICE.inference_lock:
                result = SERVICE.screen(note, cleaned, decoding)
            return self._send(200, json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            return self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False))

    def log_message(self, fmt, *args):
        # Keep patient note content out of server logs; only standard access
        # metadata is logged, and request bodies are never written.
        super().log_message(fmt, *args)


def main():
    host = os.environ.get("TRIAL_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("TRIAL_APP_PORT", "7860"))
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Trial screening app running at {url}")
    if os.environ.get("TRIAL_APP_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
