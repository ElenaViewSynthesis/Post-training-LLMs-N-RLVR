"""
Serve the MITRE ATT&CK fine-tuned Gemma adapter as a local HTTP endpoint.

Run on Lambda:
    CUDA_VISIBLE_DEVICES=0 uvicorn serve_mitre_endpoint:app --host 0.0.0.0 --port 8000

Test from Lambda:
    curl -s http://127.0.0.1:8000/analyze \
      -H 'Content-Type: application/json' \
      -d '{"question":"Explain T1059 Command and Scripting Interpreter and give detections."}'
"""

import os
from typing import Literal

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

BASE_MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-12B-it")
ADAPTER_PATH = os.getenv("MITRE_ADAPTER_PATH", "./gemma4-mitre-sft")
HF_TOKEN = os.getenv("HF_TOKEN")
GPU_ID = os.getenv("GPU_ID")
DEVICE_MAP = f"cuda:{GPU_ID}" if GPU_ID and GPU_ID.isdigit() else "auto"

SYSTEM_PROMPT = (
    "You are a cybersecurity expert with deep knowledge of the MITRE ATT&CK framework. "
    "Answer questions about adversary tactics, techniques, threat groups, malware, "
    "and mitigations accurately and concisely."
)

app = FastAPI(title="Gemma MITRE ATT&CK Endpoint")

tokenizer = None
model = None

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemma MITRE ATT&CK</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #171a1f;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main {
      width: min(960px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      font-size: 13px;
      color: #4f5b6b;
      white-space: nowrap;
    }
    form {
      display: grid;
      gap: 12px;
      padding: 16px;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      background: #ffffff;
    }
    label {
      font-size: 14px;
      font-weight: 600;
      color: #2d3642;
    }
    textarea {
      width: 100%;
      min-height: 136px;
      resize: vertical;
      border: 1px solid #c7ced9;
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      line-height: 1.45;
      color: #171a1f;
      background: #ffffff;
    }
    .controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .tokens {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #4f5b6b;
      font-size: 13px;
    }
    input[type="number"] {
      width: 96px;
      border: 1px solid #c7ced9;
      border-radius: 6px;
      padding: 8px;
      font: inherit;
    }
    button {
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 0 16px;
      font: inherit;
      font-weight: 700;
      color: #ffffff;
      background: #176b5d;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      background: #82918e;
    }
    section {
      margin-top: 16px;
      padding: 16px;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      background: #ffffff;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 15px;
      line-height: 1.3;
      letter-spacing: 0;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      color: #1f2933;
    }
    .error { color: #9b1c1c; }
    @media (max-width: 640px) {
      main { width: min(100vw - 20px, 960px); padding: 18px 0; }
      header { display: block; }
      .status { margin-top: 6px; }
      .controls { align-items: stretch; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Gemma MITRE ATT&CK</h1>
      <div class="status" id="status">Ready</div>
    </header>

    <form id="ask-form">
      <label for="question">Question</label>
      <textarea id="question" name="question">Analyze MITRE ATT&CK technique T1059 Command and Scripting Interpreter. Include attacker behavior, detections, and mitigations.</textarea>
      <div class="controls">
        <label class="tokens" for="max-tokens">
          Max tokens
          <input id="max-tokens" name="max_tokens" type="number" min="1" max="4096" value="2000">
        </label>
        <button id="submit" type="submit">Ask Gemma</button>
      </div>
    </form>

    <section>
      <h2>Answer</h2>
      <pre id="answer"></pre>
    </section>
  </main>

  <script>
    const form = document.getElementById("ask-form");
    const statusEl = document.getElementById("status");
    const answerEl = document.getElementById("answer");
    const submit = document.getElementById("submit");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = document.getElementById("question").value.trim();
      const maxNewTokens = Number(document.getElementById("max-tokens").value || 2000);
      if (!question) return;

      submit.disabled = true;
      statusEl.textContent = "Generating";
      answerEl.className = "";
      answerEl.textContent = "";

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            max_new_tokens: maxNewTokens
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(JSON.stringify(data, null, 2));
        }
        answerEl.textContent = data.answer || "";
        statusEl.textContent = "Ready";
      } catch (error) {
        answerEl.className = "error";
        answerEl.textContent = String(error.message || error);
        statusEl.textContent = "Error";
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    system_prompt: str = SYSTEM_PROMPT
    max_new_tokens: int = Field(2000, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    do_sample: bool = False
    return_format: Literal["text"] = "text"


class AnalyzeResponse(BaseModel):
    answer: str
    base_model: str
    adapter_path: str


def load_model_once():
    global tokenizer, model
    if model is not None and tokenizer is not None:
        return

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else BASE_MODEL_ID,
        token=HF_TOKEN,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=DEVICE_MAP,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()


@app.on_event("startup")
def startup():
    load_model_once()


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "loading",
        "base_model": BASE_MODEL_ID,
        "adapter_path": ADAPTER_PATH,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    load_model_once()

    messages = [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": request.max_new_tokens,
        "do_sample": request.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if request.do_sample:
        generation_kwargs.update(
            {
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
        )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return AnalyzeResponse(
        answer=answer,
        base_model=BASE_MODEL_ID,
        adapter_path=ADAPTER_PATH,
    )
