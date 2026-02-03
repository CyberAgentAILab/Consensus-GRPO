import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from typing import List, Dict, Any, Union, Optional
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from sacrebleu.metrics import BLEU
import random
from comet import download_model, load_from_checkpoint
import re
import time
import hashlib
import sqlite3
import threading

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# Fast path helpers for COMET-based rewards
_COMET_MODEL = None
_COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
_RE_SRC_FROM_PROMPT = re.compile(r"English text:\s*(.*?)\s*\n\n(?:Japanese|German|Chinese|Russian|Czech) translation:", re.S)

_BLEURT_CACHE: Dict[str, Dict[str, Any]] = {}
_JUDGE_LLM_CACHE: Dict[str, Dict[str, Any]] = {}
_EXTERNAL_JUDGE_SESSION = None
_EXTERNAL_JUDGE_MEM_CACHE: Dict[str, str] = {}
_EXTERNAL_JUDGE_MEM_CACHE_MAX = 50_000
_EXTERNAL_JUDGE_LOCK = threading.Lock()

try:
    # Allow disabling Unsloth entirely (e.g. self_judge + plain trainer path)
    _DISABLE_UNSLOTH = os.environ.get("MBR_DISABLE_UNSLOTH", "").strip().lower() in ("1", "true", "yes", "y", "on")
    if _DISABLE_UNSLOTH:
        FastLanguageModel = None
        _UNSLOTH_AVAILABLE = False
    else:
        from unsloth import FastLanguageModel
        _UNSLOTH_AVAILABLE = True
except Exception:
    FastLanguageModel = None
    _UNSLOTH_AVAILABLE = False

def _get_bleurt_components(model_name: str):
    """Lazy-load and cache BLEURT components to avoid repeated downloads/loads."""
    global _BLEURT_CACHE
    if model_name in _BLEURT_CACHE:
        entry = _BLEURT_CACHE[model_name]
        return entry["model"], entry["tokenizer"], entry["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Config load kept for compatibility/debug (even if unused elsewhere)
    _ = BleurtConfig.from_pretrained(model_name)
    model = BleurtForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = BleurtTokenizer.from_pretrained(model_name)
    _BLEURT_CACHE[model_name] = {"model": model, "tokenizer": tokenizer, "device": device}
    return model, tokenizer, device


def _normalize_torch_device(dev: Optional[Union[str, int, torch.device]]) -> torch.device:
    """
    Normalize a device spec into torch.device.
    Accepts: None | "cuda" | "cuda:1" | "cpu" | int cuda_index | torch.device
    """
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, int):
        return torch.device(f"cuda:{dev}") if torch.cuda.is_available() else torch.device("cpu")
    # string
    s = str(dev).strip()
    if s == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _get_judge_llm(
    model_name: str,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
    device: Optional[Union[str, int, torch.device]] = None,
):
    """Lazy-load and cache a judge LLM. Prefer Unsloth 4bit if available. Supports explicit device placement."""
    global _JUDGE_LLM_CACHE
    dev = _normalize_torch_device(device)
    cache_key = f"{model_name}::device={dev}"
    if cache_key in _JUDGE_LLM_CACHE:
        entry = _JUDGE_LLM_CACHE[cache_key]
        # Some Llama-family tokenizers ship without a pad token. We use padding=True in self-judge,
        # so ensure pad_token is set (safe default: eos).
        try:
            tok = entry.get("tokenizer")
            if tok is not None and getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
        except Exception:
            pass
        return entry["model"], entry["tokenizer"], entry["device"]

    if _UNSLOTH_AVAILABLE:
        # Unsloth loads on the "current" CUDA device; force it via context when possible.
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=bool(load_in_4bit),
                    fast_inference=True,
                    gpu_memory_utilization=0.8,
                )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=bool(load_in_4bit),
                fast_inference=True,
                gpu_memory_utilization=0.8,
            )
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            model.eval()
        # Best-effort: make sure the model is on the requested device.
        try:
            model = model.to(dev)
        except Exception:
            pass
        dev = next(model.parameters()).device
        # Ensure pad token exists for judge batching.
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        torch_dtype = torch.float16 if dev.type == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(dev)
        model.eval()

    _JUDGE_LLM_CACHE[cache_key] = {"model": model, "tokenizer": tokenizer, "device": dev}
    return model, tokenizer, dev


def _parse_score_0_1(text: str) -> float:
    """Parse a float score in [0,1] from model output."""
    if text is None:
        return 0.0
    s = str(text).strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return 0.0
    try:
        v = float(m.group(1))
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, v)))


def _build_self_judge_prompt(prompt: str, completion: str, reference: Optional[str] = None, target_lang: Optional[str] = None) -> str:
    prompt = "" if prompt is None else str(prompt)
    completion = "" if completion is None else str(completion)
    reference = "" if reference is None else str(reference)
    
    # Language check instruction based on target_lang
    lang_instruction = ""
    evaluation_criteria = ""
    
    if target_lang:
        lang_map = {
            "ja": "Japanese (日本語)",
            "zh": "Chinese (中文)",
            "ko": "Korean (한국어)",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "ru": "Russian",
            "ar": "Arabic",
        }
        # Extract target language from lang pairs like "en-ja", "en-zh", etc.
        target = target_lang.split("-")[-1] if "-" in target_lang else target_lang
        if target in lang_map:
            lang_instruction = (
                f"CRITICAL REQUIREMENT: The candidate MUST be entirely in {lang_map[target]}.\n"
                f"- If ANY part of the candidate is in English or other languages, return 0.0 immediately.\n"
                f"- Mixed language outputs are unacceptable and must receive 0.0.\n"
                f"- Untranslated text, code-switching, or wrong language = 0.0\n\n"
            )
            evaluation_criteria = (
                "Evaluate the translation quality based on:\n"
                "1. Accuracy: Does it convey the original meaning correctly? (0.4 weight)\n"
                "2. Fluency: Is it natural and grammatically correct? (0.3 weight)\n"
                "3. Completeness: Is all information translated? (0.2 weight)\n"
                "4. Style: Is the tone and register appropriate? (0.1 weight)\n\n"
                "Scoring guidelines:\n"
                "- 0.9-1.0: Perfect or near-perfect translation\n"
                "- 0.7-0.8: Good translation with minor issues\n"
                "- 0.5-0.6: Acceptable but with noticeable problems\n"
                "- 0.3-0.4: Poor translation with major errors\n"
                "- 0.1-0.2: Very poor, barely understandable\n"
                "- 0.0: Wrong language, untranslated, or complete failure\n\n"
            )
    
    if reference.strip():
        return (
            "You are an expert translation evaluator. Your task is to assess translation quality.\n\n"
            f"{lang_instruction}"
            f"{evaluation_criteria}"
            "Compare the candidate translation against the reference translation.\n"
            "Be strict but fair. Return ONLY a single decimal number in [0,1].\n\n"
            f"[SOURCE TEXT]\n{prompt}\n\n"
            f"[REFERENCE TRANSLATION]\n{reference}\n\n"
            f"[CANDIDATE TRANSLATION]\n{completion}\n\n"
            "Your score (0.0-1.0):"
        )
    default_criteria = "Evaluate based on accuracy, fluency, completeness, and appropriateness.\n\n"
    criteria_to_use = evaluation_criteria if evaluation_criteria else default_criteria
    return (
        "You are an expert translation evaluator. Your task is to assess translation quality.\n\n"
        f"{lang_instruction}"
        f"{criteria_to_use}"
        "Be strict and consider whether the translation accurately conveys the intended meaning.\n"
        "Return ONLY a single decimal number in [0,1].\n\n"
        f"[SOURCE TEXT]\n{prompt}\n\n"
        f"[CANDIDATE TRANSLATION]\n{completion}\n\n"
        "Your score (0.0-1.0):"
    )

def _get_comet_model():
    """Lazy-load and cache the COMET model to avoid repeated checkpoint loads."""
    global _COMET_MODEL
    if _COMET_MODEL is None:
        _COMET_MODEL = load_from_checkpoint(download_model(_COMET_MODEL_NAME))
    return _COMET_MODEL

def _extract_src_from_prompt(prompt: str) -> str:
    """Extract source (English) text once from prompt if present, else return prompt."""
    m = _RE_SRC_FROM_PROMPT.search(prompt)
    return m.group(1) if m else prompt


def _is_grouped_list(x) -> bool:
    """True if x looks like List[List[...]] (grouped by prompt)."""
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], (list, tuple))


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _get_external_judge_session():
    """Create a singleton requests.Session for external judge calls."""
    global _EXTERNAL_JUDGE_SESSION
    if _EXTERNAL_JUDGE_SESSION is None:
        if requests is None:
            raise RuntimeError("requests is required for external_judge reward but is not installed.")
        _EXTERNAL_JUDGE_SESSION = requests.Session()
    return _EXTERNAL_JUDGE_SESSION


def _external_judge_cache_get(cache_path: Optional[str], key: str) -> Optional[str]:
    """Get cached raw judge response by key from memory or sqlite cache."""
    with _EXTERNAL_JUDGE_LOCK:
        if key in _EXTERNAL_JUDGE_MEM_CACHE:
            return _EXTERNAL_JUDGE_MEM_CACHE[key]
    if not cache_path:
        return None
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        con = sqlite3.connect(cache_path)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
            row = con.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            con.close()
    except Exception:
        return None


def _external_judge_cache_set(cache_path: Optional[str], key: str, value: str) -> None:
    """Set cached raw judge response by key to memory and (optionally) sqlite cache."""
    with _EXTERNAL_JUDGE_LOCK:
        if len(_EXTERNAL_JUDGE_MEM_CACHE) >= _EXTERNAL_JUDGE_MEM_CACHE_MAX:
            # crude eviction: drop ~10% oldest insertion order isn't tracked; just clear
            _EXTERNAL_JUDGE_MEM_CACHE.clear()
        _EXTERNAL_JUDGE_MEM_CACHE[key] = value
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        con = sqlite3.connect(cache_path)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
            con.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (key, value))
            con.commit()
        finally:
            con.close()
    except Exception:
        return


def _call_external_judge_chat_completion(
    prompt: str,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
    sleep_sec: float = 0.0,
    temperature: float = 0.0,
    extra_payload: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> str:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint and return choices[0].message.content.
    Includes basic retry/backoff for 429/5xx.
    """
    if not endpoint:
        raise ValueError("external_judge requires judge_endpoint (or CYCLOUD_GENAI_ENDPOINT).")
    if not api_key:
        raise ValueError("external_judge requires an API key (or CYCLOUD_GENAI_API_KEY).")
    if not model:
        raise ValueError("external_judge requires judge_model (or CYCLOUD_JUDGE_MODEL).")

    if sleep_sec and sleep_sec > 0:
        time.sleep(float(sleep_sec))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    if extra_payload:
        payload.update(extra_payload)

    sess = _get_external_judge_session()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = sess.post(endpoint, headers=headers, json=payload, timeout=timeout)
            # retry on rate limit / transient server errors
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"external judge HTTP {r.status_code}: {r.text[:500]}")
                backoff = min(30.0, (2.0 ** attempt))
                time.sleep(backoff)
                continue
            r.raise_for_status()
            data = r.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:
            last_err = e
            backoff = min(30.0, (2.0 ** attempt))
            time.sleep(backoff)
            continue
    raise RuntimeError(f"external judge call failed after retries: {last_err}")


def parse_match_score_judge_output(text: str) -> Dict[str, Any]:
    """
    Parse judge output with markers:
      - [[MATCH]] / [[MISMATCH]]
      - Score: 1-5
      - Reason: ...
    """
    if "[[MATCH]]" in (text or ""):
        match = True
    elif "[[MISMATCH]]" in (text or ""):
        match = False
    else:
        match = None

    m = re.search(r"Score:\s*([1-5])\b", text or "")
    score = int(m.group(1)) if m else None

    reason_match = re.search(r"Reason:\s*(.+)$", text or "", flags=re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""

    return {"label_match": match, "quality_score": score, "judge_raw": text or "", "judge_reason": reason}
class RewardRegistry: 
    # Use a writable location for the default TensorBoard log dir
    _default_log_dir = os.environ.get(
        "MT_GRPO_LOGDIR", 
        os.path.join(os.path.expanduser("~"), "mt_grpo_results")
    )
    os.makedirs(_default_log_dir, exist_ok=True)
    _writer = SummaryWriter(log_dir=os.path.join(_default_log_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
    _rewards = {}
    _writer = None      
    _steps_count = 0
    _defaults: Dict[str, Any] = {}
    @classmethod
    def set_writer(cls, writer):

        cls._writer = writer

    @classmethod
    def set_defaults(cls, **kwargs):
        """Set default kwargs used by reward functions when caller does not pass them."""
        cls._defaults.update(kwargs or {})
    
    @classmethod
    def _log_to_tensorboard(cls, tag, value):

        if cls._writer is not None:
            cls._writer.add_scalar(tag, value)
    @classmethod
    def register(cls, name):
        """Decorator to register a reward function."""
        def decorator(func):
            cls._rewards[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name):
        """Get a reward function by name."""
        if name not in cls._rewards:
            raise ValueError(f"Reward function '{name}' not found")
        return cls._rewards[name]
    
    @classmethod
    def list_rewards(cls):
        """List all registered reward functions."""
        return list(cls._rewards.keys())
    
    @classmethod
    def steps_count(cls):
        return cls._steps_count

_BLEU_DE = BLEU(
    tokenize='13a',        
    effective_order=True,
)


@RewardRegistry.register("bleu")
def bleu_de_reward(completions: list[str],
                   ground_truth: list[str],
                   **kwargs) -> list[float]:
    """
    MBR-style BLEU reward: For each completion, compute BLEU score against all other completions
    and return the mean as the reward for that completion.
    """
    n_completions = len(completions)
    mbr_scores = []
    
    for i, target_completion in enumerate(completions):
        total_score = 0.0
        
        for j, candidate_completion in enumerate(completions):
            bleu_score = _BLEU_DE.sentence_score(candidate_completion, [target_completion]).score / 100.0
            total_score += bleu_score
        
        avg_score = total_score / (n_completions) if n_completions > 1 else 0.0
        mbr_scores.append(avg_score)
    
    return mbr_scores

@RewardRegistry.register("bleurt")
def bleurt_reward(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    MBR-style BLEURT reward: For each completion, compute BLEURT score against all other completions
    and return the sum (or mean) as the reward for that completion.
    """
    model_name = kwargs.get('bleurt_model', "lucadiliello/BLEURT-20-D12")
    model, tokenizer, device = _get_bleurt_components(model_name)
    
    n_completions = len(completions)
    mbr_scores = []
    
    batch_size = 32
    # import ipdb; ipdb.set_trace()
    
    # For each completion, calculate its BLEURT score against all other completions
    for i, target_completion in enumerate(completions):
        total_score = 0.0
        
        # Compare target_completion with all other completions
        for j in range(0, n_completions, batch_size):
            batch_end = min(j + batch_size, n_completions)
            batch_candidates = completions[j:batch_end]
            
            # Create batch with target_completion as reference for all candidates
            batch_references = [target_completion] * len(batch_candidates)
            
            inputs = tokenizer(
                batch_references,
                batch_candidates,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_scores = outputs.logits.squeeze(-1)
                
                if batch_scores.dim() == 0:
                    batch_scores = batch_scores.unsqueeze(0)
                
                # Normalize scores to [0, 1] range
                normalized_batch_scores = (batch_scores + 1) / 2
                total_score += normalized_batch_scores.sum().item()
        
        # Average score across all comparisons (optional: could use sum instead)
        avg_score = total_score / n_completions
        mbr_scores.append(avg_score)
    
    return mbr_scores



@RewardRegistry.register("random")
def random_reward(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Random reward: For each completion, return a random score between 0 and 1.
    """
    return [random.gauss(mu=0.0, sigma=1.0) for _ in completions]


@RewardRegistry.register("bleurt_eval")
def bleurt_eval_reward(completions: List[str], ground_truth: List[str] = None, **kwargs) -> List[float]:
    """
    Calculate BLEURT scores for completions against ground truth.
    
    Args:
        completions: List of generated completions
        ground_truth: List of reference texts (optional)
        
    Returns:
        List of normalized BLEURT scores (0-1 range)
    """
    
    
    # Handle case where ground_truth contains empty strings
    if all(not gt.strip() for gt in ground_truth):
        return [0.5] * len(completions)
    
    # Get model name from kwargs or use default
    model_name = kwargs.get('bleurt_model', "lucadiliello/BLEURT-20-D12")
    model, tokenizer, device = _get_bleurt_components(model_name)
    
    scores = []
    
    # Process in batches to avoid memory issues
    batch_size = 8
    for i in range(0, len(completions), batch_size):
        batch_completions = completions[i:i+batch_size]
        batch_references = ground_truth[i:i+batch_size] if i+batch_size <= len(ground_truth) else ground_truth[i:]
        
        # Pad references if needed
        while len(batch_references) < len(batch_completions):
            batch_references.append(batch_references[-1] if batch_references else "")
        
        # Tokenize inputs with explicit max_length to prevent tensor size mismatch
        inputs = tokenizer(
            batch_references,
            batch_completions,
            padding=True,
            truncation=True,
            max_length=512,  # Explicitly set max length to match model's position embeddings
            return_tensors="pt"
        ).to(device)
        
        # Calculate scores
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1).tolist()
            
        # Handle single item case
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
            
        # BLEURT scores typically range from -1 to 1, normalize to 0-1
        normalized_scores = [(score + 1) / 2 for score in batch_scores]
        scores.extend(normalized_scores)
    
    return scores




@RewardRegistry.register("comet_eval")
def comet_eval_reward(completions: List[str], ground_truth: List[str] = None,prompts: List[str] = None, **kwargs) -> List[float]:
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    # scores = []
    if prompts is None:
        prompts = kwargs['prompts']
    data = []
    for prompt, completion, reference in zip(prompts,completions, ground_truth):
        match = re.search(r"(English text:.*?Japanese translation:)", prompt, re.S)
        if match:
            cleaned = re.sub(r"English text:\s*", "", match.group(1))
            cleaned = re.sub(r"\n\nJapanese translation:\s*", "", cleaned)
        else:
            cleaned = prompt
        
        d_ = {}
        d_["src"] = cleaned
        d_["mt"] = completion
        d_["ref"] = reference
        data.append(d_)
        # scores.append(model.predict(data).scores[0])
    scores = model.predict(data,batch_size=8).scores
    # import ipdb; ipdb.set_trace()
    return scores


@RewardRegistry.register("self_judge_eval")
def self_judge_eval_reward(completions: List[str], ground_truth: List[str] = None, prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Self-judge reward against ground truth (if provided).
    The judge LLM returns a numeric score in [0,1] per completion.
    """
    judge_model = kwargs.get("judge_model") or RewardRegistry._defaults.get("judge_model")
    if not judge_model:
        raise ValueError(
            "self_judge_eval requires judge_model. "
            "Set RewardRegistry.set_defaults(judge_model=...) or pass judge_model=..."
        )

    max_seq_length = int(kwargs.get("judge_max_seq_length") or RewardRegistry._defaults.get("judge_max_seq_length") or 1024)
    load_in_4bit = bool(
        kwargs["judge_load_in_4bit"] if "judge_load_in_4bit" in kwargs else RewardRegistry._defaults.get("judge_load_in_4bit", True)
    )
    judge_device = (
        kwargs.get("judge_device")
        or RewardRegistry._defaults.get("judge_device")
        or os.environ.get("JUDGE_DEVICE")
    )
    batch_size = int(kwargs.get("judge_batch_size") or RewardRegistry._defaults.get("judge_batch_size") or 8)
    max_new_tokens = int(kwargs.get("judge_max_new_tokens") or RewardRegistry._defaults.get("judge_max_new_tokens") or 8)

    model, tok, device = _get_judge_llm(
        judge_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        device=judge_device,
    )

    n = len(completions)
    if prompts is None:
        prompts = kwargs.get("prompts") or RewardRegistry._defaults.get("prompts") or [""] * n
    
    ground_truth = [""] * n
    
    # Get target language for language checking
    target_lang = kwargs.get("target_lang") or RewardRegistry._defaults.get("target_lang")

    # pad to length
    if len(prompts) < n:
        prompts = list(prompts) + [""] * (n - len(prompts))
    if len(ground_truth) < n:
        ground_truth = list(ground_truth) + [""] * (n - len(ground_truth))

    # Extract source text from prompts (prompts contain full chat template formatting)
    # _build_self_judge_prompt expects the raw source text, not the formatted prompt
    src_list = [_extract_src_from_prompt(p) for p in prompts]
    
    judge_inputs = [
        _build_self_judge_prompt(src_list[i], completions[i], ground_truth[i], target_lang=target_lang)
        for i in range(n)
    ]

    scores: List[float] = []
    for i in range(0, n, batch_size):
        batch = judge_inputs[i : i + batch_size]
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        gen = out[:, inputs["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        scores.extend([_parse_score_0_1(t) for t in texts])
        
    return scores


@RewardRegistry.register("self_judge")
def self_judge_reward(completions: List[str], ground_truth: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Self-judge reward WITHOUT reference.
    Judge LLM evaluates completions based on prompt only, ignoring ground_truth.
    """
    # Force ground_truth to empty to avoid using reference
    empty_ground_truth = [""] * len(completions)
    return self_judge_eval_reward(completions, empty_ground_truth, prompts=prompts, **kwargs)






