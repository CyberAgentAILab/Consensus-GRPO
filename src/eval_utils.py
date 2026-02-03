import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import evaluate
import torch
from tqdm import tqdm

from reward_functions import RewardRegistry
from data.dataset_utils import clean_summary_text

logger = logging.getLogger(__name__)


def _get_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_text_mcq(x: str) -> str:
    import re

    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"[\s\t\n\r]", "", x)
    x = re.sub(r"[\.\!\?\,\:\;\-\(\)\[\]\{\}\"'`""''、。・「」『』（）]", "", x)
    return x


def evaluate_model_like_grpo(model, tokenizer, test_dataset, args, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate similarly to mt_grpo.py and save a JSON file to args.output_dir.
    Supported tasks: mt, summary, mcq, math
    Expects test_dataset prepared via prepare_dataset_for_grpo (has prompt/ground_truth/source etc).
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    task = getattr(args, "task", "mt")
    output_dir = getattr(args, "output_dir", ".")
    bleurt_model = getattr(args, "bleurt_model", "lucadiliello/BLEURT-20-D12")
    max_new = int(getattr(args, "max_completion_length", 200))

    os.makedirs(output_dir, exist_ok=True)
    device = _get_device(model)
    try:
        model = model.to(device)
    except Exception:
        pass

    prompts = test_dataset["prompt"] if "prompt" in test_dataset.column_names else []
    sources = test_dataset["source"] if "source" in test_dataset.column_names else []
    references = (
        test_dataset["ground_truth"]
        if "ground_truth" in test_dataset.column_names
        else (test_dataset["reference"] if "reference" in test_dataset.column_names else [])
    )

    logger.info(f"Evaluating task={task} on {len(test_dataset)} examples...")

    results: Dict[str, Any] = {
        "model": getattr(args, "model_name", ""),
        "task": task,
        "test_dataset": getattr(args, "test_dataset", ""),
        "metrics": {},
        "examples": [],
    }

    if task in ("mt", "summary"):
        generations: List[str] = []
        for i in tqdm(range(len(prompts)), desc=f"Generating ({task})"):
            prompt = prompts[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            in_len = inputs.input_ids.shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.1,
                    max_new_tokens=max_new,
                )
            gen_ids = out[0][in_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if task == "summary":
                text = clean_summary_text(text)
            generations.append(text)

            if i < 3 or i % 100 == 0:
                logger.info(f"Example {i}:")
                if sources:
                    logger.info(f"  Source: {sources[i]}")
                if references and i < len(references):
                    logger.info(f"  Reference: {references[i]}")
                logger.info(f"  Output: {text}")

        if references:
            bleurt_scores = RewardRegistry.get("bleurt_eval")(generations, references, bleurt_model=bleurt_model)
            avg_bleurt = float(sum(bleurt_scores) / len(bleurt_scores))
            results["metrics"]["bleurt"] = {"average": avg_bleurt, "scores": bleurt_scores}

            if sources:
                comet_scores = RewardRegistry.get("comet_eval")(generations, references, sources)
                avg_comet = float(sum(comet_scores) / len(comet_scores))
                results["metrics"]["comet"] = {"average": avg_comet, "scores": comet_scores}

            if task == "summary":
                rouge = evaluate.load("rouge")
                rouge_scores = rouge.compute(
                    references=references,
                    predictions=generations,
                    use_stemmer=True,
                    tokenizer=lambda x: x.lower(),
                    rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
                )
                results["metrics"]["rouge"] = {k: round(v, 4) for k, v in rouge_scores.items()}

        for i in range(len(generations)):
            ex: Dict[str, Any] = {"output": generations[i]}
            if sources and i < len(sources):
                ex["source"] = sources[i]
            if references and i < len(references):
                ex["reference"] = references[i]
            results["examples"].append(ex)


    elif task == "mcq":
        unknown_synonyms_norm = {
            _normalize_text_mcq(s)
            for s in [
                "判断できない",
                "わからない",
                "不明",
                "情報がない",
                "情報はない",
                "情報はありません",
                "情報なし",
                "insufficient information",
                "cannot be determined",
                "unknown",
                "not enough information",
                "cannot decide",
            ]
        }

        generations: List[str] = []
        correct = 0
        total = len(prompts)

        for i in tqdm(range(total), desc="Generating (mcq)"):
            prompt = prompts[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            in_len = inputs.input_ids.shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.1,
                    max_new_tokens=max_new,
                )
            gen_ids = out[0][in_len:]
            out_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            generations.append(out_text)

            gold = str(references[i]) if references and i < len(references) else ""
            gold_norm = _normalize_text_mcq(gold)
            out_norm = _normalize_text_mcq(out_text)

            ok = False
            if gold_norm and gold_norm in out_norm:
                ok = True
            if not ok and gold_norm in unknown_synonyms_norm and out_norm in unknown_synonyms_norm:
                ok = True
            if ok:
                correct += 1

        accuracy = float(correct / total) if total else 0.0
        results["metrics"]["accuracy"] = accuracy
        for i in range(len(generations)):
            ex: Dict[str, Any] = {"output": generations[i]}
            if references and i < len(references):
                ex["gold"] = str(references[i])
            results["examples"].append(ex)

    else:
        logger.warning(f"Unknown task {task}; skipping evaluation.")

    filename = f"{task}_eval_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved evaluation results to {path}")
    return results


