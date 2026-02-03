import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.dataset_utils import (
    create_summarization_prompt,
    create_translation_prompt,
    load_datasets,
    prepare_dataset_for_grpo,
)
from reward_functions import RewardRegistry

logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO, log_file=None):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized with level {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Logs will be saved to {log_file}")



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model with MBR-style decoding for text tasks "
            "(machine translation, summarization, math, MCQ)."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sbintuitions/sarashina2.2-3b-instruct-v0.1",
        help="Name or path of the model to use for the task"
    )
    parser.add_argument(
        "--train_datasets",
        type=str,
        default="dataset/wmt.en-ja/newstest2021.en-ja.all.xml,dataset/wmt.en-ja/wmttest2022.en-ja.all.xml,dataset/wmt.en-ja/wmttest2023.en-ja.all.xml",
        help="Comma-separated list of training dataset paths or Hugging Face datasets (e.g., 'EdinburghNLP/xsum:train,knkarthick/samsum:train')"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="dataset/wmt.en-ja/wmttest2024.en-ja.all.xml",
        help="Path to the test dataset or Hugging Face dataset (e.g., 'EdinburghNLP/xsum:test')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/pg/grpo/mt_grpo_results",
        help="Directory to save the model and results"
    )
    parser.add_argument(
        "--reward_functions",
        type=str,
        default="combined",
        help="Comma-separated list of reward functions to use"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mt",
        help=(
            "Task to perform: "
            "'mt' (machine translation), "
            "'summary' (summarization), "
            "'math' (mathematical reasoning), "
            "'mcq' (multiple-choice QA)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of generations for MBR"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=200,
        help="Maximum length for generated completions"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--bleurt_model",
        type=str,
        default="lucadiliello/BLEURT-20-D12",
        help="BLEURT model to use for evaluation"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="grpo_ad",
        help="Project name for S3 storage"
    )
    return parser.parse_args()

def setup_model_and_tokenizer(model_name: str):
    logger.info(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is not available. Falling back to CPU; this will be slow.")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def setup_models_and_tokenizers(args):
    """Single-model setup to keep API stable."""
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    return model, tokenizer


def evaluate_mbr_model(model, tokenizer, test_dataset, args, timestamp: str = None):
    """
    MBR 風のデコードで各タスクを評価する。
    - mt / summary: BLEURT + COMET
    - math: 正解率
    - mcq: 正解率（柔軟マッチ）
    """

    logger.info("Evaluating model on test dataset...")

    column_names = set(test_dataset.column_names)
    sources = test_dataset["source"] if "source" in column_names else None
    if "ground_truth" in column_names:
        references = test_dataset["ground_truth"]
    elif "reference" in column_names:
        references = test_dataset["reference"]
    else:
        references = None
    prompts = test_dataset["prompt"] if "prompt" in column_names else None

    num_examples = len(test_dataset)
    logger.info(f"Test dataset size: {num_examples} examples")

    device = next(model.parameters()).device

    log_bleurt: Optional[float] = None
    log_comet: Optional[float] = None
    log_accuracy: Optional[float] = None

    if args.task in ("mt", "summary"):
        generations: List[str] = []

        for i in tqdm(range(num_examples), desc=f"Generating for task={args.task}"):
            prompt = prompts[i] if prompts is not None else (
                create_translation_prompt(sources[i], args)
                if args.task == "mt"
                else create_summarization_prompt(sources[i], args)
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[-1]

            logger.debug(f"Generating outputs for example {i}")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=0.9 if args.task == "mt" else None,
                    temperature=args.temperature,
                    max_new_tokens=args.max_completion_length,
                    num_return_sequences=args.num_generations,
                )

            texts: List[str] = []
            for seq in output_ids:
                gen_ids = seq[input_length:]
                texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

            bleurt_mbr = RewardRegistry.get("bleurt")(
                texts,
                references if references is not None else [],
                bleurt_model=args.bleurt_model,
            )
            best_idx = int(np.argmax(bleurt_mbr))
            best_text = texts[best_idx]

            generations.append(best_text)

            if i < 3 or i % 100 == 0:
                logger.info(f"Example {i}:")
                if sources is not None:
                    logger.info(f"  Source: {sources[i]}")
                if references is not None and i < len(references):
                    logger.info(f"  Reference: {references[i]}")
                logger.info(f"  Output: {best_text}")

        metrics: Dict[str, Any] = {}

        if references is not None:
            logger.info(f"Calculating BLEURT scores using model: {args.bleurt_model}...")
            bleurt_scores = RewardRegistry.get("bleurt_eval")(
                generations,
                references,
                bleurt_model=args.bleurt_model,
            )
            avg_bleurt = float(sum(bleurt_scores) / len(bleurt_scores))
            metrics["bleurt"] = {
                "average": avg_bleurt,
                "scores": bleurt_scores,
            }
            log_bleurt = avg_bleurt
        else:
            bleurt_scores = [0.0] * len(generations)

        if sources is not None and references is not None:
            comet_scores = RewardRegistry.get("comet_eval")(
                generations,
                references,
                sources,
            )
            avg_comet = float(sum(comet_scores) / len(comet_scores))
            metrics["comet"] = {
                "average": avg_comet,
                "scores": comet_scores,
            }
            log_comet = avg_comet
        else:
            comet_scores = [0.0] * len(generations)

        # summary タスクでは ROUGE も計算
        if args.task == "summary" and references is not None:
            rouge_metric = evaluate.load("rouge")
            rouge_scores = rouge_metric.compute(
                references=references,
                predictions=generations,
                use_stemmer=True,
                tokenizer=lambda x: x.lower(),
                rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            )
            metrics["rouge"] = {k: round(v, 4) for k, v in rouge_scores.items()}

        examples = []
        for i in range(num_examples):
            ex: Dict[str, Any] = {
                "output": generations[i],
                "bleurt": bleurt_scores[i] if i < len(bleurt_scores) else None,
            }
            if sources is not None:
                ex["source"] = sources[i]
            if references is not None and i < len(references):
                ex["reference"] = references[i]
            if args.task == "mt":
                ex["translation"] = generations[i]
            else:
                ex["summary"] = generations[i]
            if "comet" in metrics and i < len(comet_scores):
                ex["comet"] = comet_scores[i]
            examples.append(ex)

        results = {
            "model": args.model_name,
            "task": args.task,
            "test_dataset": args.test_dataset,
            "metrics": metrics,
            "examples": examples,
        }

    

    elif args.task == "mcq":
        import re

        def normalize_text(x: str) -> str:
            if x is None:
                return ""
            x = str(x).strip().lower()
            x = re.sub(r"[\s\t\n\r]", "", x)
            x = re.sub(r"[\.\!\?\,\:\;\-\(\)\[\]\{\}\"'`""''、。・「」『』（）]", "", x)
            return x

        unknown_synonyms_norm = {
            normalize_text(s)
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

        gold_texts = references if references is not None else []
        all_choices = test_dataset["choices"] if "choices" in column_names else None

        generations: List[str] = []
        correct = 0

        for i in tqdm(range(num_examples), desc="Generating MCQ answers"):
            prompt = prompts[i] if prompts is not None else (sources[i] if sources is not None else "")

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[-1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_completion_length,
                    num_return_sequences=args.num_generations,
                )

            candidates: List[str] = []
            for seq in output_ids:
                gen_ids = seq[input_length:]
                candidates.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

            gold = str(gold_texts[i]) if i < len(gold_texts) else ""
            gold_norm = normalize_text(gold)

            if candidates:
                bleurt_mbr = RewardRegistry.get("bleurt")(
                    candidates,
                    references if references is not None else [],
                    bleurt_model=args.bleurt_model,
                )
                best_idx = int(np.argmax(bleurt_mbr)) if bleurt_mbr else 0
                best_text = candidates[best_idx]
            else:
                best_text = ""

            generations.append(best_text)

            out_norm = normalize_text(best_text)
            ok = False
            if gold_norm and gold_norm in out_norm:
                ok = True
            if (
                not ok
                and gold_norm in unknown_synonyms_norm
                and out_norm in unknown_synonyms_norm
            ):
                ok = True

            if not ok and all_choices is not None:
                choices_i = all_choices[i] or []
                chosen_exact = None
                for ch in choices_i:
                    if str(best_text).strip() == str(ch).strip():
                        chosen_exact = ch
                        break
                if chosen_exact is not None and str(chosen_exact).strip() == gold.strip():
                    ok = True

            if ok:
                correct += 1

            if i < 3 or i % 100 == 0:
                logger.info(f"Example {i}:")
                logger.info(f"  Gold: {gold}")
                logger.info(f"  Output(best): {best_text}")
                logger.info(f"  Correct: {ok}")

        total = len(generations)
        accuracy = float(correct / total) if total else 0.0

        examples = []
        for i in range(num_examples):
            ex: Dict[str, Any] = {
                "output": generations[i],
            }
            if sources is not None:
                ex["prompt"] = sources[i]
            if references is not None and i < len(references):
                ex["gold"] = str(references[i])
            if all_choices is not None:
                ex["choices"] = all_choices[i]

            if references is not None and i < len(references):
                gold = str(references[i])
                gold_norm = normalize_text(gold)
                out_norm = normalize_text(generations[i])
                ok = False
                if gold_norm and gold_norm in out_norm:
                    ok = True
                if (
                    not ok
                    and gold_norm in unknown_synonyms_norm
                    and out_norm in unknown_synonyms_norm
                ):
                    ok = True
                if not ok and all_choices is not None:
                    choices_i = all_choices[i] or []
                    chosen_exact = None
                    for ch in choices_i:
                        if str(generations[i]).strip() == str(ch).strip():
                            chosen_exact = ch
                            break
                    if chosen_exact is not None and str(chosen_exact).strip() == gold.strip():
                        ok = True
                ex["correct"] = ok
            examples.append(ex)

        metrics = {
            "accuracy": accuracy,
        }

        results = {
            "model": args.model_name,
            "task": args.task,
            "test_dataset": args.test_dataset,
            "metrics": metrics,
            "examples": examples,
        }

        log_accuracy = accuracy

    else:
        logger.warning(f"Unknown task {args.task}; skipping evaluation.")
        results = {
            "model": args.model_name,
            "task": args.task,
            "test_dataset": args.test_dataset,
            "metrics": {},
            "examples": [],
        }

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mt_mbr_eval_{timestamp}.json"

    local_output_path = os.path.join(args.output_dir, filename)

    logger.info(f"Saving evaluation results to {local_output_path}")
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation results saved locally to {local_output_path}")
    if log_bleurt is not None:
        logger.info(f"BLEURT: {log_bleurt:.4f}")
    if log_comet is not None:
        logger.info(f"COMET: {log_comet:.4f}")
    if log_accuracy is not None:
        logger.info(f"Accuracy: {log_accuracy:.4f}")

    return results


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, f"mt_mbr_{timestamp}.log")
    setup_logging(log_level=logging.INFO, log_file=log_file)
    
    task_name = "summarization" if args.task == "summary" else "machine translation"
    logger.info(f"Starting MBR {task_name} training with timestamp {timestamp}")
    logger.info(f"Command line arguments: {args}")
    
    logger.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        logger.info(f"Loading datasets from {args.train_datasets} and {args.test_dataset}")
        train_paths = args.train_datasets.split(',')
        # タスク種別に応じてローダを切り替える
        train_dataset, test_dataset = load_datasets(train_paths, args.test_dataset, task=args.task)
        
        model, tokenizer = setup_models_and_tokenizers(args)
        
        logger.info("Preparing datasets for MBR")
        train_dataset = prepare_dataset_for_grpo(train_dataset, tokenizer, args)
    
        
        
        if test_dataset:
            logger.info("Preparing test dataset for evaluation")
            test_dataset = prepare_dataset_for_grpo(test_dataset, tokenizer, args)
            
            evaluate_mbr_model(model, tokenizer, test_dataset, args, timestamp)
        
        logger.info("Training and evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()