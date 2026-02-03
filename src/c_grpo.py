import argparse
import json
import logging
import os
from datetime import datetime

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel, FastVisionModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from data.dataset_utils import (
    clean_summary_text,
    create_summarization_prompt,
    create_translation_prompt,
    load_datasets,
    prepare_dataset_for_grpo,
    split_dataset,
)
from reward_functions import RewardRegistry

PatchFastRL("GRPO", FastLanguageModel)
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
    parser = argparse.ArgumentParser(description="Train a model using GRPO for machine translation or summarization")
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
        help="Task to perform: 'mt' (machine translation), 'summary' (summarization), 'mcq' (multiple-choice QA), 'math' (mathematical reasoning), or 'image_caption' (image captioning)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of generations for GRPO"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=150,
        help="Save checkpoint every X steps"
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
        "--bleurt_mix_beta",
        type=float,
        default=0.0,
        help="Mixing coefficient beta in [0,1] for bleurt_eval_mix: mixed=(1-beta)*bleurt_eval + beta*bleurt(MBR)"
    )
    parser.add_argument(
        "--train_device",
        type=int,
        default=0,
        help="CUDA device index for training model (used to select the active device for Unsloth/FastLanguageModel).",
    )
    parser.add_argument(
        "--judge_device",
        type=int,
        default=1,
        help="CUDA device index for judge model (used by self_judge/self_judge_eval rewards).",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="grpo_ad",
        help="Project name for S3 storage"
    )
    return parser.parse_args()

def setup_model_and_tokenizer(args):
    logger.info(f"Loading model: {args.model_name}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will be slow.")
    
    try:
        # Check if this is a VLM task
        if args.task == "image_caption":
            model, tokenizer = FastVisionModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = args.max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = False, # Enable vLLM fast inference
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)
            model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers

    r = args.lora_rank,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = args.lora_rank,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = args.seed,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)
            return model, tokenizer
        else:
            # Standard LLM loading
            logger.info("Initializing model and tokenizer with FastLanguageModel")
            if args.model_name == "Qwen/Qwen3-8B":
                model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                load_in_4bit=True,
                fast_inference=True,
                max_lora_rank=args.lora_rank,
                gpu_memory_utilization=0.8,
                # enable_thinking = False
            )
            else:
                model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                load_in_4bit=True,
                fast_inference=True,
                max_lora_rank=args.lora_rank,
                gpu_memory_utilization=0.8)
            logger.info(f"Setting up LoRA with rank {args.lora_rank}")
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_rank,
                target_modules=[
                    "q_proj", "k_proj", 
                ],
                lora_alpha=args.lora_rank,
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            
            logger.info("Model and tokenizer setup completed successfully")
            return model, tokenizer
    except Exception as e:
        logger.error(f"Error setting up model and tokenizer: {str(e)}", exc_info=True)
        raise

def train_with_grpo(model, tokenizer, train_dataset, val_dataset, args):

    logger.info("Setting up GRPO training")
    
    reward_function_names = args.reward_functions.split(',')
    reward_funcs = []
    
    for name in reward_function_names:
        try:
            reward_func = RewardRegistry.get(name.strip())
            reward_funcs.append(reward_func)
            logger.info(f"Using reward function: {name}")
        except ValueError as e:
            logger.warning(f"Warning: {e}")
    
    if not reward_funcs:
        logger.warning("No valid reward functions specified. Using bleurt reward.")
        reward_funcs = [RewardRegistry.get("bleurt")]
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    else:
        logger.info("No validation dataset provided")
    
    logger.info("Configuring GRPO training arguments")
    train_args = GRPOConfig(
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        temperature=args.temperature,
        max_prompt_length=args.max_seq_length,
        use_vllm=False if args.task == "image_caption" else True,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=0.1, 
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )
    
    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    args_path = os.path.join(args.output_dir, "args.json")
    logger.info(f"Saving arguments to {args_path}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Initializing GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        reward_funcs=reward_funcs,
        args=train_args,
    )
    
    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("GRPO training completed")
    
    return trainer
try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
except Exception:
    qwen_process_vision_info = None

def evaluate_model(model, tokenizer, test_dataset, args, timestamp: str = None):

    logger.info("Evaluating model on test dataset...")
    
    sources = test_dataset["source"] if "source" in test_dataset.column_names else []

    if args.task == "image_caption" and "caption_0" in test_dataset.column_names:
        references = test_dataset["caption_0"]
    else:
        references = test_dataset["reference"] if "reference" in test_dataset.column_names else []

    logger.info(f"Test dataset size: {len(sources) if sources else len(references)} examples")
    model = model.to("cuda")  

    
    if args.task == "mt":
        translations = []
        for i in tqdm(range(len(sources)), desc="Generating translations"):
            source = sources[i]
            prompt = create_translation_prompt(source, args)
            
            logger.debug(f"Creating prompt for example {i}")
            if args.model_name == "Qwen/Qwen3-8B":
                formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[-1]
            
            logger.debug(f"Generating translation for example {i}")
            with torch.no_grad():
           
                output_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_completion_length,
                        temperature=0.1,
                    )
            
            generated_ids = output_ids[0][input_length:]
            translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            translations.append(translation)
          
            
            if i < 3 or i % 100 == 0:
                logger.info(f"Example {i}:")
                logger.info(f"  Source: {source}")
                logger.info(f"  Reference: {references[i]}")
                logger.info(f"  Translation: {translation}")
        
        logger.info(f"Calculating BLEURT scores using model: {args.bleurt_model}...")
        bleurt_scores = RewardRegistry.get("bleurt_eval")(
            translations, 
            references, 
            bleurt_model=args.bleurt_model
        )
        comet_scores = RewardRegistry.get("comet_eval")(
                translations,
                references,
                sources
            )
        avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
        avg_comet = sum(comet_scores) / len(comet_scores)
        logger.info("Creating evaluation results...")
        results = {
            "model": args.model_name,
            "test_dataset": args.test_dataset,
            "metrics": {
                "bleurt": {
                    "average": avg_bleurt,
                    "scores": bleurt_scores
                },
                "comet": {
                    "average": avg_comet,
                    "scores": comet_scores
                },
            },
            "examples": [
                {
                    "source": sources[i],
                    "reference": references[i],
                    "translation": translations[i],
                    "bleurt": bleurt_scores[i],
                    "comet": comet_scores[i],
                }
                for i in range(len(sources)) 
            ]
        }
    elif args.task == "summary":
        summaries = []
        for i in tqdm(range(len(sources)), desc="Generating summaries"):
            source = sources[i]
            prompt = create_summarization_prompt(source, args)
            
            logger.debug(f"Creating prompt for example {i}")
            if args.model_name == "Qwen/Qwen3-8B":
                formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[-1]
            
            logger.debug(f"Generating summary for example {i}")
            with torch.no_grad():
                
                
                output_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_completion_length,
                        temperature=0.1,
                    )
            generated_ids = output_ids[0][input_length:]
            summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
            summary = clean_summary_text(summary)
            summaries.append(summary)
            
            if i < 3 or i % 100 == 0:
                logger.info(f"Example {i}:")
                logger.info(f"  Source: {source}")
                logger.info(f"  Reference: {references[i]}")
                logger.info(f"  Summary: {summary}")
        
        logger.info(f"Calculating BLEURT scores using model: {args.bleurt_model}...")
        bleurt_scores = RewardRegistry.get("bleurt_eval")(
            summaries, 
            references, 
            bleurt_model=args.bleurt_model
        )
        comet_scores = RewardRegistry.get("comet_eval")(
                summaries,
                references,
                sources
            )
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(
        references=references,
        predictions=summaries,
        use_stemmer=True,           
        tokenizer=lambda x: x.lower(), 
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )
        avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
        avg_comet = sum(comet_scores) / len(comet_scores)
        logger.info("Creating evaluation results...")
        results = {
            "model": args.model_name,
            "test_dataset": args.test_dataset,
            "metrics": {
                "bleurt": {
                    "average": avg_bleurt,
                    "scores": bleurt_scores
                },
                "comet": {
                    "average": avg_comet,
                    "scores": comet_scores
                },
                "rouge": {k: round(v, 4) for k, v in rouge_scores.items()}
            },
            "examples": [
                {
                    "source": sources[i],
                    "reference": references[i],
                    "summary": summaries[i],
                    "bleurt": bleurt_scores[i],
                    "comet": comet_scores[i],
                }
                for i in range(len(sources)) 
            ]
        }
    
    elif args.task == "mcq":
        logger.info("Evaluating MCQ with flexible matching (gold substring allow)...")
        import re
        def normalize_text(x: str) -> str:
            if x is None:
                return ""
            x = str(x).strip().lower()
            # remove spaces and common punctuation (incl. Japanese)
            x = re.sub(r"[\s\t\n\r]", "", x)
            x = re.sub(r"[\.\!\?\,\:\;\-\(\)\[\]\{\}\"'`""''、。・「」『』（）]", "", x)
            return x
        unknown_synonyms_norm = {
            normalize_text(s)
            for s in [
                "判断できない", "わからない", "不明", "情報がない", "情報はない", "情報はありません", "情報なし",
                "insufficient information", "cannot be determined", "unknown", "not enough information", "cannot decide"
            ]
        }
        prompts = test_dataset["prompt"] if "prompt" in test_dataset.column_names else []
        gold_texts = test_dataset["ground_truth"] if "ground_truth" in test_dataset.column_names else references
        all_choices = test_dataset["choices"] if "choices" in test_dataset.column_names else None
        generations = []
        correct = 0
        total = len(prompts)
        for i in tqdm(range(total), desc="Generating answers"):
            prompt = prompts[i]
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[-1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_completion_length,
                    temperature=0.1,
                )
            generated_ids = output_ids[0][input_length:]
            out_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(out_text)
            gold = str(gold_texts[i])
            gold_norm = normalize_text(gold)
            out_norm = normalize_text(out_text)
            # Flexible match: gold appears in output (after normalization)
            ok = False
            if gold_norm and gold_norm in out_norm:
                ok = True
            # Unknown-synonym handling when gold indicates undecidable/unknown
            if not ok and gold_norm in unknown_synonyms_norm and out_norm in unknown_synonyms_norm:
                ok = True
            # Fallback: if choices are present, and output equals exactly one choice, use that
            if not ok and all_choices is not None:
                choices_i = all_choices[i] or []
                chosen_exact = None
                for ch in choices_i:
                    if str(out_text).strip() == str(ch).strip():
                        chosen_exact = ch
                        break
                if chosen_exact is not None and str(chosen_exact).strip() == gold.strip():
                    ok = True
            if ok:
                correct += 1
            if i < 3 or i % 100 == 0:
                logger.info(f"Example {i}:")
                logger.info(f"  Gold: {gold}")
                logger.info(f"  Output: {out_text}")
                logger.info(f"  Correct: {ok}")
        accuracy = correct / total if total else 0.0
        results = {
            "model": args.model_name,
            "test_dataset": args.test_dataset,
            "metrics": {
                "accuracy": accuracy,
            },
            "examples": [
                {
                    "prompt": prompts[i],
                    "gold": str(gold_texts[i]),
                    "output": generations[i],
                    "correct": (
                        (lambda g, o: (g and g in o) or (g in unknown_synonyms_norm and o in unknown_synonyms_norm))(
                            normalize_text(gold_texts[i]),
                            normalize_text(generations[i])
                        )
                    )
                }
                for i in range(total)
            ],
        }
    else:
        logger.warning(f"Unknown task {args.task}; skipping evaluation.")
        results = {}
        
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mt_grpo_eval_{timestamp}.json"
    
    local_output_path = os.path.join(args.output_dir, filename)
    
    logger.info(f"Saving evaluation results to {local_output_path}")
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation results saved locally to {local_output_path}")
    if args.task == "mt" or args.task == "summary":
        logger.info(f"BLEURT: {avg_bleurt:.4f}")
        logger.info(f"COMET: {avg_comet:.4f}")
    elif args.task == "math":
        logger.info(f"Accuracy: {results.get('metrics',{}).get('accuracy',0.0):.4f}")
        logger.info(f"Correct: {results.get('metrics',{}).get('correct',0)}/{results.get('metrics',{}).get('total',0)}")
    elif args.task == "mcq":
        logger.info(f"Accuracy: {results.get('metrics',{}).get('accuracy',0.0):.4f}")
    elif args.task == "image_caption":
        logger.info(f"BLEU: {results.get('metrics',{}).get('bleu',{}).get('average',0.0):.4f}")
        logger.info(f"BLEURT: {results.get('metrics',{}).get('bleurt',{}).get('average',0.0):.4f}")
    # logger.info(f"COMET_23: {avg_comet_23:.4f}")
    return results


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, f"mt_grpo_{timestamp}.log")
    setup_logging(log_level=logging.INFO, log_file=log_file)

    # Select training CUDA device early (Unsloth typically uses the current CUDA device).
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(args.train_device))
        except Exception as e:
            logger.warning(f"Failed to set CUDA device to {args.train_device}: {e}")

    # Default context for reward functions (used by self-judge, etc.)
    RewardRegistry.set_defaults(
        judge_model=args.model_name,
        bleurt_model=args.bleurt_model,
        judge_max_seq_length=args.max_seq_length,
        bleurt_mix_beta=args.bleurt_mix_beta,
        judge_device=int(args.judge_device),
    )
    
    if args.task == "summary":
        task_name = "summarization"
    elif args.task == "math":
        task_name = "mathematical reasoning"
    elif args.task == "mcq":
        task_name = "multiple-choice QA"
    elif args.task == "image_caption":
        task_name = "image captioning"
    else:
        task_name = "machine translation"
    logger.info(f"Starting GRPO {task_name} training with timestamp {timestamp}")
    logger.info(f"Command line arguments: {args}")
    
    logger.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        logger.info(f"Loading datasets from {args.train_datasets} and {args.test_dataset}")
        train_paths = args.train_datasets.split(',')
        train_dataset, test_dataset = load_datasets(train_paths, args.test_dataset, task=args.task)
        
        model, tokenizer = setup_model_and_tokenizer(args)
        
        logger.info("Preparing datasets for GRPO")
        train_dataset = prepare_dataset_for_grpo(train_dataset, tokenizer, args)
        
        logger.info(f"Splitting dataset with validation size {args.val_size}")
        train_dataset, val_dataset = split_dataset(
            train_dataset, 
            test_size=args.val_size, 
            seed=args.seed
        )
        
        train_with_grpo(model, tokenizer, train_dataset, val_dataset, args)
        
        if test_dataset:
            logger.info("Preparing test dataset for evaluation")
            test_dataset = prepare_dataset_for_grpo(test_dataset, tokenizer, args)
            
            evaluate_model(model, tokenizer, test_dataset, args, timestamp)
        
        logger.info("Training and evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()