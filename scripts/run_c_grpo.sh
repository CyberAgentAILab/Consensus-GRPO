#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_DATASETS="dataset/wmt.en-ja/newstest2021.en-ja.all.xml,dataset/wmt.en-ja/wmttest2022.en-ja.all.xml,dataset/wmt.en-ja/wmttest2023.en-ja.all.xml"
TEST_DATASET="dataset/wmt.en-ja/wmttest2024.en-ja.all.xml"
# TRAIN_DATASETS="EdinburghNLP/xsum:train"
# TEST_DATASET="EdinburghNLP/xsum:test"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"


OUTPUT_DIR=""
REWARD_FUNCTIONS="" 
LORA_RANK=64

BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
NUM_GENERATIONS=32
LEARNING_RATE=2e-6
SAVE_STEPS=1000
TEMPERATURE=0.7
VAL_SIZE=0.2
SEED=42
BLEURT_MODEL="lucadiliello/BLEURT-20-D12"
BLEURT_MIX_BETA=0.1
PROJECT_NAME="grpo"
TASK="mt"
TRAIN_DEVICE="${TRAIN_DEVICE:-0}"
JUDGE_DEVICE="${JUDGE_DEVICE:-1}"

# Set parameters based on task type
if [[ "$TASK" == "summary" ]]; then
  MAX_SEQ_LENGTH=2500
  MAX_COMPLETION_LENGTH=300
  NUM_TRAIN_EPOCHS=1
  NUM_GENERATIONS=16
  # Set default reward function if not specified
  if [[ -z "$REWARD_FUNCTIONS" ]]; then
    REWARD_FUNCTIONS="bleurt"
  fi
elif [[ "$TASK" == "mcq" ]]; then
  MAX_SEQ_LENGTH=1000
  MAX_COMPLETION_LENGTH=300
  NUM_TRAIN_EPOCHS=1
  NUM_GENERATIONS=16
  # Set default reward function if not specified
  if [[ -z "$REWARD_FUNCTIONS" ]]; then
    REWARD_FUNCTIONS="bleurt"
  fi
else
  MAX_SEQ_LENGTH=1000
  MAX_COMPLETION_LENGTH=200
  NUM_TRAIN_EPOCHS=2
  NUM_GENERATIONS=32   
  # Set default reward function if not specified
  if [[ -z "$REWARD_FUNCTIONS" ]]; then
      REWARD_FUNCTIONS="bleurt"
  fi
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --train_datasets)
      TRAIN_DATASETS="$2"
      shift 2
      ;;
    --test_dataset)
      TEST_DATASET="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --reward_functions)
      REWARD_FUNCTIONS="$2"
      shift 2
      ;;
    --max_seq_length)
      MAX_SEQ_LENGTH="$2"
      shift 2
      ;;
    --lora_rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gradient_accumulation_steps)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --num_generations)
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --num_train_epochs)
      NUM_TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max_completion_length)
      MAX_COMPLETION_LENGTH="$2"
      shift 2
      ;;
    --val_size)
      VAL_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --bleurt_model)
      BLEURT_MODEL="$2"
      shift 2
      ;;
    --bleurt_mix_beta)
      BLEURT_MIX_BETA="$2"
      shift 2
      ;;
    --project_name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --train_device)
      TRAIN_DEVICE="$2"
      shift 2
      ;;
    --judge_device)
      JUDGE_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to extract language codes from dataset paths
extract_languages() {
  local datasets="$1"
  local languages=""
  for lang in ja zh de ru cs uk; do
    if [[ "$datasets" == *"en-${lang}"* ]]; then
      if [[ -z "$languages" ]]; then
        languages="$lang"
      else
        languages="$languages $lang"
      fi
    fi
  done
  echo "$languages"
}

# Auto-generate OUTPUT_DIR if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
  MODEL_SHORT=$(basename "$MODEL_NAME")
  MODEL_CLEAN=$(echo "$MODEL_SHORT" | sed 's/[^a-zA-Z0-9.-]/_/g')
  if [[ "$TASK" == "summary" ]]; then
    if [[ "$TRAIN_DATASETS" == *"xsum"* ]] && [[ "$TRAIN_DATASETS" == *"samsum"* ]]; then
      LANG_SUFFIX="_xsum_samsum"
    elif [[ "$TRAIN_DATASETS" == *"xsum"* ]]; then
      LANG_SUFFIX="_xsum"
    elif [[ "$TRAIN_DATASETS" == *"samsum"* ]]; then
      LANG_SUFFIX="_samsum"
    else
      LANG_SUFFIX="_summary"
    fi
  elif [[ "$TASK" == "mcq" ]]; then
    LANG_SUFFIX="_mcq"
  else
    DETECTED_LANGS=$(extract_languages "$TRAIN_DATASETS $TEST_DATASET")
    LANG_SUFFIX=""
    if [[ -n "$DETECTED_LANGS" ]]; then
      LANG_SUFFIX="_$(echo "$DETECTED_LANGS" | tr ' ' '_')"
    fi
  fi
  if [[ "$TASK" == "summary" ]]; then
    OUTPUT_DIR="summary_grpo_${MODEL_CLEAN}${LANG_SUFFIX}_gen${NUM_GENERATIONS}"
  elif [[ "$TASK" == "mcq" ]]; then
    OUTPUT_DIR="mcq_grpo_${MODEL_CLEAN}${LANG_SUFFIX}_gen${NUM_GENERATIONS}"
  elif [[ "$TASK" == "image_caption" ]]; then
    OUTPUT_DIR="image_caption_grpo_${MODEL_CLEAN}_gen${NUM_GENERATIONS}"
  elif [[ "$TASK" == "math" ]]; then
    OUTPUT_DIR="math_grpo_${MODEL_CLEAN}_gen${NUM_GENERATIONS}"
  else
    OUTPUT_DIR="mt_grpo_${MODEL_CLEAN}${LANG_SUFFIX}_gen${NUM_GENERATIONS}"
  fi
  echo "📁 Auto-generated output directory: $OUTPUT_DIR"
fi

# Print the parameters
if [[ "$TASK" == "summary" ]]; then
  echo "Running GRPO summarization training with the following parameters:"
elif [[ "$TASK" == "mcq" ]]; then
  echo "Running GRPO MCQ training with the following parameters:"
elif [[ "$TASK" == "math" ]]; then
  echo "Running GRPO mathematical reasoning training with the following parameters:"
elif [[ "$TASK" == "image_caption" ]]; then
  echo "Running GRPO image caption training with the following parameters:"
else
  echo "Running GRPO machine translation training with the following parameters:"
fi
echo "  Model name: $MODEL_NAME"
echo "  Training datasets: $TRAIN_DATASETS"
echo "  Test dataset: $TEST_DATASET"
echo "  Output directory: $OUTPUT_DIR"
echo "  Reward functions: $REWARD_FUNCTIONS"
echo "  Max sequence length: $MAX_SEQ_LENGTH"
echo "  LoRA rank: $LORA_RANK"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Number of generations: $NUM_GENERATIONS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Number of training epochs: $NUM_TRAIN_EPOCHS"
echo "  Save steps: $SAVE_STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  Max completion length: $MAX_COMPLETION_LENGTH"
echo "  Validation size: $VAL_SIZE"
echo "  Random seed: $SEED"
echo "  BLEURT model: $BLEURT_MODEL"
echo "  Project name: $PROJECT_NAME"
echo "  Task: $TASK"
echo "  Train device: $TRAIN_DEVICE"
echo "  Judge device: $JUDGE_DEVICE"

# Function to extract language codes from dataset paths
extract_languages() {
  local datasets="$1"
  local languages=""
  for lang in ja zh de ru cs uk; do
    if [[ "$datasets" == *"en-${lang}"* ]]; then
      if [[ -z "$languages" ]]; then
        languages="$lang"
      else
        languages="$languages $lang"
      fi
    fi
  done
  echo "$languages"
}

# Check if we need to download datasets for machine translation
if [[ "$TASK" == "mt" ]]; then
  echo ""
  echo "🔍 Detecting required languages from dataset paths..."
  REQUIRED_LANGS=$(extract_languages "$TRAIN_DATASETS $TEST_DATASET")
  if [[ -n "$REQUIRED_LANGS" ]]; then
    echo "📋 Required languages: $REQUIRED_LANGS"
    for lang in $REQUIRED_LANGS; do
      dataset_dir="dataset/wmt.en-${lang}"
      echo ""
      echo "📂 Checking for $lang datasets in $dataset_dir..."
      if [[ ! -d "$dataset_dir" ]] || [[ -z "$(ls -A "$dataset_dir" 2>/dev/null)" ]]; then
        echo "📥 $lang datasets not found. Downloading..."
        bash "$ROOT_DIR/data/get_wmt.sh" "$lang" "$dataset_dir"
      else
        echo "✅ $lang datasets already exist."
      fi
    done
  else
    echo "⚠️  No supported language detected in dataset paths."
    echo "   Supported patterns: en-ja, en-zh, en-de, en-ru, en-cs, en-uk"
    exit 1
  fi
elif [[ "$TASK" == "summary" ]]; then
  echo ""
  echo "📝 Summarization task detected - using Hugging Face datasets"
  echo "📋 Training datasets: $TRAIN_DATASETS"
  echo "📋 Test dataset: $TEST_DATASET"
elif [[ "$TASK" == "mcq" ]]; then
  echo ""
  echo "🧩 MCQ task detected - using local JSONL datasets"
  echo "📋 Training dataset: $TRAIN_DATASETS"
  echo "📋 Test dataset: $TEST_DATASET"
elif [[ "$TASK" == "math" ]]; then
  echo ""
  echo "🔢 Math reasoning task detected - using Hugging Face datasets"
  echo "📋 Training datasets: $TRAIN_DATASETS"
  echo "📋 Test dataset: $TEST_DATASET"
elif [[ "$TASK" == "image_caption" ]]; then
  echo ""
  echo "🖼️  Image caption task detected - using Hugging Face datasets"
  echo "📋 Training datasets: $TRAIN_DATASETS"
  echo "📋 Test dataset: $TEST_DATASET"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Start resource logger in the background
echo "Starting resource logger..."
python3 "$ROOT_DIR/data/resource_logger.py" --log_file "${OUTPUT_DIR}/resource_log.txt" --interval 1 --disk_paths / /pg --format csv --quiet &
LOGGER_PID=$!

# Set PYTHONPATH to include the repo root (data/src imports)
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Run the Python script
echo "Starting main training script..."
PY_SCRIPT="$ROOT_DIR/src/c_grpo.py"

python3 "$PY_SCRIPT" \
  --model_name "$MODEL_NAME" \
  --train_datasets "$TRAIN_DATASETS" \
  --test_dataset "$TEST_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --reward_functions "$REWARD_FUNCTIONS" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --lora_rank "$LORA_RANK" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_generations "$NUM_GENERATIONS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --temperature "$TEMPERATURE" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --val_size "$VAL_SIZE" \
  --seed "$SEED" \
  --bleurt_model "$BLEURT_MODEL" \
  --bleurt_mix_beta "$BLEURT_MIX_BETA" \
  --project_name "$PROJECT_NAME" \
  --task "$TASK" \
  --train_device "$TRAIN_DEVICE" \
  --judge_device "$JUDGE_DEVICE"

# Stop the resource logger
echo "Stopping resource logger..."
kill $LOGGER_PID

# Print completion message
if [[ "$TASK" == "summary" ]]; then
  echo "GRPO summarization training completed."
elif [[ "$TASK" == "mcq" ]]; then
  echo "GRPO MCQ training completed."
elif [[ "$TASK" == "math" ]]; then
  echo "GRPO mathematical reasoning training completed."
elif [[ "$TASK" == "image_caption" ]]; then
  echo "GRPO image caption training completed."
else
  echo "GRPO machine translation training completed."
fi