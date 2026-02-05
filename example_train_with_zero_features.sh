#!/bin/bash
# Example training command for MSRVTT dataset with zero features

# Configuration
NPROC=1  # Number of GPUs
BATCH_SIZE=8
BATCH_SIZE_VAL=16
GRAD_ACCUM=1

# Dataset paths
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
FEATURES_PATH="data/msrvtt/msrvtt_videos_features.pickle"

# Model configuration
OUTPUT_DIR="ckpts/ckpt_msrvtt_caption_zero_features"
BERT_MODEL="bert-base-uncased"
INIT_MODEL="weight/univl.pretrained.bin"

# Training hyperparameters
EPOCHS=10
LR=3e-5
MAX_WORDS=48
MAX_FRAMES=48
VISUAL_LAYERS=6
DECODER_LAYERS=3

# Run training with ZERO FEATURES
echo "Starting training with ZERO FEATURES..."
echo "Output directory: $OUTPUT_DIR"
echo ""

torchrun --nproc_per_node=$NPROC --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=$EPOCHS --batch_size=$BATCH_SIZE \
--n_display=50 \
--train_csv $TRAIN_CSV \
--val_csv $VAL_CSV \
--data_path $DATA_PATH \
--features_path $FEATURES_PATH \
--output_dir $OUTPUT_DIR \
--bert_model $BERT_MODEL \
--do_lower_case --lr $LR \
--max_words $MAX_WORDS --max_frames $MAX_FRAMES \
--batch_size_val $BATCH_SIZE_VAL \
--visual_num_hidden_layers $VISUAL_LAYERS \
--decoder_num_hidden_layers $DECODER_LAYERS \
--datatype msrvtt --stage_two \
--init_model $INIT_MODEL \
--gradient_accumulation_steps=$GRAD_ACCUM \
--use_zero_features

echo ""
echo "Training completed! Check results in: $OUTPUT_DIR"
