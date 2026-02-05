# Using Zero/Fake Video Features for Testing

This guide explains how to test the UniVL video captioning model with zero vectors or random features instead of actual video features from a pickle file.

## Overview

The `main_task_caption_test.py` script now supports testing with fake video features through the `--use_zero_features` flag. This is useful for:
- Testing model behavior when visual information is not available
- Debugging text-only performance
- Experimenting with different feature initialization strategies
- Understanding the contribution of visual features to the model

## Usage

### Basic Example with Zero Vectors

To test with zero vectors (all zeros), simply add the `--use_zero_features` flag to your training command:

```bash
training_command = f"""torchrun --nproc_per_node={nproc} --standalone \\
main_task_caption_test.py \\
--do_train --num_thread_reader=4 \\
--epochs=10 --batch_size={batch_size} \\
--n_display=50 \\
--train_csv data/msrvtt/MSRVTT_train.9k.csv \\
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \\
--data_path data/msrvtt/MSRVTT_data.json \\
--features_path data/msrvtt/msrvtt_videos_features.pickle \\
--output_dir ckpts/ckpt_msrvtt_caption_zero \\
--bert_model bert-base-uncased \\
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \\
--batch_size_val {batch_size_val} --visual_num_hidden_layers 6 \\
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \\
--init_model weight/univl.pretrained.bin \\
--gradient_accumulation_steps={grad_accum} \\
--use_zero_features
"""
```

**Note:** When using `--use_zero_features`, the `--features_path` is still required (for compatibility with the dataloader initialization), but the actual features loaded from the pickle file will be replaced with the specified fake features.

### Advanced Options

#### 1. Use Random Uniform Features (values between 0 and 1)

```bash
--use_zero_features --fake_video_type random
```

#### 2. Use Gaussian (Normal) Distribution Features

```bash
--use_zero_features --fake_video_type gaussian
```

#### 3. Control Random Seed for Reproducibility

```bash
--use_zero_features --fake_video_type random --random_seed_video 42
```

To get different random features each run:
```bash
--use_zero_features --fake_video_type random --random_seed_video -1
```

### Complete Example Commands

#### Zero Vectors (Default)
```bash
torchrun --nproc_per_node=2 --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=10 --batch_size=128 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path data/msrvtt/msrvtt_videos_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption_zeros \
--bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \
--init_model weight/univl.pretrained.bin \
--gradient_accumulation_steps=1 \
--use_zero_features
```

#### Random Uniform Features
```bash
torchrun --nproc_per_node=2 --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=10 --batch_size=128 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path data/msrvtt/msrvtt_videos_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption_random \
--bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \
--init_model weight/univl.pretrained.bin \
--gradient_accumulation_steps=1 \
--use_zero_features --fake_video_type random --random_seed_video 42
```

#### Gaussian Features
```bash
torchrun --nproc_per_node=2 --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=10 --batch_size=128 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path data/msrvtt/msrvtt_videos_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption_gaussian \
--bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \
--init_model weight/univl.pretrained.bin \
--gradient_accumulation_steps=1 \
--use_zero_features --fake_video_type gaussian --random_seed_video 42
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_zero_features` | flag | False | Enable fake video features instead of loading from pickle file |
| `--fake_video_type` | str | 'zeros' | Type of fake features: 'zeros', 'random' (uniform), or 'gaussian' |
| `--random_seed_video` | int | 42 | Random seed for generating fake features (use -1 for non-deterministic) |

## Expected Behavior

When `--use_zero_features` is enabled:
1. The script will display a warning message in the logs indicating that fake video features are being used
2. The actual video features from the pickle file will be loaded but immediately replaced with the specified fake features
3. All other aspects of training/evaluation remain the same
4. The output directory should be different to avoid overwriting results from real feature experiments

## Vietnamese Answer / Câu trả lời

**Muốn test với zero vector thì sửa đoạn training command như sau:**

Thêm flag `--use_zero_features` vào cuối command:

```python
training_command = f"""torchrun --nproc_per_node={nproc} --standalone \\
main_task_caption_test.py \\
--do_train --num_thread_reader=4 \\
--epochs=10 --batch_size={batch_size} \\
--n_display=50 \\
--train_csv data/msrvtt/MSRVTT_train.9k.csv \\
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \\
--data_path data/msrvtt/MSRVTT_data.json \\
--features_path data/msrvtt/msrvtt_videos_features.pickle \\
--output_dir ckpts/ckpt_msrvtt_caption_zero --bert_model bert-base-uncased \\
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \\
--batch_size_val {batch_size_val} --visual_num_hidden_layers 6 \\
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \\
--init_model weight/univl.pretrained.bin \\
--gradient_accumulation_steps={grad_accum} \\
--use_zero_features
"""
```

**Các tùy chọn khác:**
- `--use_zero_features`: Dùng vector zero (mặc định)
- `--use_zero_features --fake_video_type random`: Dùng random vector (uniform distribution)
- `--use_zero_features --fake_video_type gaussian`: Dùng Gaussian distribution
- `--random_seed_video 42`: Kiểm soát random seed (dùng -1 để random mỗi lần chạy)
