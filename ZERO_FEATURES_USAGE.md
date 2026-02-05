# Using Zero Features for Testing

## Overview
The training script `main_task_caption_test.py` now supports testing with zero vectors instead of using actual video features. This is useful for ablation studies to understand the model's behavior when visual information is replaced with zeros while maintaining realistic temporal structure.

## Usage

To use zero features, simply add the `--use_zero_features` flag to your training command:

```bash
# Original command (loads actual features from pickle file)
torchrun --nproc_per_node=1 --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=10 --batch_size=8 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path data/msrvtt/msrvtt_videos_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 16 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \
--init_model weight/univl.pretrained.bin \
--gradient_accumulation_steps=1

# Modified command (uses zero vectors)
torchrun --nproc_per_node=1 --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=10 --batch_size=8 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path data/msrvtt/msrvtt_videos_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption_zero --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 16 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \
--init_model weight/univl.pretrained.bin \
--gradient_accumulation_steps=1 \
--use_zero_features
```

## How It Works

When `--use_zero_features` is enabled:

1. **Feature Loading**: The pickle file specified in `--features_path` **is still loaded** to extract metadata (video lengths, temporal information)
2. **Feature Values**: All video feature values are replaced with zeros, but the temporal structure (video length, frame count) is preserved from the original data
3. **Feature Dimension**: Uses the dimension specified by `--video_dim` (default: 1024)
4. **Video Masks**: Masks are generated based on actual video lengths, ensuring realistic temporal attention patterns

This approach allows for proper ablation studies where:
- The model architecture remains unchanged
- Temporal structures and video lengths are realistic
- Only the feature values themselves are zeroed out
- The model can learn purely from text without visual information

## Notes

- The `--features_path` argument is **required** even when using `--use_zero_features` because the pickle file contains metadata about video lengths
- This feature works with both MSRVTT and YouCook datasets
- Useful for ablation studies to understand the importance of visual features
- Can help debug model architecture issues or test text-only performance
- Memory usage is similar to normal mode since the pickle file is still loaded (only feature values are replaced with zeros)
