# Zero Features Implementation Summary

## Overview
This implementation adds support for testing the UniVL model with zero vectors instead of actual video features. This enables ablation studies to understand the model's dependence on visual information.

## Key Features

1. **Simple flag-based activation**: Just add `--use_zero_features` to your training command
2. **Maintains temporal structure**: Video lengths and masks are preserved from actual data
3. **Memory optimized**: For YouCook dataset, actual features are not loaded
4. **Validation warnings**: Alerts if feature_dim doesn't match actual dimension
5. **Works with both datasets**: MSRVTT and YouCook support

## Quick Start

### Vietnamese (Tiếng Việt)
Để test với zero vector (vector không), chỉ cần thêm flag `--use_zero_features`:

```bash
torchrun --nproc_per_node=1 --standalone \
main_task_caption_test.py \
--do_train --num_thread_reader=4 \
--epochs=10 --batch_size=8 \
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

### English
To test with zero vectors, simply add the `--use_zero_features` flag to your command.

See `example_train_with_zero_features.sh` for a complete example.

## Implementation Details

### What happens when `--use_zero_features` is enabled:

1. ✅ Feature pickle file is still loaded (for metadata)
2. ✅ Video lengths are extracted from actual data
3. ✅ All feature values are replaced with zeros
4. ✅ Video masks reflect actual video lengths
5. ✅ Feature dimension can be customized via `--video_dim`

### Files Modified

- `dataloaders/dataloader_msrvtt_caption.py` - MSRVTT dataset support
- `dataloaders/dataloader_youcook_caption.py` - YouCook dataset support  
- `main_task_caption_test.py` - Command-line argument
- `.gitignore` - Exclude Python cache files

### New Files

- `ZERO_FEATURES_USAGE.md` - Detailed documentation
- `example_train_with_zero_features.sh` - Example script
- `test_implementation.py` - Validation test suite
- `ZERO_FEATURES_SUMMARY.md` - This file

## Testing

Run the validation test:
```bash
python3 test_implementation.py
```

All tests should pass ✓

## Use Cases

1. **Ablation Study**: Understand how much the model depends on visual features
2. **Text-only Performance**: Test caption generation with only text encoder
3. **Debug Architecture**: Verify model works without visual encoder complications
4. **Baseline Comparison**: Compare against models with actual features

## Notes

- The `--features_path` is still required (for metadata)
- Works with both MSRVTT and YouCook datasets
- Memory optimized for YouCook (features not loaded)
- MSRVTT loads full pickle for length extraction
- No changes to model architecture required
- Compatible with existing checkpoints

## Questions?

See `ZERO_FEATURES_USAGE.md` for detailed documentation.
