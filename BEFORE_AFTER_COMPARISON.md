# Before & After: Zero Vector Testing Implementation

## Problem Statement (Vietnamese)

```
muốn test với zero vector thì sửa đoạn này lại như nào nhỉ
```

The user wanted to modify their training command to test with zero vectors instead of loading features from a pickle file.

## Original Training Command

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
--output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased \\
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \\
--batch_size_val {batch_size_val} --visual_num_hidden_layers 6 \\
--decoder_num_hidden_layers 3 --datatype msrvtt --stage_two \\
--init_model weight/univl.pretrained.bin \\
--gradient_accumulation_steps={grad_accum}
"""
```

**Problem:** No way to test with zero vectors - features_path always loads real features from pickle file.

---

## Solution: Modified Training Command

### Minimal Change - Just Add One Flag

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
--use_zero_features                            # ← ADD THIS LINE
"""
```

**What changed:** Added `--use_zero_features` flag at the end.

**Result:** Now uses zero vectors instead of features from pickle file!

---

## Advanced Options

### Option 1: Zero Vectors (Default)
```bash
--use_zero_features
```
All video features will be zeros.

### Option 2: Random Uniform Distribution
```bash
--use_zero_features --fake_video_type random
```
Video features will be random values between 0 and 1.

### Option 3: Gaussian/Normal Distribution
```bash
--use_zero_features --fake_video_type gaussian
```
Video features will follow normal distribution (mean=0, std=1).

### Option 4: Control Random Seed
```bash
--use_zero_features --fake_video_type random --random_seed_video 42
```
Use specific seed for reproducibility.

---

## What Happens Under the Hood

### Before (Without --use_zero_features)
```
Load Data → Load Real Features from Pickle → Train Model
```

### After (With --use_zero_features)
```
Load Data → Load Features from Pickle → Replace with Zeros/Random → Train Model
                                        ↑
                                   FakeVideoDataLoader
```

The FakeVideoDataLoader wrapper intercepts batches and replaces the video tensor with fake features.

---

## Code Implementation Details

### New Arguments Added to `main_task_caption_test.py`
```python
parser.add_argument('--use_zero_features', action='store_true',
                    help="Use zero/fake video features instead of loading from pickle file")
parser.add_argument('--fake_video_type', type=str, default='zeros', 
                    choices=['zeros', 'random', 'gaussian'],
                    help="Type of fake video features")
parser.add_argument('--random_seed_video', type=int, default=42,
                    help="Random seed for generating fake video features")
```

### New FakeVideoDataLoader Class
```python
class FakeVideoDataLoader:
    """Wrapper that replaces video features with fake ones"""
    def __init__(self, original_dataloader, video_dim, max_frames, 
                 fake_type='zeros', random_seed=42):
        # ... initialization

    def __iter__(self):
        for batch in self.original_dataloader:
            # Replace video features (batch[3]) with fake features
            if self.fake_type == 'zeros':
                fake_video = torch.zeros_like(original_video)
            elif self.fake_type == 'random':
                fake_video = torch.from_numpy(
                    self.rng.uniform(0, 1, size=original_video.shape)
                ).float()
            elif self.fake_type == 'gaussian':
                fake_video = torch.from_numpy(
                    self.rng.normal(0, 1, size=original_video.shape)
                ).float()
            yield modified_batch
```

### All 4 Dataloaders Updated
```python
# Before
def dataloader_msrvtt_train(args, tokenizer):
    # ... create dataloader
    return dataloader, len(dataset), sampler

# After
def dataloader_msrvtt_train(args, tokenizer):
    # ... create dataloader
    
    # Wrap with fake video generator if flag is set
    if args.use_zero_features:
        dataloader = FakeVideoDataLoader(
            dataloader,
            video_dim=args.video_dim,
            max_frames=args.max_frames,
            fake_type=args.fake_video_type,
            random_seed=args.random_seed_video
        )
    
    return dataloader, len(dataset), sampler
```

---

## Complete Example Commands

### Example 1: Zero Vectors
```bash
torchrun --nproc_per_node=2 --standalone \
  main_task_caption_test.py \
  --do_train --num_thread_reader=4 \
  --epochs=10 --batch_size=128 \
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

### Example 2: Random Features
```bash
torchrun --nproc_per_node=2 --standalone \
  main_task_caption_test.py \
  --do_train --num_thread_reader=4 \
  --epochs=10 --batch_size=128 \
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

---

## Files Modified/Created

1. **main_task_caption_test.py** - Added fake video feature support (105 lines)
2. **USAGE_ZERO_FEATURES.md** - Comprehensive usage guide
3. **IMPLEMENTATION_SUMMARY.md** - Implementation summary
4. **BEFORE_AFTER_COMPARISON.md** - This file
5. **test_fake_video_dataloader.py** - Test/validation script
6. **.gitignore** - Updated to exclude Python cache

---

## Validation & Quality Assurance

✅ **Syntax Validation** - Code compiles without errors
✅ **Structural Validation** - All required methods present
✅ **Code Review** - Passed with minor whitespace fixes
✅ **Security Scan** - CodeQL found 0 vulnerabilities
✅ **Argument Validation** - All 3 new arguments properly added
✅ **Dataloader Validation** - All 4 dataloaders correctly updated

---

## Vietnamese Summary / Tóm tắt tiếng Việt

### Câu hỏi
"muốn test với zero vector thì sửa đoạn này lại như nào nhỉ"

### Trả lời
Chỉ cần thêm `--use_zero_features` vào cuối command:

```python
training_command = f"""torchrun --nproc_per_node={nproc} --standalone \\
main_task_caption_test.py \\
... (các tham số như cũ) ... \\
--use_zero_features
"""
```

### Các tùy chọn khác
- Dùng zero vectors: `--use_zero_features` (mặc định)
- Dùng random vectors: `--use_zero_features --fake_video_type random`
- Dùng Gaussian: `--use_zero_features --fake_video_type gaussian`
- Kiểm soát random seed: `--random_seed_video 42`

### Lưu ý
- File `--features_path` vẫn cần có nhưng sẽ bị thay thế bằng zero/random vectors
- Nên đổi `--output_dir` để không ghi đè kết quả thí nghiệm khác
- Xem thêm chi tiết trong `USAGE_ZERO_FEATURES.md`
