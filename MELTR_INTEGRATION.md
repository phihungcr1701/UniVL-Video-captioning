# MELTR Integration Guide for UniVL-Video-captioning

## Overview
This document summarizes the integration of MELTR (Meta-Learning Task Reweighting) from the MELTR/univl codebase into UniVL-Video-captioning. All changes follow the original MELTR/univl structure exactly.

## Files Created/Modified

### New Files Created:
1. **`modules/meltr.py`** - MELTR core module containing:
   - `MELTRgrad`: Computes hypergradients for task weight optimization
   - `MELTROptimizer`: Meta-optimizer for auxiliary network parameters
   - `MELTR`: Auxiliary neural network that learns task weights

2. **`utils/common_utils.py`** - Utility functions:
   - `str2list()`: Parse string arguments to lists
   - `AverageMeter()`: Track average metrics
   - Supporting helper functions

### Modified Files:

#### 1. **`modules/modeling.py`**
**Changes:**
- Added `tasks` parameter to `forward()` method (default: `[0, 1, 0, 0, 0, 0, 1, 0]`)
- Updated forward method to return **list of losses** instead of single loss
- Added loss functions: `self.loss_fct_joint` and `self.loss_fct_align`
- Added two new similarity methods:
  - `get_similarity_logits_joint()`: Mean pooling similarity (no cross-encoder)
  - `get_similarity_logits_align()`: Cross-encoder similarity

**Task Structure (8 tasks):**
```
Task 0: Joint similarity (without masking)
Task 1: Alignment similarity (cross-encoder, without masking)
Task 2: MLM loss (masked language modeling)
Task 3: MFM loss (masked frame modeling)
Task 4: M-Joint similarity (with masking)
Task 5: M-Align similarity (with masking)
Task 6: Decoder loss (without masking)
Task 7: Decoder loss (with masking)
```

#### 2. **`trainers/trainer.py`**
**Changes:**
- Updated `train_epoch()` signature to include:
  - `auxiliary_combine_net`: MELTR auxiliary network
  - `meta_optimizer`: MELTROptimizer instance
- Added logic to:
  - Call model with `tasks=args.tasks`
  - Combine losses using auxiliary network
  - Execute meta-optimization every `args.auxgrad-every` steps
  - Store batch history for meta-learning

#### 3. **`main_task_caption.py`**
**Changes:**
- Added MELTR-specific imports:
  - `from modules.meltr import MELTR, MELTROptimizer`
  - `from utils.common_utils import str2list, AverageMeter`
- Added MELTR command-line arguments:
  - `--tasks`: Task indicator list
  - `--target_tasks`: Meta-learning target tasks
  - `--lr_vnet`: Meta-learning rate
  - `--decay_vnet`: Meta-learning weight decay
  - `--auxgrad-every`: Meta-optimizer update frequency
  - `--transformer_dim`: MELTR network dimensions
  - `--vnet_max_grad`: Max gradient norm for meta-optimizer
  - `--max_grad_norm`: Model gradient norm
  - `--gamma`: Regularization weight
  - `--reg`: Regularization flag
- Added task calculation logic:
  - Calculates `taskNum` from `tasks` vector
  - Sets `target_tasks` based on task type (caption/retrieval)
- Updated main() to:
  - Initialize MELTR auxiliary network
  - Create meta-optimizer
  - Pass these to `train_epoch()`

## How It Works

### 1. Standard Training (without MELTR)
```python
loss = model(..., tasks=[0, 1, 0, 0, 0, 0, 1, 0])  # Returns list of losses
loss_combined = sum(loss) or auxiliary_combine_net(loss)
```

### 2. Meta-Learning (with MELTR)
```
For each training step:
  1. Forward pass with tasks=args.tasks
  2. Get list of losses
  3. Combine using auxiliary network
  4. Backward pass and optimizer step
  
Every N steps (auxgrad-every):
  5. Compute validation loss with target_tasks
  6. Compute meta-gradients
  7. Update auxiliary network parameters
```

## Usage Examples

### Basic Caption Training with MELTR:
```bash
python main_task_caption.py \
  --do_train \
  --output_dir ./output \
  --bert_model bert-base-uncased \
  --task_type caption \
  --tasks [1,1,1,1,1,1,1,1] \
  --target_tasks [0,0,0,0,0,0,1,0] \
  --auxgrad-every 3 \
  --lr_vnet 0.001
```

### Retrieval Training with MELTR:
```bash
python main_task_caption.py \
  --do_train \
  --output_dir ./output \
  --bert_model bert-base-uncased \
  --task_type retrieval \
  --tasks [1,1,1,1,1,1,1,1] \
  --target_tasks [0,1,0,0,0,0,0,0] \
  --auxgrad-every 3 \
  --lr_vnet 0.001
```

## Key Differences from Original UniVL

1. **Loss Output**: Model now returns a **list of 8 losses** per task
2. **Training Loop**: Uses dual optimizers (model optimizer + meta-optimizer)
3. **Meta-Learning**: Learns optimal task weights per data batch
4. **Flexibility**: Non-MELTR training still supported by default

## Integration Points

### In `modeling.py`:
- Line ~202: Forward method now uses `tasks` parameter
- Line ~178-179: Loss functions for joint/align tasks
- Line ~388-406: New similarity methods for tasks

### In `trainer.py`:
- Line ~6-7: New parameters in function signature
- Line ~32-39: Task-aware loss handling
- Line ~68-102: Meta-optimizer update logic

### In `main_task_caption.py`:
- Line ~15-16: MELTR imports
- Line ~100-110: MELTR arguments
- Line ~126-133: Task number calculation
- Line ~156-167: MELTR initialization
- Line ~177-182: MELTR in training call

## Testing Checklist

- [ ] MELTR module imports without errors
- [ ] Model forward pass returns list of 8 losses
- [ ] Auxiliary network initializes correctly
- [ ] Training loop executes meta-optimizer steps
- [ ] Model saves and loads correctly
- [ ] Evaluation metrics computed properly
- [ ] Both caption and retrieval tasks work
- [ ] Can switch on/off any task via tasks vector

## Debugging Notes

1. **Memory Issues**: MELTR requires more memory due to extra optimization steps. Reduce batch size if needed.

2. **Loss Explosion**: If auxiliary network creates unstable weights, reduce `--lr_vnet` or increase `--auxgrad-every`.

3. **Task Selection**: Ensure `tasks` vector length matches number of tasks (8).

4. **Target Tasks**: For caption, set only task 6 to 1; for retrieval, set tasks 1 or align to 1.

## References

- Original MELTR implementation: `d:\DaiHoc\Nam_4\NCKH\MELTR\univl\`
- Key files:
  - `MELTR/univl/modules/meltr.py`
  - `MELTR/univl/main.py`
  - `MELTR/univl/trainers/trainer.py` (if exists)

## Future Enhancements

1. Add warmup period before starting meta-learning
2. Implement adaptive `auxgrad-every` based on convergence
3. Add logging for individual task weights during training
4. Support for custom task definitions beyond the 8 defaults
