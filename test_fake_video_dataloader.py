#!/usr/bin/env python3
"""
Quick test to verify the FakeVideoDataLoader functionality works correctly.
"""
import torch
import numpy as np
from main_task_caption_test import FakeVideoDataLoader

def create_mock_batch():
    """Create a mock batch similar to what the real dataloader would return."""
    batch_size = 4
    max_frames = 10
    video_dim = 1024
    max_words = 20
    
    # Simulating a batch tuple structure
    input_ids = torch.randint(0, 1000, (batch_size, max_words))
    input_mask = torch.ones(batch_size, max_words)
    segment_ids = torch.zeros(batch_size, max_words)
    video = torch.randn(batch_size, max_frames, video_dim)  # This will be replaced
    video_mask = torch.ones(batch_size, max_frames)
    
    return (input_ids, input_mask, segment_ids, video, video_mask)

class MockDataLoader:
    """Mock dataloader for testing."""
    def __init__(self, num_batches=3):
        self.num_batches = num_batches
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield create_mock_batch()

def test_zero_features():
    """Test zero features generation."""
    print("Testing zero features...")
    mock_loader = MockDataLoader(num_batches=2)
    fake_loader = FakeVideoDataLoader(
        mock_loader, 
        video_dim=1024, 
        max_frames=10, 
        fake_type='zeros',
        random_seed=42
    )
    
    for batch in fake_loader:
        video = batch[3]
        assert torch.all(video == 0), "Zero features should be all zeros"
        print(f"  ✓ Batch shape: {video.shape}, all zeros: {torch.all(video == 0)}")
    
    print("  ✓ Zero features test passed!\n")

def test_random_features():
    """Test random uniform features generation."""
    print("Testing random uniform features...")
    mock_loader = MockDataLoader(num_batches=2)
    fake_loader = FakeVideoDataLoader(
        mock_loader, 
        video_dim=1024, 
        max_frames=10, 
        fake_type='random',
        random_seed=42
    )
    
    for batch in fake_loader:
        video = batch[3]
        assert torch.all(video >= 0) and torch.all(video <= 1), "Random features should be in [0, 1]"
        assert not torch.all(video == 0), "Random features should not be all zeros"
        print(f"  ✓ Batch shape: {video.shape}, min: {video.min():.4f}, max: {video.max():.4f}")
    
    print("  ✓ Random features test passed!\n")

def test_gaussian_features():
    """Test Gaussian distributed features generation."""
    print("Testing Gaussian features...")
    mock_loader = MockDataLoader(num_batches=2)
    fake_loader = FakeVideoDataLoader(
        mock_loader, 
        video_dim=1024, 
        max_frames=10, 
        fake_type='gaussian',
        random_seed=42
    )
    
    all_values = []
    for batch in fake_loader:
        video = batch[3]
        all_values.append(video.numpy())
        print(f"  ✓ Batch shape: {video.shape}, mean: {video.mean():.4f}, std: {video.std():.4f}")
    
    all_values = np.concatenate([v.reshape(-1) for v in all_values])
    overall_mean = all_values.mean()
    overall_std = all_values.std()
    print(f"  Overall mean: {overall_mean:.4f}, std: {overall_std:.4f}")
    assert abs(overall_mean) < 0.2, "Gaussian mean should be close to 0"
    assert abs(overall_std - 1.0) < 0.2, "Gaussian std should be close to 1"
    print("  ✓ Gaussian features test passed!\n")

def test_reproducibility():
    """Test that same random seed produces same results."""
    print("Testing reproducibility with random seed...")
    
    mock_loader1 = MockDataLoader(num_batches=1)
    fake_loader1 = FakeVideoDataLoader(
        mock_loader1, 
        video_dim=1024, 
        max_frames=10, 
        fake_type='random',
        random_seed=42
    )
    
    mock_loader2 = MockDataLoader(num_batches=1)
    fake_loader2 = FakeVideoDataLoader(
        mock_loader2, 
        video_dim=1024, 
        max_frames=10, 
        fake_type='random',
        random_seed=42
    )
    
    batch1 = next(iter(fake_loader1))[3]
    batch2 = next(iter(fake_loader2))[3]
    
    assert torch.all(batch1 == batch2), "Same seed should produce same results"
    print("  ✓ Reproducibility test passed!\n")

def test_dataloader_length():
    """Test that wrapper preserves dataloader length."""
    print("Testing dataloader length preservation...")
    mock_loader = MockDataLoader(num_batches=5)
    fake_loader = FakeVideoDataLoader(
        mock_loader, 
        video_dim=1024, 
        max_frames=10, 
        fake_type='zeros',
        random_seed=42
    )
    
    assert len(fake_loader) == len(mock_loader), "Wrapper should preserve length"
    print(f"  ✓ Length preserved: {len(fake_loader)}\n")

if __name__ == "__main__":
    print("="*60)
    print("Testing FakeVideoDataLoader Implementation")
    print("="*60 + "\n")
    
    try:
        test_zero_features()
        test_random_features()
        test_gaussian_features()
        test_reproducibility()
        test_dataloader_length()
        
        print("="*60)
        print("✅ All tests passed successfully!")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
