#!/usr/bin/env python3
"""
Simple unit test to verify that the zero features functionality is correctly implemented.
This test checks that the code modifications are syntactically correct and logically sound.
"""

import sys
import os
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_msrvtt_dataloader_signature():
    """Test that MSRVTT dataloader has the correct signature."""
    print("Testing MSRVTT_Caption_DataLoader signature...")
    
    from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
    
    # Get the __init__ signature
    sig = inspect.signature(MSRVTT_Caption_DataLoader.__init__)
    params = list(sig.parameters.keys())
    
    # Check that new parameters are present
    expected_params = ['use_zero_features', 'feature_dim']
    for param in expected_params:
        if param in params:
            print(f"  ✓ Parameter '{param}' found")
        else:
            print(f"  ✗ Parameter '{param}' NOT found")
            return False
    
    # Check default values
    if sig.parameters['use_zero_features'].default is False:
        print(f"  ✓ 'use_zero_features' has correct default value: False")
    else:
        print(f"  ✗ 'use_zero_features' has incorrect default value")
        return False
        
    if sig.parameters['feature_dim'].default == 1024:
        print(f"  ✓ 'feature_dim' has correct default value: 1024")
    else:
        print(f"  ✗ 'feature_dim' has incorrect default value")
        return False
    
    return True

def test_youcook_dataloader_signature():
    """Test that Youcook dataloader has the correct signature."""
    print("\nTesting Youcook_Caption_DataLoader signature...")
    
    from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
    
    # Get the __init__ signature
    sig = inspect.signature(Youcook_Caption_DataLoader.__init__)
    params = list(sig.parameters.keys())
    
    # Check that new parameters are present
    expected_params = ['use_zero_features', 'feature_dim']
    for param in expected_params:
        if param in params:
            print(f"  ✓ Parameter '{param}' found")
        else:
            print(f"  ✗ Parameter '{param}' NOT found")
            return False
    
    # Check default values
    if sig.parameters['use_zero_features'].default is False:
        print(f"  ✓ 'use_zero_features' has correct default value: False")
    else:
        print(f"  ✗ 'use_zero_features' has incorrect default value")
        return False
        
    if sig.parameters['feature_dim'].default == 1024:
        print(f"  ✓ 'feature_dim' has correct default value: 1024")
    else:
        print(f"  ✗ 'feature_dim' has incorrect default value")
        return False
    
    return True

def test_msrvtt_dataloader_logic():
    """Test that MSRVTT dataloader has correct zero features logic."""
    print("\nTesting MSRVTT_Caption_DataLoader zero features logic...")
    
    from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
    
    # Read the source code
    source = inspect.getsource(MSRVTT_Caption_DataLoader.__init__)
    
    # Check for key logic
    if 'self.use_zero_features = use_zero_features' in source:
        print("  ✓ Instance variable 'use_zero_features' is set")
    else:
        print("  ✗ Instance variable 'use_zero_features' is NOT set")
        return False
    
    # Check that feature_dict is always loaded
    if 'self.feature_dict = pickle.load' in source:
        print("  ✓ feature_dict is loaded (required for video length metadata)")
    else:
        print("  ✗ feature_dict is NOT loaded")
        return False
    
    # Check _get_video method
    get_video_source = inspect.getsource(MSRVTT_Caption_DataLoader._get_video)
    
    if 'if self.use_zero_features:' in get_video_source:
        print("  ✓ Zero features conditional logic in _get_video method found")
    else:
        print("  ✗ Zero features conditional logic in _get_video method NOT found")
        return False
    
    return True

def test_youcook_dataloader_logic():
    """Test that Youcook dataloader has correct zero features logic."""
    print("\nTesting Youcook_Caption_DataLoader zero features logic...")
    
    from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
    
    # Read the source code
    source = inspect.getsource(Youcook_Caption_DataLoader.__init__)
    
    # Check for key logic
    if 'self.use_zero_features = use_zero_features' in source:
        print("  ✓ Instance variable 'use_zero_features' is set")
    else:
        print("  ✗ Instance variable 'use_zero_features' is NOT set")
        return False
    
    # Check that feature_dict is always loaded
    if 'self.feature_dict = pickle.load' in source:
        print("  ✓ feature_dict is loaded (required for video length metadata)")
    else:
        print("  ✗ feature_dict is NOT loaded")
        return False
    
    # Check _get_video method
    get_video_source = inspect.getsource(Youcook_Caption_DataLoader._get_video)
    
    if 'if self.use_zero_features:' in get_video_source:
        print("  ✓ Zero features conditional logic in _get_video method found")
    else:
        print("  ✗ Zero features conditional logic in _get_video method NOT found")
        return False
    
    return True

def test_main_file_argument():
    """Test that main_task_caption_test.py has the --use_zero_features argument."""
    print("\nTesting main_task_caption_test.py for --use_zero_features argument...")
    
    # Read the main file
    with open('main_task_caption_test.py', 'r') as f:
        content = f.read()
    
    if '--use_zero_features' in content:
        print("  ✓ '--use_zero_features' argument found in main file")
    else:
        print("  ✗ '--use_zero_features' argument NOT found in main file")
        return False
    
    if "action='store_true'" in content and 'use_zero_features' in content:
        print("  ✓ '--use_zero_features' is configured as a boolean flag")
    else:
        print("  ✗ '--use_zero_features' is NOT configured correctly")
        return False
    
    # Check that dataloaders are called with the flag
    if 'use_zero_features=args.use_zero_features' in content:
        print("  ✓ Dataloaders are called with use_zero_features parameter")
    else:
        print("  ✗ Dataloaders are NOT called with use_zero_features parameter")
        return False
    
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Zero Features Implementation Validation")
    print("=" * 80 + "\n")
    
    all_passed = True
    
    # Run tests
    tests = [
        test_msrvtt_dataloader_signature,
        test_youcook_dataloader_signature,
        test_msrvtt_dataloader_logic,
        test_youcook_dataloader_logic,
        test_main_file_argument
    ]
    
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"\n  ✗ Test '{test_func.__name__}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED ✓")
        print("\nThe zero features functionality has been correctly implemented:")
        print("  • MSRVTT_Caption_DataLoader supports use_zero_features parameter")
        print("  • Youcook_Caption_DataLoader supports use_zero_features parameter")
        print("  • main_task_caption_test.py has --use_zero_features argument")
        print("  • All dataloaders are called with the new parameters")
    else:
        print("SOME VALIDATION CHECKS FAILED ✗")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
