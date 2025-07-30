#!/usr/bin/env python3

import sys
import os
import time

# Add your project path
sys.path.append('/home/shakyafernando/projects/ZoeDepth')

from zoedepth.data.diode import get_diode_loader

def test_dataset():
    config_path = "config.yaml" 
    
    print("TESTING DATASET LOADING")
    
    try:
        print("1. Creating dataset...")
        start_time = time.time()
        
        loader = get_diode_loader(
            config_path=config_path,
            batch_size=1, 
            num_workers=0,  # Important: no multiprocessing
            shuffle=False
        )
        
        init_time = time.time() - start_time
        print(f"   Dataset initialized in {init_time:.2f} seconds")
        print(f"   Dataset size: {len(loader.dataset)}")
        
        if len(loader.dataset) == 0:
            print("   ERROR: Dataset is empty!")
            return False
        
        print("\n2. Loading first batch...")
        start_time = time.time()
        
        batch = next(iter(loader))
        
        load_time = time.time() - start_time
        print(f"   First batch loaded in {load_time:.2f} seconds")
        print(f"   Image shape: {batch['image'].shape}")
        print(f"   Depth shape: {batch['depth'].shape}")
        print(f"   Valid shape: {batch['valid'].shape}")
        print(f"   Dataset name: {batch['dataset']}")
        
        print("\n3. Loading second batch...")
        start_time = time.time()
        
        batch = next(iter(loader))
        
        load_time = time.time() - start_time
        print(f"   Second batch loaded in {load_time:.2f} seconds")
        
        print("\n4. Testing multiple batches...")
        start_time = time.time()
        
        count = 0
        for batch in loader:
            count += 1
            if count >= 5:  # Test 5 batches
                break
                
        total_time = time.time() - start_time
        print(f"   Loaded {count} batches in {total_time:.2f} seconds")
        print(f"   Average time per batch: {total_time/count:.2f} seconds")
        
        print("\n" + "=" * 50)
        print("DATASET TEST PASSED!")
        print("=" * 50)
        return True
        
    except FileNotFoundError as e:
        if "config.yaml" in str(e):
            print(f"\nERROR: Could not find config.yaml file at: {config_path}")
            print("Make sure your config.yaml file exists in the correct location.")
            print("Current working directory:", os.getcwd())
        else:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_config():
    """Test with a different config file path if needed."""
    possible_config_paths = [
        "config.yaml",
        "../config.yaml", 
        "../../config.yaml",
        "/home/shakyafernando/projects/ZoeDepth/config.yaml"
    ]
    
    print("Searching for config.yaml in possible locations...")
    
    for config_path in possible_config_paths:
        if os.path.exists(config_path):
            print(f"Found config file at: {config_path}")
            return test_dataset_with_path(config_path)
    
    print("No config.yaml file found in any of these locations:")
    for path in possible_config_paths:
        print(f"  - {path}")
    return False

def test_dataset_with_path(config_path):
    """Test with a specific config path."""
    print(f"TESTING DATASET WITH CONFIG: {config_path}")
    
    try:
        loader = get_diode_loader(config_path=config_path, batch_size=1, num_workers=0)
        print(f"Dataset loaded successfully with {len(loader.dataset)} samples")
        
        # Test one batch
        batch = next(iter(loader))
        print(f"Batch loaded - Image: {batch['image'].shape}, Depth: {batch['depth'].shape}")
        return True
        
    except Exception as e:
        print(f"Error with config {config_path}: {e}")
        return False

if __name__ == "__main__":
    # First try with default config path
    success = test_dataset()
    
    # If that fails, try searching for config in different locations
    if not success:
        print("\nTrying alternative config locations...")
        success = test_with_different_config()
    
    if not success:
        print("\nDataset test failed. Fix the issues above before training.")
        print("\nMake sure:")
        print("1. Your config.yaml file exists and is accessible")
        print("2. The paths in config.yaml are correct")
        print("3. The OS_ENV setting in config.yaml matches your environment")
        sys.exit(1)
    else:
        print("\nDataset is working correctly!")