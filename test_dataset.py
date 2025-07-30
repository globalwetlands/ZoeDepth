#!/usr/bin/env python3

import sys
import os
import time

# Add your project path
sys.path.append('/home/shakyafernando/projects/ZoeDepth')

from zoedepth.data.diode import get_diode_loader

def test_dataset():
    data_path = "/mnt/d/project-monocular-data-prep/data/ground-truths/tnc"
    
    print("TESTING DATASET LOADING")
    
    try:
        print("1. Creating dataset...")
        start_time = time.time()
        
        loader = get_diode_loader(
            data_path, 
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
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset()
    if not success:
        print("\nDataset test failed. Fix the issues above before training.")
        sys.exit(1)
    else:
        print("\nDataset is working correctly!")