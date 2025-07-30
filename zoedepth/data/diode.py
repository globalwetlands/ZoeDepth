import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        self.resize = transforms.Resize((480, 640))
        self.transform_image = transforms.Compose([transforms.ToTensor()])
        self.transform_depth = transforms.ToTensor()
        self.transform_valid = transforms.ToTensor()

    def __call__(self, sample):
        image, depth, valid = sample["image"], sample["depth"], sample["valid"]
        
        image = self.transform_image(image)
        depth = self.transform_depth(depth)
        valid = self.transform_valid(valid.astype(np.float32))
        
        # Resize all components
        image = self.resize(image)
        depth = self.resize(depth)
        valid = self.resize(valid)
        
        return {"image": image, "depth": depth, "valid": valid, "mask": valid, "dataset": "diode_outdoor"}


class DIODE(Dataset):
    def __init__(self, data_dir_root, deployments=None):
        print(f"Initializing DIODE dataset from: {data_dir_root}")
        self.data_dir_root = data_dir_root
        
        # Check if data directory exists
        if not os.path.exists(data_dir_root):
            raise ValueError(f"Data directory does not exist: {data_dir_root}")
        
        # Find available deployments
        if deployments is None:
            deployments = self._find_deployments()
        
        if not deployments:
            print(f"No deployment directories found in {data_dir_root}")
            print(f"Directory contents: {os.listdir(data_dir_root) if os.path.exists(data_dir_root) else 'Directory does not exist'}")
            raise ValueError(f"No deployment directories found in {data_dir_root}")
        
        print(f"Found {len(deployments)} deployments: {deployments}")
        
        # Collect all valid file triplets
        print("Collecting file triplets...")
        self.triplets = self._collect_file_triplets(deployments)
        
        if not self.triplets:
            print(f"Found deployments: {deployments}")
            print("Checking directory structure...")
            self._debug_directory_structure(deployments[:2])
            raise ValueError(f"No valid image-depth-mask triplets found in {data_dir_root}")
        
        print(f"Successfully loaded {len(self.triplets)} samples")
        self.transform = ToTensor()

    def _find_deployments(self):
        """Find all deployment directories."""
        images_dir = os.path.join(self.data_dir_root, "extracted-frames-20250617")
        deployments = []
        
        if os.path.exists(images_dir):
            for item in os.listdir(images_dir):
                if os.path.isdir(os.path.join(images_dir, item)) and item.startswith("deployment_"):
                    deployments.append(item)
        
        return sorted(deployments)

    def _collect_file_triplets(self, deployments):
        """Collect valid image-depth-mask file triplets."""
        triplets = []
        total_images = 0
        
        for deployment in deployments:
            # Get all images for this deployment
            image_pattern = os.path.join(
                self.data_dir_root, "extracted-frames-20250617", deployment, "*.png"
            )
            images = glob.glob(image_pattern)
            total_images += len(images)
            
            valid_count = 0
            for img_path in images:
                img_name = os.path.basename(img_path)
                
                # Generate corresponding depth and mask paths
                depth_path = os.path.join(
                    self.data_dir_root, "norm-depth-maps-20250729-deployment", 
                    deployment, img_name.replace(".png", "_depth.npy")
                )
                mask_path = os.path.join(
                    self.data_dir_root, "binary-masks-20250729-deployment", 
                    deployment, img_name.replace(".png", "_depth_mask.npy")
                )
                
                # Only include if all files exist
                if os.path.exists(depth_path) and os.path.exists(mask_path):
                    triplets.append((img_path, depth_path, mask_path))
                    valid_count += 1
            
            print(f"  {deployment}: {len(images)} images, {valid_count} valid triplets")
        
        print(f"Total: {total_images} images, {len(triplets)} valid triplets")
        return triplets

    def __getitem__(self, idx):
        image_path, depth_path, mask_path = self.triplets[idx]
        
        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        
        # Load depth and mask as numpy arrays
        depth = np.load(depth_path).astype(np.float32)
        valid = np.load(mask_path)
        
        sample = {"image": image, "depth": depth, "valid": valid}
        return self.transform(sample)

    def _debug_directory_structure(self, deployments):
        """Debug helper to check directory structure."""
        for deployment in deployments:
            print(f"\nChecking {deployment}:")
            
            img_dir = os.path.join(self.data_dir_root, "extracted-frames-20250617", deployment)
            depth_dir = os.path.join(self.data_dir_root, "norm-depth-maps-20250729-deployment", deployment)
            mask_dir = os.path.join(self.data_dir_root, "binary-masks-20250729-deployment", deployment)
            
            print(f"  Images dir: {img_dir} (exists: {os.path.exists(img_dir)})")
            print(f"  Depth dir: {depth_dir} (exists: {os.path.exists(depth_dir)})")
            print(f"  Masks dir: {mask_dir} (exists: {os.path.exists(mask_dir)})")
            
            if os.path.exists(img_dir):
                img_count = len(glob.glob(os.path.join(img_dir, "*.png")))
                print(f"  Images found: {img_count}")
                if img_count > 0:
                    sample_img = glob.glob(os.path.join(img_dir, "*.png"))[0]
                    img_name = os.path.basename(sample_img)
                    expected_depth = os.path.join(depth_dir, img_name.replace(".png", "_depth.npy"))
                    expected_mask = os.path.join(mask_dir, img_name.replace(".png", "_depth_mask.npy"))
                    print(f"  Sample image: {img_name}")
                    print(f"  Expected depth: {expected_depth} (exists: {os.path.exists(expected_depth)})")
                    print(f"  Expected mask: {expected_mask} (exists: {os.path.exists(expected_mask)})")

    def __len__(self):
        return len(self.triplets)


def get_diode_loader(data_dir_root, batch_size=1, num_workers=0, deployments=None, **kwargs):

    # Use only deployments that have data by default
    if deployments is None:
        deployments = ['deployment_16', 'deployment_3019']
    
    dataset = DIODE(data_dir_root, deployments=deployments)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)


# Test the data loader
if __name__ == "__main__":
    data_dir = "/mnt/d/project-monocular-data-prep/data/ground-truths/tnc"
    
    try:
        # Test with all deployments
        # loader = get_diode_loader(data_dir, batch_size=1)
        # print(f"Dataset loaded successfully with {len(loader.dataset)} samples")

        # Use specific deployments  
        loader = get_diode_loader(data_dir, deployments=['deployment_16', 'deployment_3019'])
        print(f"Dataset loaded successfully with {len(loader.dataset)} samples")
        
        # Test loading one batch
        batch = next(iter(loader))
        print(f"Image: {batch['image'].shape}, Depth: {batch['depth'].shape}, Valid: {batch['valid'].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")