import os
import glob
import platform
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class ToTensor(object):
    def __init__(self, config):
        self.config = config
        resize_height = config["TRANSFORMS"]["RESIZE"]["HEIGHT"]
        resize_width = config["TRANSFORMS"]["RESIZE"]["WIDTH"]
        
        self.resize = transforms.Resize((resize_height, resize_width))
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
        
        return {
            "image": image, 
            "depth": depth, 
            "valid": valid, 
            "mask": valid, 
            "dataset": self.config["DATASET"]["NAME"]
        }


class DIODE(Dataset):
    def __init__(self, config):
        self.config = config
        # OS
        os_env = config.get("OS_ENV", "WSL")  # Default to WSL
        prefix = config["PLATFORMS"][os_env]["PREFIX"]
        
        # Base dir
        self.data_dir_root = os.path.join(prefix, config["DIRECTORIES"]["BASE_DIR"])
        
        print(f"Initializing DIODE dataset from: {self.data_dir_root}")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir_root):
            raise ValueError(f"Data directory does not exist: {self.data_dir_root}")
        
        # Find all available deployments
        available_deployments = self._find_deployments()
        
        # Use deployments based on configuration
        if config["DEPLOYMENTS"]["USE_ALL"]:
            deployments = available_deployments
            print(f"Using all available deployments: {len(deployments)} found")
        else:
            deployments = config["DEPLOYMENTS"]["SPECIFIC"]
            print(f"Using specific deployments from config: {deployments}")
        
        # Validate deployments exist
        valid_deployments = [d for d in deployments if d in available_deployments]
        
        if not valid_deployments:
            raise ValueError(
                f"None of the specified deployments {deployments} found. "
                f"Available deployments: {available_deployments}"
            )
        
        print(f"Valid deployments: {valid_deployments}")
        
        # Collect all valid file triplets
        print("Collecting file triplets...")
        self.triplets = self._collect_file_triplets(valid_deployments)
        
        if not self.triplets:
            raise ValueError(f"No valid image-depth-mask triplets found in {self.data_dir_root}")
        
        print(f"Successfully loaded {len(self.triplets)} samples")
        self.transform = ToTensor(config)

    def _find_deployments(self):
        """Find all deployment directories."""
        images_dir = os.path.join(self.data_dir_root, self.config["DIRECTORIES"]["IMAGES"])
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
        
        dirs = self.config["DIRECTORIES"]
        files = self.config["FILES"]
        
        for deployment in deployments:
            # Get all images for this deployment
            image_pattern = os.path.join(
                self.data_dir_root, dirs["IMAGES"], deployment, 
                f"*{files['IMAGE_EXTENSION']}"
            )
            images = glob.glob(image_pattern)
            total_images += len(images)
            
            valid_count = 0
            for img_path in images:
                img_name = os.path.basename(img_path)
                base_name = img_name.replace(files["IMAGE_EXTENSION"], "")
                
                # Generate corresponding depth and mask paths
                depth_path = os.path.join(
                    self.data_dir_root, dirs["DEPTH"], 
                    deployment, base_name + files["DEPTH_SUFFIX"]
                )
                mask_path = os.path.join(
                    self.data_dir_root, dirs["MASKS"], 
                    deployment, base_name + files["MASK_SUFFIX"]
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

    def __len__(self):
        return len(self.triplets)


def get_diode_loader(config_path="config.yaml", **kwargs):

    config = load_config(config_path)
    
    # Use config defaults, allow kwargs to override
    batch_size = kwargs.pop('batch_size', config["DATALOADER"]["BATCH_SIZE"])
    num_workers = kwargs.pop('num_workers', config["DATALOADER"]["NUM_WORKERS"])
    shuffle = kwargs.pop('shuffle', config["DATALOADER"]["SHUFFLE"])
    
    dataset = DIODE(config)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, **kwargs)


# # Test the data loader
# if __name__ == "__main__":
#     try:
#         # All configuration comes from the YAML file
#         loader = get_diode_loader()
#         print(f"Dataset loaded successfully with {len(loader.dataset)} samples")
        
#         # Test loading one batch
#         batch = next(iter(loader))
#         print(f"Image: {batch['image'].shape}, Depth: {batch['depth'].shape}, Valid: {batch['valid'].shape}")
        
#         # Only DataLoader-specific parameters can be overridden
#         # loader = get_diode_loader(batch_size=2, num_workers=1)
#         # print(f"Dataset with custom batch size: {len(loader.dataset)} samples")
        
#     except Exception as e:
#         print(f"Error loading dataset: {e}")