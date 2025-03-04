# Data structure:
Data directories:
 - base: `data/ground-truths/tnc`
 - images/frames (PNG): `data/ground-truths/tnc/extracted-frames/`
 - depth-maps (.npy): `data/ground-truths/tnc/depth-maps-metric/`

 Data location:
  - Located in (SharePoint)[https://griffitheduau.sharepoint.com/sites/glowstorage?e=1%3Ae5e828d3869f424b801530d23b597d99&CT=1709096554091&OR=OWA-NT&CID=f6317c35-7544-4bc6-45a4-465892c46794]
# Training the model
- Create virtual environment:
`python -m venv env`

- Install requirements:
`pip install -r requirements`

- Run train_mono:
`WANDB_MODE=disabled python3 train_mono.py -d diode_outdoor -m zoedepth`

# Modifications
scripts are located under `zoedepth/`

## data
- Scripts modified: 
    - `data_mono.py`: All lines of code relavent to `masks` have benn commented out
    - `diode.py`: 
        - Use a named identiy instead of `lambda` function
        - Loads an image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. 
            - Code has been simplified to covert and image into Tensor in ranged [0,1].
        - Paths to access frames (RBG images), and depth maps (.npy files) have been adjusted.    
        - Code (including paths) relevant to `masks` have been removed.
        - Path to load diode dataset have been adjusted
        - Print statement to show image and depth shape

## Trainers           
- Scripts modified:
    - `base_trainer`: Need adjustements to `wandb`, however `WANDB_MODE=disabled` is use during training.
    - `zoedepth_trainer`: Commented out code relavent to `masks` and/or set `masks=None`.

## Utils
- Scripts modified:
    - `config`: Adjusted the path for `diode_outdoor` dataset.