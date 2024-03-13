# MMDetection Hello World

## Installation

### Clone the Repository

To get started, clone this repository along with its submodule. Run the following command in your terminal:

```bash
git clone --recurse-submodules https://github.com/aklein1995/object_detection_mmdetection.git
```
### Create Virtual Environment

```bash
python3.10 -m venv .venv
```

### PyTorch
```
pip install torch==2.1.0
```

### MMCV 
MMCV is a foundational library for computer vision that supports MMDetection. Install the specific version of MMCV according to [URL](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) (check "install with pip" section to get the concrete installation command)
```
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html 
```
In our case, assume to have installed CUDA 12.1 and torch2.1.

### MMDetection
Navigate to the mmdetection submodule directory and install MMDetection in editable mode:
```
cd mmdetection
pip install -v -e .
```

## Download Checkpoints
Visit the MMDetection GitHub repository, concretely at [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) section. 

Select the desired Object Detector. 

For instance, [Faster-RCNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn), and by clicking "model" download the .pth. Then, move it to ```/checkpoints``` folder.
