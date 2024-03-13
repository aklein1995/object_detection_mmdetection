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

## Non-Maximum Suppression (NMS) 
It can be adjusted in two ways:
1. Modify the score threshold in the configuration file, which is usually codified in the **test_cfg** [section](https://github.com/aklein1995/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/_base_/models/faster-rcnn_r50_fpn.py):
```
model = dict(
    ...
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.05,  # The score threshold
        nms=dict(type='nms', iou_threshold=0.5),  # The NMS IoU threshold
        max_per_img=100)
    ...
)
```

2. Filter the output based on the scores after the detections are made:
```
# Define your detection threshold
detection_threshold = 0.5  # Example threshold

# Assuming 'result' is the output from 'inference_detector' and contains the bounding boxes and scores
for i, (boxes, scores) in enumerate(zip(result[0], result[1])):
    # Filter out detections with scores below the threshold
    indices = scores > detection_threshold
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
```
