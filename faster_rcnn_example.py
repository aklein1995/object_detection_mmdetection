
# https://mmdetection.readthedocs.io/en/v2.9.0/1_exist_data_model.html
# https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html

from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
import cv2
import numpy as np
from PIL import Image

COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
'hair drier', 'toothbrush']

# Specify the path to model config and checkpoint file
config_file = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cpu'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=device)  

# test a single image
img = 'data/ski.png'
img_np = np.array(Image.open(img))
result = inference_detector(model, img)  # DetDataSample type object is provided in recent mmdetection
# save the visualization results to image files
# deprecated --> model.show_result(img, result, out_file='test_mmdetect.jpg')

###########################################################################################
# Assuming 'result' is the output from 'inference_detector' function and is an instance of DetDataSample or similar format
if isinstance(result, DetDataSample):
    # Access the pred_instances which contains the predictions
    pred_instances = result.pred_instances

    # Extract bounding boxes, scores, and labels
    bboxes = pred_instances.bboxes  # Bounding boxes
    scores = pred_instances.scores  # Confidence scores
    labels = pred_instances.labels  # Class labels

    # Print or process the extracted information
    print("Bounding Boxes:\n", bboxes)
    print("Scores:\n", scores)
    print("Labels:\n", labels)
else:
    # Handle other types of result formats (e.g., list of numpy arrays for simpler object detection models)
    print("The result format is not DetDataSample, please check your model type and result format.")

def draw_detections(boxes,colors,names,img):
    
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        # Draw rectangle (bounding box)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        # Put text (object name)
        cv2.putText(img, name, (xmin, ymin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return img

colors = np.random.uniform(0,255,size=labels.shape[0])
names = [COCO_NAMES[l] for l in labels]
detections = draw_detections(boxes= bboxes.cpu().numpy().astype(int),  # Assuming boxes are on GPU and in torch.Tensor format,
                colors= colors,
                names=names,#labels.cpu().numpy().astype(str),
                img=img_np)
# save image
detections_image = Image.fromarray(detections)  
detections_image.save('test_mmdetect.jpg')
