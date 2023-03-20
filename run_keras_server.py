# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import json

import cv2
import torch
import numpy as np
import flask

# initialize our Flask application and the Keras model
from flask import make_response
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

app = flask.Flask(__name__)
device = select_device('0')
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)

    # model = ResNet50(weights="imagenet")
    weights = '../runs/train/fixloss_220_300/weights/best.pt'
    global model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    print(" * 加载模型中...")


def prepare_image(image, target):
    # resize the input image and preprocess it
    # image = image.resize(target)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    # Read image

    img0 = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    # 将bgr转为rbg
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    # Padded resize
    img = letterbox(img0, target, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    # Convert . if the image mode is not RGB, convert it
    img = np.ascontiguousarray(img)
    # return the processed image and original image
    return img, img0


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the

    # ensure an image was properly uploaded to our endpoint
    data = {}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image
            image = flask.request.files["image"].read()
            # preprocess the image and prepare it for detection
            img, img0 = prepare_image(image, target=(640, 640))
            h, w, c = img0.shape
            print(img0.shape)
            print(img.shape)
            img = torch.from_numpy(img).to(device)
            img = img.half()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            print("开始推理...")
            global model
            model.half()
            model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))  # run once
            pred = model(img, augment=False)[0] # detect
            # NMS
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=0,
                                       agnostic=False,
                                       kpt_label=True)
            data["images"] = [{
                "width": w,
                "height": h
            }]
            data["categories"] = [{
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "metadata": {},
                "keypoints": ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
                              "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip"]
            }]
            data["annotations"] = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image ,here only one 处理每张图片,这里其实只有一张
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    scale_coords(img.shape[2:], det[:, :4], img0.shape, kpt_label=False)
                    scale_coords(img.shape[2:], det[:, 6:], img0.shape, kpt_label=True, step=3)
                    # Write results 处理每组预测结果
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        # Add bbox to image
                        # p1 = (x1, y1) = 矩形框的左上角 top-left   p2 = (x2, y2) = 矩形框的右下角 down-right
                        bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        keypoints = det[det_index, 6:].tolist()
                        for j in range(2, len(keypoints) + 1, 3):
                            if int(keypoints[j - 1]) != 0 and int(keypoints[j - 2]) != 0:
                                keypoints[j] = 2
                            if int(keypoints[j - 1]) == 0 and int(keypoints[j - 2]) == 0:
                                keypoints[j] = 0
                        item = {"id": det_index,
                                "image_id": 0,
                                "category_id": 1,
                                "segmentation": [],
                                "area": 0,
                                "bbox": bbox,
                                "iscrowd": False,
                                "isbbox": True,
                                "color": "#55e39f",
                                "keypoints": keypoints,
                                "metadata": {},
                                "num_keypoints": 13}
                        data["annotations"].append(item)

    # return the data dictionary as a JSON response
    res = {"coco": data}
    print(res)
    return json.dumps(res)


@app.after_request
def af_request(resp):
    """
    #请求钩子，在所有的请求发生后执行，加入headers。
    :param resp:
    :return:
    """
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(port=6688)
