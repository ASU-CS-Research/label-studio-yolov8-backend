import os
import random

import numpy as np
import requests
from PIL import Image
from io import BytesIO
import yaml
from loguru import logger

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path

assert os.path.exists("credentials.yaml"), ("Please create a file './credentials.yaml' with the following fields: \n"
                                            "LS_URL: http://<your_label_studio_host>:<port> \n"
                                            "LS_API_TOKEN: <your_label_studio_api_token> \n")
assert os.path.exists("config.yaml"), ("Please create a file './config.yaml' with the following fields: \n"
                                       "MODEL_PATH: <path_to_your_model> \n"
                                       'MODEL_TYPE: <model_type, either "seg" or "detect"> \n'
                                       "MODEL_NAME: <model_name> \n"
                                       "MODEL_VERSION: <model_version_tag> \n"
                                       "MODEL_CLASSES: <model_class_list> \n")

# URL with host
with open("credentials.yaml", 'r') as stream:
    credentials = yaml.safe_load(stream)
    LS_URL = credentials['LS_URL']
    LS_API_TOKEN = credentials['LS_API_TOKEN']
# Retireve model information:
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    MODEL_PATH = config['MODEL_PATH']
    MODEL_TYPE = config['MODEL_TYPE']
    MODEL_NAME = config['MODEL_NAME']
    MODEL_VERSION = config['MODEL_VERSION']
    MODEL_CLASSES = config['MODEL_CLASSES']


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables (PolygonLabels or RectangleLabels)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image')
        self.labels = MODEL_CLASSES
        # Load model
        self.model = YOLO(MODEL_PATH)

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = LS_URL + image_url
        print(10*"#", "Received Request", 10*"#")
        print("Image URL:", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN
        }
        
        # Getting URL and loading image
        image_bytesio = BytesIO(requests.get(
            full_url, headers=header).content
        )
        print(f"Image loaded with size {image_bytesio.getbuffer().nbytes} bytes")
        image = Image.open(image_bytesio)
        # Height and width of image
        original_width, original_height = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0

        # Getting prediction using model
        results = self.model.predict(image)
        
        # Getting mask segments, boxes from model prediction
        for result in results:
            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.xy)):
                polygon_points = segm / \
                                 np.array([original_width, original_height]) * 100

                polygon_points = polygon_points.tolist()
                # Adding dict to prediction
                predictions.append({
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "id": str(i),
                    "type": "polygonlabels",
                    "score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_width,
                    "image_rotation": 0,
                    "value": {
                        "points": polygon_points,
                        "polygonlabels": [self.labels[int(box.cls.item())]]
                    }})

                # Calculating score
                score += box.conf.item()

        print(10*"#", "Returned Prediction", 10*"#")

        # Dict with final dicts with predictions
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": f"{MODEL_NAME}-{MODEL_VERSION}"
        }]
        return final_prediction
    
    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}