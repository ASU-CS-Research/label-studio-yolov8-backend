import os
import random

import numpy as np
import requests
from PIL import Image
from io import BytesIO

from flask import jsonify
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path

# URL with host
LS_URL = "http://127.0.0.1:8080"
# LS_URL = "http://192.168.100.3:8080"
LS_API_TOKEN = "f015bd5469e57d9b150e31ea63bc006d29e541fa"


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables (PolygonLabels or RectangleLabels)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image')
        self.labels = ['stripe']
        # Load model
        self.model = YOLO("best-seg.pt")

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = LS_URL + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN
        }
        
        # Getting URL and loading image
        image_bytesio = BytesIO(requests.get(
            full_url, headers=header).content
        )
        print("CONTENT: ", requests.get(full_url, headers=header).content)
        # print("IMAGE FULL URL: ", full_url)
        # print("IMAGE BYTESIO: ", image_bytesio)
        image = Image.open(image_bytesio)
        # Height and width of image
        original_width, original_height = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        

        # Getting prediction using model
        results = self.model.predict(image)
        
        max_input_dim = 256
        max_image_dim = max(image.size)
        print("ORIGINAL IMAGE DIM: ", image.size)
        print("MAX IMAGE DIM: ", max_image_dim)
        print("MAX INPUT DIM: ", max_input_dim)
        scale = max_input_dim / max_image_dim

        min_image_dim = min(image.size)
        # Find padding distance
        padding = (min_image_dim * scale) % 32
        # scale = 1
        # Getting mask segments, boxes from model prediction
        for result in results:
            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.xy)):
                # segm = segm.cpu().numpy()
                # print(f'scale: {scale}')
                # # segm = (segm.xy[0] * scale / 2).tolist()
                # segm = segm.xy[0].tolist()
                # print("SEGM: " + str(segm))
                # # print("SEGM SHAPE: " + str(segm.shape))
                # # print("SEGM TYPE: " + str(type(segm)))
                # # 2D array with poligon points
                # polygon_points = segm

                polygon_points = segm / \
                                 np.array([original_width, original_height]) * 100

                polygon_points = polygon_points.tolist()

                # print("SCORE TYPE: " + str(type(box.conf.item())))
                # print("POLYGON LABELS TYPE: " + str(type(self.labels[int(box.cls.item())])))
                # print("POLYGON POINTS TYPE: " + str(type(polygon_points)))
                # print("ORIGINAL WIDTH TYPE: " + str(type(original_width)))
                # print("ORIGINAL HEIGHT TYPE: " + str(type(original_height)))


                # Adding dict to prediction
                predictions.append({
                    "from_name" : self.from_name,
                    "to_name" : self.to_name,
                    "id": str(i),
                    "type": "polygonlabels",
                    "score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_width,
                    "image_rotation": 0,
                    "value": {
                        "points": polygon_points,
                        # polygonlabels
                        "polygonlabels": [self.labels[int(box.cls.item())]]
                    }})

                # Calculating score
                score += box.conf.item()


        print(10*"#", "Returned Prediction", 10*"#")

        # Dict with final dicts with predictions
        final_prediction = [{  # [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8s"
        }]  # ]
        # print("FINAL SCORE TYPE: " + str(type(score)))
        # print("FINAL PREDICTION TYPE: " + str(type(final_prediction)))
        # print("FINAL PREDICTIONS TYPE: " + str(type(predictions)))
        # print("FINAL PREDICTION: ", jsonify(final_prediction))

        return final_prediction
    
    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}