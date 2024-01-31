import os
import random
from enum import Enum

import numpy as np
import requests
from PIL import Image
from io import BytesIO
import yaml
from loguru import logger

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys


# Enumerated type for model type
class ModelType(str, Enum):
    SEG = "segment"
    DETECT = "detect"


MISSING_CREDENTIALS_MSG = ("Please create a file './credentials.yaml' (or a different path, specified by the "
                           " `--kwarg credentials_path=<path_to_credentials_file>` when starting the backend server) "
                           "with the following fields: \n"
                           "LS_URL: http://<your_label_studio_host>:<port> \n"
                           "LS_API_TOKEN: <your_label_studio_api_token> \n")
MISSING_CONFIG_MSG = ("Please create a config yaml file './config.yaml' (or a different path, specified by the "
                      " `--kwarg config_path=<path_to_config_file>` when starting the backend server) with the "
                      "following fields: \n"
                      "MODEL_PATH: <path_to_your_model> \n"
                      f'MODEL_TYPE: <model_type, either "{ModelType.SEG}" or "{ModelType.DETECT}"> \n'
                      "MODEL_NAME: <model_name> \n"
                      "MODEL_VERSION: <model_version_tag> \n"
                      "MODEL_CLASSES: <model_class_list> \n")
INCORRECT_MODEL_TYPE_MSG = (f'Incorrect model type. Please specify either "{ModelType.SEG}" '
                            f'or "{ModelType.DETECT}" in config.yaml')


# Initialize class inherited from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        logger.info('Initializaing YOLOv8Model')
        # Verify and parse credentials file
        credentials_path = kwargs.get('credentials_path', 'credentials.yaml')
        self._ls_url, self._ls_api_token = self._verify_and_parse_credentials_file(credentials_path)
        # Verify and parse label config
        config_path = kwargs.get('config_path', 'config.yaml')
        self._model_name, self._model_version, self._model_type, self._model_classes, self._model_path = \
            self._verify_and_parse_config(config_path)
        self._model_version_full = f"{self._model_name}-{self._model_version}"
        kwargs["model_version"] = self._model_version_full
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)
        # Initialize logger
        os.makedirs('logs', exist_ok=True)
        logger.add(f'logs/log_{self._model_version_full}.log', rotation='500 MB', level='INFO')
        # Initialize self variables (PolygonLabels or RectangleLabels)
        control_type = 'PolygonLabels' if self._model_type == ModelType.SEG else 'RectangleLabels'
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, control_type=control_type, object_type='Image')
        self.labels = self._model_classes
        # Load model
        self.model = YOLO(self._model_path)
        logger.debug("Model loaded.")

    @staticmethod
    def _verify_and_parse_config(label_config_path):
        if not os.path.exists(label_config_path):
            logger.error(MISSING_CONFIG_MSG)
            raise FileNotFoundError(MISSING_CONFIG_MSG)
        # Retrieve model information:
        with open(label_config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            model_path = config['MODEL_PATH']
            if not os.path.exists(model_path):
                model_path_does_not_exist_msg = f"Model path {model_path} does not exist"
                logger.error(model_path_does_not_exist_msg)
                raise FileNotFoundError(model_path_does_not_exist_msg)
            model_type = config['MODEL_TYPE']
            model_type = model_type.lower()
            if model_type not in [ModelType.SEG, ModelType.DETECT]:
                logger.error(INCORRECT_MODEL_TYPE_MSG)
                raise ValueError(INCORRECT_MODEL_TYPE_MSG)
            model_name = str(config['MODEL_NAME'])
            model_version = str(config['MODEL_VERSION'])
            model_classes = config['MODEL_CLASSES']
            if not isinstance(model_classes, list):
                model_class_not_list_msg = "MODEL_CLASSES must be a list"
                logger.error(model_class_not_list_msg)
                raise TypeError(model_class_not_list_msg)
        return model_name, model_version, model_type, model_classes, model_path

    @staticmethod
    def _verify_and_parse_credentials_file(credentials_path):
        if not os.path.exists(credentials_path):
            logger.error(MISSING_CREDENTIALS_MSG)
            raise FileNotFoundError(MISSING_CREDENTIALS_MSG)
        if not credentials_path[-5:] == '.yaml':
            logger.error("Credentials file must be a yaml file")
            raise ValueError("Credentials file must be a yaml file")

        # URL with host
        with open("credentials.yaml", 'r') as stream:
            credentials = yaml.safe_load(stream)
            ls_url = credentials['LS_URL']
            ls_api_token = credentials['LS_API_TOKEN']
        return ls_url, ls_api_token

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = self._ls_url + image_url
        logger.info("Received Request")
        logger.debug(f"Image URL: {full_url}")

        # Header to get request
        header = {
            "Authorization": "Token " + self._ls_api_token
        }
        
        # Getting URL and loading image
        image_bytesio = BytesIO(requests.get(
            full_url, headers=header).content
        )
        image = Image.open(image_bytesio)
        # Height and width of image
        original_width, original_height = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0

        # Getting prediction using model
        results = self.model.predict(image)
        
        # Getting mask segments, boxes from model prediction
        count = 0
        for result in results:
            if self._model_type == ModelType.DETECT:
                for i, box in enumerate(result.boxes):
                    box_points = box.xyxy[0].tolist()
                    predictions.append({
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": box.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "x": box_points[0] / original_width * 100,
                            "y": box_points[1] / original_height * 100,
                            "width": (box_points[2] - box_points[0]) / original_width * 100,
                            "height": (box_points[3] - box_points[1]) / original_height * 100,
                            "rectanglelabels": [self.labels[int(box.cls.item())]]
                        }
                    })
                    score += box.conf.item()
                    count += 1
            else:
                for i, (box, segm) in enumerate(zip(result.boxes, result.masks.xy)):
                    polygon_points = segm / np.array([original_width, original_height]) * 100

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

                    # Calculating score (average of all scores, this is just the summing step)
                    score += box.conf.item()
                    count += 1

        logger.info("Returning Prediction")

        # Dict with final dicts with predictions
        score = float(score / (count + 1))
        print(f'Score: {score}')
        final_prediction = [{
            "result": predictions,
            "score": score,
            "model_version": self._model_version_full
        }]
        return final_prediction
    
    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}
