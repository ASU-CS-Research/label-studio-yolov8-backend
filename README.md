# YOLOv8 ML backend for the Label Studio

This project contains an ML backend for classifying pills in Label Studio. It uses the YOLOv8 model and can segment and classify pills as capsules or tablets. This project is a fork of [label-studio-yolov8-backend](https://github.com/seblful/label-studio-yolov8-backend) written by [seblful](https://github.com/seblful). 

## Project Structure
### The repository contains the following files and directories:

- **Dockerfile**: The Dockerfile for building the backend container.

- **docker-compose.yml**: The docker-compose file for running the backend.

- **model.py**: The Python code for the ML backend model.

- **models/**: The pre-trained YOLOv8 models for classification. `best-detect.pt` was trained on video frames from the entrance of the AppMAIS hives to detect and classify drone and worker honeybees. `stripe-seg.pt` was trained on segmented bee images to segment the stripes from the bee abdomens.

- **config_files/**: The config files associated with each model. `config-detect.yaml` specifies the required attributes for the `best-detect.pt` trained model and `config-seg.yaml` does the same for `stripe-seg.pt`. The required attributes are:
    - `MODEL_PATH`: Path to the model, such as `"./models/best-detect.pt"`.
    - `MODEL_TYPE`: Model output type, currently only supports "segment" or "detect", with polygon labels and rectangle labels respectively.
    - `MODEL_NAME`: Any name for the model, only used to concatenate with the `MDOEL_VERSION` to record what model gave the annotation. Annotation information recorded as `<MODEL_NAME>-<MDOEL_VERSION>`.
    - `MODEL_VERSION`: Any string for the model version, only used to concatenate with the `MDOEL_NAME` to record what model gave the annotation. Annotation information recorded as `<MODEL_NAME>-<MDOEL_VERSION>`.
    - `MODEL_CLASSES`: List of strings for the names of the model classes. Case-sensitivity applies here for matching model classes to the label classes in Label Studio.

- **uwsgi.ini**: The uWSGI configuration file for running the backend.

- **supervisord.conf**: The supervisord configuration file for running the backend processes.

- **requirements.txt**: The list of Python dependencies for the backend.

- **credentials.yaml**: **Not saved in the repo, but required to run!** this file is similar to the `config.yaml` files, with only two fields. The fields are `LS_URL` written as `"http://<host>:<port>"`, and `LS_API_TOKEN` written as `"<API_KEY>"`. 


## Getting Started
1. Clone the Label Studio Machine Learning Backend git repository. From the command line, run the following:

    ```git clone https://github.com/seblful/label-studio-yolov8-backend.git```

2. Paste you Label Studio API key in ```model.py```

3. To use this backend, you'll need to have Docker and docker-compose installed. Then, run the following command to start the backend:

    ```docker-compose up```

This will start the backend on localhost:9090.

Check if it works:

```$ curl http://localhost:9090/health```

```{"status":"UP"}```

4. Connect running backend to Label Studio:

    ```label-studio start --init new_project --ml-backends http://localhost:9090```

5. Start the labeling process.

## Training
Model training is **not included** in this project. This will probably be added later.

## Contributing
Contributions to this project are welcome. To contribute, please submit an issue or pull request.
