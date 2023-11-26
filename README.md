# ECG Hospitalization Prediction
This project aims to develop a system for predicting hospitalization outcomes based on electrocardiogram (ECG) data. Given ECG data from individuals in an ambulatory setting (AP) and subsequent hospitalization records, the task is to create predictive models that can automatically interpret vital sign monitor data and forecast the likelihood of hospitalization.

# Data
The data consists of ECG recordings and patient information from a large-scale study of cardiac patients in the Czech Republic. The ECG data is stored in HDF5 files, each containing a single patient’s ECG signal and metadata. The patient information is stored in a CSV file, containing demographic and clinical variables for each patient. The hospitalization records are stored in a XML file, containing the admission and discharge dates, diagnoses, and procedures for each patient.

# Methodology
The methodology consists of three main steps:

## ECG feature extraction: 
This step involves parsing the ECG data and extracting key metrics such as ventricular and atrial rates, QRS duration, QT interval, and heart rate variability. These features are computed using the compute_ecg_features.py script, which uses the PatientEKGFile class from the libs folder. The output of this step is a CSV file containing the ECG features for each patient.

## ECG condition prediction: 
This step involves using a pre-trained deep learning model to predict specific heart conditions based on the ECG features and patient information. The model is a convolutional neural network (CNN) that was trained on a large dataset of ECG signals and labels from the PhysioNet Computing in Cardiology Challenge 2020. The model is loaded using the load_model function from the tensorflow.keras.models module. The model takes as input a vector of ECG features and patient information, and outputs a vector of probabilities for each of the 9 possible heart conditions. The predictions are made using the main_predictor.py script, which uses the dataloader.py and model.py modules from the ribeiro_model folder. The output of this step is a CSV file containing the ECG condition predictions for each patient.

## Hospitalization decision: 
This step involves using a pre-loaded gradient boosting classifier to make a final decision on whether the patient should be sent home or to a cardiology clinic based on the ECG condition predictions. The classifier is a XGBoost model that was trained on a subset of the data using the main_autoencoder.py script, which uses the resnet_model.py module. The classifier is loaded using the joblib.load function from the joblib module. The classifier takes as input a vector of ECG condition predictions, and outputs a binary label indicating the hospitalization decision. The decisions are made using the predict.py script, which uses the gbc.pkl file. The output of this step is a CSV file containing the hospitalization decisions for each patient.

# Usage
First, clone the repository and install required packages:

```
git clone https://github.com/CaptainTrojan/hackhealth2023.git
cd hackhealth2023
pip install -r requirements.txt
```

Run prediction:
```
python predict.py
```

The required arguments are ECG muse file location, patient's age, and patient's visit reason
```
predict.py [-h] --ecg ECG --age AGE --visit_reason {1,2,3,4,9}

options:
  -h, --help            show this help message and exit
  --ecg ECG             Path to MUSE XML ECG
  --age AGE             Patient age
  --visit_reason {1,2,3,4,9}
                        Patient visit reason: 1: bolest na hrudi, 2: dušnost, 3: palpitace, 4: bezvědomí, 9: jiný důvod
```
