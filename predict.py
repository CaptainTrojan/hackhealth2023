from libs.pacient_ekg import PatientEKGFile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import argparse
import pandas as pd
import joblib 


def scale_to_milivolts_and_add_leads(array):
    """
    V1, V2, V3, V4, V5, V6, aVL, I, aVR, II, aVF, III
    """
    ekg_matrix = np.zeros(shape=(*array.shape[:-1], 12), dtype=np.float32)
    ekg_matrix[..., 0:6] = array[..., 0:6]
    ekg_matrix[..., 7] = array[..., 6]
    ekg_matrix[..., 9] = array[..., 7]

    # aVL = I - II/2
    ekg_matrix[..., 6] = np.subtract(array[..., 6], array[..., 7] / 2)
    # aVR = -(I + II)/2
    ekg_matrix[..., 8] = -(np.add(array[..., 6], array[..., 7]) / 2)
    # aVF = II - I/2
    ekg_matrix[..., 10] = np.subtract(array[..., 7], array[..., 6] / 2)
    # II - I = III
    ekg_matrix[..., 11] = np.subtract(array[..., 7], array[..., 6])

    return ekg_matrix * 0.00488


def parse_ecg(full_path, BLOCK_SIZE):
    pac_ekg = PatientEKGFile()
    try:
        pac_ekg.load_data(full_path)
    except Exception as e:
        # If the EKG has invalid structure, skip it.
        # print(full_path, e)
        return None

    if pac_ekg.get_shape() is None:
        # print(full_path, "bad shape")
        return None

    unique_id = pac_ekg.get_unique_identifier()
    gender = str(pac_ekg.gender).lower()
    if gender == 'none':
        is_male = -1
    elif gender == 'male':
        is_male = 1
    else:
        is_male = 0

    age = pac_ekg.age if pac_ekg.age is not None else -1
    ventricular_rate = int(pac_ekg.ekg_mesurements['VentricularRate']) \
        if pac_ekg.ekg_mesurements is not None and \
           'VentricularRate' in pac_ekg.ekg_mesurements \
        else -1
    atrial_rate = int(pac_ekg.ekg_mesurements['AtrialRate']) \
        if pac_ekg.ekg_mesurements is not None and \
           'AtrialRate' in pac_ekg.ekg_mesurements \
        else -1
    weight = int(pac_ekg.weight) if pac_ekg.weight is not None else -1
    height = int(pac_ekg.height) if pac_ekg.height is not None else -1
    acquisition_date = str(pac_ekg.acquisition_date) if pac_ekg.acquisition_date is not None else '<unknown>'

    np_mat = pac_ekg.gen_tensor_matrix(False)  # (8, 5000)
    if np_mat.shape[1] < BLOCK_SIZE:
        true_size = np_mat.shape[1]
        pad_size = BLOCK_SIZE - true_size
        np_mat = np.pad(np_mat, ((0, 0), (0, pad_size)))
    elif np_mat.shape[1] > BLOCK_SIZE:
        cut_size = np_mat.shape[1] - BLOCK_SIZE
        start_cut_size = cut_size // 2
        end_cut_size = cut_size // 2 + cut_size % 2
        np_mat = np_mat[:, start_cut_size:-end_cut_size]  # cut 1s from front and
        true_size = BLOCK_SIZE
    else:
        true_size = BLOCK_SIZE

    assert np_mat.shape[1] == BLOCK_SIZE
    np_mat = np_mat.T
    # np_mat /= 2 ** 15 * 0.001 * 4.88
    np_mat = scale_to_milivolts_and_add_leads(np_mat)

    return np_mat, ventricular_rate, atrial_rate


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    
    reason_map = {
        1: 'bolest na hrudi',
        2: 'dušnost',
        3: 'palpitace',
        4: 'bezvědomí',
        9: 'jiný důvod'
    }
    
    patient_visit_reason_help = 'Patient visit reason: ' + ', '.join([f'{k}: {v}' for k, v in reason_map.items()])
    
    parser.add_argument("--ecg", type=str, help='Path to MUSE XML ECG', required=True)
    parser.add_argument("--age", type=int, help='Patient age', required=True)
    parser.add_argument("--visit_reason", type=int, choices=[1, 2, 3, 4, 9], help=patient_visit_reason_help, required=True)
    
    args = parser.parse_args()

    np_mat, ventricular_rate, atrial_rate = parse_ecg(args.ecg, 4096)
    
    model = load_model("ribeiro_model/model/model.hdf5", compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    
    ribeiro_shuffled = tf.convert_to_tensor(np.expand_dims(np_mat, 0), dtype=tf.float32)
    # Create a list of indices in the target order
    indices = [7, 9, 11, 6, 10, 8] + list(range(6))

    # Use tf.gather to rearrange the channels
    ribeiro = tf.gather(ribeiro_shuffled, indices, axis=2)    
    model_out = model.predict(ribeiro)
    
    features = {
        'ventricular_rate': ventricular_rate,
        'atrial_rate': atrial_rate,
        'age': args.age,
        'VisitReason': args.visit_reason
    }
    
    for key, prediction in zip(['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], model_out[0]):
        features[key] = prediction
        
    features_df = pd.DataFrame.from_dict(features, orient='index')
    # print(features_df)
    
    loaded_model = joblib.load('gbc.pkl')
    predictions = loaded_model.predict(features_df.T)
    prediction = predictions[0]
    
    print("\n\n")
    
    print("Verdict:")
    if prediction == 0:
        print("Patient should be sent home (outpatient).")
    else:
        print("Patient should be sent to the cardiology clinic.")