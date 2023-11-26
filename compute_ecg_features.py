from dataloader import ECGDataModule, HDF5ECGDataset
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm

if __name__ == '__main__':
    model = load_model("ribeiro_model/model/model.hdf5", compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    
    # model output:
    # output: shape = (N, 6). Each entry contains a probability between 0 and 1, 
    # and can be understood as the probability of a given abnormality to be present. 
    # The abnormalities it predicts are (in that order): 1st degree AV block(1dAVb), 
    # right bundle branch block (RBBB), left bundle branch block (LBBB), 
    # sinus bradycardia (SB), atrial fibrillation (AF), sinus tachycardia (ST). 
    # The abnormalities are not mutually exclusive, so the probabilities do not 
    # necessarily sum to one.
    
    dm = ECGDataModule('datasets/hhmusedata', batch_size=32, mode=HDF5ECGDataset.Mode.MODE_ECG_WITH_EXAM_ID,
                       num_workers=0, sample_size=23291, train_fraction=1.0, dev_fraction=0.0, test_fraction=0.0)
        
    dl = dm.train_dataloader()
    
    with open("ribeiro_model_predictions.csv", "w") as f:
        f.write(",".join(['exam_id', '1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']) + "\n")
        for i, batch in tqdm(enumerate(dl)):
            ours = batch['ecg']
            ribeiro_shuffled = tf.convert_to_tensor(ours.numpy(), dtype=tf.float32)
            # Create a list of indices in the target order
            indices = [7, 9, 11, 6, 10, 8] + list(range(6))

            # Use tf.gather to rearrange the channels
            ribeiro = tf.gather(ribeiro_shuffled, indices, axis=2)

            print(ribeiro.shape)
            
            model_out = model.predict(ribeiro)
            
            for prediction, exam_id in zip(model_out, batch['exam_ids']):
                row = [int(exam_id), *[round(float(v), 4) for v in prediction]]
                f.write(",".join([str(v) for v in row]) + "\n")