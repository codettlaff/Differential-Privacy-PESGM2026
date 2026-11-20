import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide Warnings

# Settings
settings1 = {
    'dataset_name': 'ampds_cnn_mini',
    'model_name': 'cnn_nilm_mini',
    'batch_size': 32,
    'num_epochs': 50,
    'rand_seed': 42}

def generate_model(dataset_name, model_name, batch_size, num_epochs, rand_seed):
    tf.keras.utils.set_random_seed(rand_seed)

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    processed_data_dir = os.path.join(base_dir, 'processed_data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    processed_data_filepath = os.path.join(processed_data_dir, dataset_name + '.npz')
    model_save_filepath = os.path.join(models_dir, model_name + '.keras')
    predictions_save_filepath = os.path.join(results_dir, model_name + '_predictions.npz')
    eacc_results_save_filepath = os.path.join(results_dir, model_name + '_eacc.npz')

    data = np.load(processed_data_filepath)
    appliance_names = data['appliance_names']
    agg_column_names = data['agg_labels']
    apps_column_names = data['apps_labels']

    # Load Normalized and Original Data
    p_agg_norm = data['agg_norm'][:, 2] # Keep only P-agg Column
    q_agg_norm = data['agg_norm'][:, 4] # Keep only Q-agg Column
    p_agg = data['agg_data'][:, 2] # Keep only P-agg Column
    q_agg = data['agg_data'][:, 3] # Keep only Q-agg Column

    datetimes = data['apps_norm'][:, :1] # Isolate Time Column
    p_apps_norm = data['apps_norm'][:, 2:] # Keep Appliance Columns, discard time and id columns
    p_apps = data['apps_data'][:, 2:] # Keep Appliance Columns, discard time and id columns

    train_indices = data['train_indices']
    val_indices = data['val_indices']
    test_indices = data['test_indices']

    window_length = data['window_length']

    # Scaling Factors
    apps_min = data['apps_min']
    apps_max = data['apps_max']

    def make_pq_image(p, q):
        S = np.zeros((p.size + 1, q.size + 1))
        S[1:, 0] = np.abs(p)
        S[0, 1:] = np.abs(q)
        S[1:, 1:] = np.hypot(p[:, None], q[None, :])
        return S

    def create_sample(frame_start_idx, p_agg_norm, q_agg_norm, p_apps_norm, window_length):
        p = p_agg_norm[frame_start_idx : frame_start_idx + window_length]
        q = q_agg_norm[frame_start_idx : frame_start_idx + window_length]
        center_timestep = frame_start_idx + window_length // 2
        targets = p_apps_norm[center_timestep]
        S = make_pq_image(p, q)
        return S, targets

    def create_batch(indices, p_agg_norm, q_agg_norm, p_apps_norm, window_length):
        S_list, targets_list = [], []
        for i in indices:
            S, targets = create_sample(i, p_agg_norm, q_agg_norm, p_apps_norm, window_length)
            S_list.append(S)
            targets_list.append(targets)
        S_batch = np.stack(S_list)[..., np.newaxis]
        return S_batch, np.array(targets_list)

    def create_model_architecture(n_appliances):
        input_layer = layers.Input(shape=(31, 31, 1))

        # Single convolutional branch
        x = layers.Conv2D(30, (10, 10), activation='relu')(input_layer)
        x = layers.Conv2D(30, (8, 8), activation='relu')(x)
        x = layers.Conv2D(40, (6, 6), activation='relu')(x)
        x = layers.Conv2D(50, (5, 5), activation='relu')(x)
        x = layers.Conv2D(50, (5, 5), activation='relu')(x)
        x = layers.Flatten()(x)

        # Dense + Output Layers
        x = layers.Dense(1024, activation='relu')(x)
        out = layers.Dense(n_appliances)(x)   # one output per appliance

        # Model
        model = models.Model(inputs=input_layer, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse')

        return model

    # Model Training
    def train_epoch(model, p_agg_norm, q_agg_norm, p_apps_norm, train_indices, window_length, batch_size):

        train_loss, num_batches = 0.0, 0

        for i in tqdm(range(0, len(train_indices), batch_size), desc="Training", leave=False):
            batch_input_indices = train_indices[i : i + batch_size]
            S, targets = create_batch(batch_input_indices, p_agg_norm, q_agg_norm, p_apps_norm, window_length)
            loss = model.train_on_batch(S, targets)
            train_loss += loss
            num_batches += 1

        train_loss /= num_batches
        return train_loss

    def validate_epoch(model, p_agg_norm, q_agg_norm, p_apps_norm, val_indices, window_length, batch_size):

        val_loss, num_batches = 0.0, 0

        for i in tqdm(range(0, len(val_indices), batch_size), desc="Validating", leave=False):
            batch_input_indices = val_indices[i : i + batch_size]
            S, targets = create_batch(batch_input_indices, p_agg_norm, q_agg_norm, p_apps_norm, window_length)

            loss = model.test_on_batch(S, targets)
            val_loss += loss
            num_batches += 1

        val_loss /= num_batches
        return val_loss

    def train_model(p_agg_norm, q_agg_norm, p_apps_norm, train_indices, val_indices,
                window_length, batch_size, num_epochs, patience=5, model_save_filepath=None):

        model = create_model_architecture(n_appliances=p_apps_norm.shape[1])

        best_val_loss = np.inf
        best_weights = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in tqdm(range(num_epochs), desc='Epochs'):

            # Shuffle training windows each epoch
            shuffled_train_indices = np.random.permutation(train_indices)

            train_loss = train_epoch(model, p_agg_norm, q_agg_norm, p_apps_norm,
                                    shuffled_train_indices, window_length, batch_size)
            val_loss = validate_epoch(model, p_agg_norm, q_agg_norm, p_apps_norm,
                                    val_indices, window_length, batch_size)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Early stopping + keep best weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping')
                    break

        # Restore best model
        if best_weights is not None:
            model.set_weights(best_weights)

        if model_save_filepath is not None:
            model.save(model_save_filepath)

        return model, history

    model, history = train_model(p_agg_norm, q_agg_norm, p_apps_norm, train_indices, val_indices,
                window_length, batch_size, num_epochs, model_save_filepath=model_save_filepath)

    # Inference
    def predict(model, p_agg_norm, q_agg_norm, p_apps_norm, test_indices, window_length, batch_size):
        apps_pred_all = []

        for i in tqdm(range(0, len(test_indices), batch_size), desc='Predicting'):
            batch_inp = test_indices[i : i + batch_size]
            S, _ = create_batch(
                batch_inp, p_agg_norm, q_agg_norm,
                p_apps_norm, window_length)
            apps_pred_all.append(np.asarray(model.predict_on_batch(S)))

        apps_pred = np.concatenate(apps_pred_all, axis=0)
        return np.clip(apps_pred, 0, None)   # power can't be negative

    p_apps_norm_dt = np.hstack([datetimes, p_apps_norm])
    apps_pred_norm = predict(model, p_agg_norm, q_agg_norm, p_apps_norm, test_indices, window_length, batch_size)

    # Construct Datetimes column for predicted indices
    target_indices = test_indices + window_length // 2       # window start -> center (target) timestep
    pred_datetimes = datetimes[target_indices]               # shape (len(test_indices), 1)
    apps_pred_norm_dt = np.hstack([pred_datetimes, apps_pred_norm])
    apps_pred_norm_dt = apps_pred_norm_dt[np.argsort(apps_pred_norm_dt[:, 0])]

    # Un-normalize data
    def unnormalize(data_dt, apps_min, apps_max):
        out = data_dt.copy()
        out = data_dt * (apps_max - apps_min) + apps_min
        return out

    p_apps_dt = np.hstack([datetimes, p_apps])
    pred_datetimes = apps_pred_norm_dt[:, :1] # remove pred_datetimes from apps_pred_norm_dt
    apps_pred_norm_only = apps_pred_norm_dt[:, 1:]
    apps_pred = unnormalize(apps_pred_norm_only, apps_min, apps_max) # pass through unnormalize function
    apps_pred_dt = np.hstack([pred_datetimes, apps_pred]) # add pred_datetimes back on

    # Match actual and predicted rows by datetime
    row_idx = np.searchsorted(p_apps_dt[:, 0], apps_pred_dt[:, 0])
    y_true_norm = p_apps_norm_dt[row_idx, 1:]
    y_pred_norm = apps_pred_norm_dt[:, 1:]
    y_true = p_apps_dt[row_idx, 1:]
    y_pred = apps_pred_dt[:, 1:]

    predictions = {
        'y_true_norm': y_true_norm,
        'y_pred_norm': y_pred_norm,
        'y_true': y_true,
        'y_pred' : y_pred}

    # save predictions
    np.savez(predictions_save_filepath, **predictions)

    # Compute EACC
    def compute_eacc(y_true, y_pred):
        return 1 - np.sum(np.abs(y_pred - y_true)) / (2 * np.sum(y_true))

    eacc = {name: compute_eacc(y_true[:, a], y_pred[:, a]) for a, name in enumerate(appliance_names)}
    eacc['Overall'] = compute_eacc(y_true, y_pred)

    # save eacc
    pd.Series(eacc, name='EACC').to_csv(os.path.join(results_dir, model_name + '_eacc.csv'))

if __name__ == '__main__': generate_model(**settings1)
