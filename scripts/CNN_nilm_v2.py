import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide Warnings

# Settings
settings1 = {
    'dataset_name': 'ampds_CNN_1week',
    'model_name': 'cnn_nilm_1week',
    'batch_size': 32,
    'num_epochs': 50}

settings2 = {
    'dataset_name': 'ampds_CNN_2year',
    'model_name': 'cnn_nilm_2year',
    'batch_size': 32,
    'num_epochs': 50}

settings = [settings1, settings2]

# Samples Generation
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

# Model Architecture
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

def denormalize(apps_norm, apps_min, apps_max):
    apps = apps_norm.copy()
    apps = apps_norm * (apps_max - apps_min) + apps_min
    return apps

# Analyze Results
def compute_eacc(y_true, y_pred):
    return 1 - np.sum(np.abs(y_pred - y_true)) / (2 * np.sum(y_true))

def plot_pred_versus_actual(x_true, y_pred, y_true, appliance_names, title='Predicted versus Actual Appliance Power', save_filepath=None, show=False):
    # Plot multiple time-series plots on one figure.
    # The first plot has x_true and y_true;
    # subsequent plots are y_pred and y_true, one plot per appliance.
    n_apps = len(appliance_names)
    t = np.arange(len(x_true))

    fig, axes = plt.subplots(n_apps + 1, 1, figsize=(14, 3 * (n_apps + 1)), sharex=True, squeeze=False)
    axes = axes[:, 0]

    # Aggregate vs. true appliance power
    axes[0].plot(t, np.ravel(x_true), label='Aggregate (true)', color='black', linewidth=1)
    for i, app in enumerate(appliance_names):
        axes[0].plot(t, y_true[:, i], label=f'{app}', linewidth=0.8, alpha=0.8)
    axes[0].set_title('Aggregate and True Appliance Power')
    axes[0].set_ylabel('Power')
    axes[0].legend(loc='upper right', fontsize='small')

    # Predicted vs. true, per appliance
    for i, app in enumerate(appliance_names):
        ax = axes[i + 1]
        ax.plot(t, y_true[:, i], label='True', linewidth=1)
        ax.plot(t, y_pred[:, i], label='Predicted', linewidth=1, alpha=0.8)
        ax.set_title(app)
        ax.set_ylabel('Power')
        ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Timestamp')
    fig.suptitle(title)
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=300, bbox_inches='tight')
    
    if show: plt.show()

def generate_model(dataset_name, model_name, batch_size, num_epochs):

    # Directories
    base_dir = os.path.join(os.path.basename(__file__), '..')
    processed_data_dir = os.path.join(base_dir, 'processed_data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    processed_data_filepath = os.path.join(processed_data_dir, dataset_name + '.npz')
    model_save_filepath = os.path.join(models_dir, model_name + '.keras')
    model_results_folderpath = os.path.join(results_dir, model_name + '_results')
    os.makedirs(model_results_folderpath, exist_ok=True)
    predictions_save_filepath = os.path.join(model_results_folderpath, model_name + '_predictions.npz')
    results_eacc_save_filepath = os.path.join(model_results_folderpath, model_name + '_eacc.csv')
    results_plot_save_filepath = os.path.join(model_results_folderpath, model_name + '_results_plot.png')

    # Load Data
    data = np.load(processed_data_filepath)

    p_agg_norm = data['agg_norm'][:, 0] # Keep only P-agg Column
    q_agg_norm = data['agg_norm'][:, 2] # Keep only Q-agg Column
    p_agg = data['agg_data'][:, 0] # Keep only P-agg Column

    # Train Model
    model, history = train_model(
        p_agg_norm,
        q_agg_norm,
        data['apps_norm'],
        data['train_indices'],
        data['val_indices'],
        data['window_length'],
        batch_size,
        num_epochs,
        model_save_filepath=model_save_filepath)

    # Inference
    apps_pred_norm = predict(
        model,
        p_agg_norm,
        q_agg_norm,
        data['apps_norm'],
        data['test_indices'],
        data['window_length'],
        batch_size)

    # Denormalize Predictions
    apps_pred = denormalize(apps_pred_norm, data['apps_min'], data['apps_max'])

    # Re-assign Datetimes Columns
    datetimes = pd.to_datetime(data['datetimes'])
    original_data_dt = pd.DataFrame({
        'datetime': datetimes,
        'p_agg': np.ravel(p_agg),
        'p_agg_norm': np.ravel(p_agg_norm),
        **{f'p_apps_{app}': data['apps_data'][:, i] for i, app in enumerate(data['appliance_names'])},
        **{f'p_apps_norm_{app}': data['apps_norm'][:, i] for i, app in enumerate(data['appliance_names'])}})

    # Construct Datetimes Column for Predicted Data
    target_indices = data['test_indices'] + data['window_length'] // 2
    order = np.argsort(target_indices)
    pred_datetimes = datetimes[target_indices][order]

    pred_data_dt = pd.DataFrame({
        'datetime': pred_datetimes,
        **{f'p_apps_pred_{app}': apps_pred[:, i] for i, app in enumerate(data['appliance_names'])},
        **{f'p_apps_pred_norm_{app}': apps_pred_norm[:, i] for i, app in enumerate(data['appliance_names'])}})

    pred_data_dt = pred_data_dt.sort_values('datetime').reset_index(drop=True)

    merged_data_dt = pd.merge(original_data_dt, pred_data_dt, on='datetime', how='inner')

    # Compute Estimation Accuracy

    x_true = merged_data_dt['p_agg'].to_numpy()
    y_true = merged_data_dt[[f'p_apps_{app}' for app in data['appliance_names']]].to_numpy()
    y_pred = merged_data_dt[[f'p_apps_pred_{app}' for app in data['appliance_names']]].to_numpy()

    eacc_per_app = {app: compute_eacc(y_true[:, i], y_pred[:, i]) for i, app in enumerate(data['appliance_names'])}
    eacc_overall = compute_eacc(y_true, y_pred)

    # Save Results
    predictions = {
        'apps_pred': apps_pred,
        'apps_pred_norm': apps_pred_norm}

    np.savez_compressed(predictions_save_filepath, **predictions)

    eacc_df = pd.DataFrame(list(eacc_per_app.items()) + [('overall', eacc_overall)], columns=['appliance', 'eacc'])
    eacc_df.to_csv(results_eacc_save_filepath, index=False)

    # Visualize Results 
    plot_pred_versus_actual(x_true, y_pred, y_true, data['appliance_names'], title='Predicted versus Actual Appliance Power', save_filepath=results_plot_save_filepath)

if __name__ == '__main__':
    for s in settings:
        generate_model(**s)