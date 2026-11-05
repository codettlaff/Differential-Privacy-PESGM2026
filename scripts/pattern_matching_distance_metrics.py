import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import pickle

# Directories
notebooks_dir = os.path.dirname(__file__)
base_dir = os.path.join(notebooks_dir, '..')
processed_data_dir = os.path.join(base_dir, 'processed_data')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Settings
dataset_name = 'ampds_dataset_full'

# Load Data
processed_data_filepath = os.path.join(processed_data_dir, dataset_name + '.npz')
data = np.load(processed_data_filepath)

window_length, stride = data['window_length'], data['stride']
agg_labels = data['agg_labels']
apps_labels = data['apps_labels']
electrical_quantities = data['electrical_quantities']
appliance_names = data['appliance_names']

agg_norm = data['agg_norm']

train_indices = data['train_indices']
val_indices = data['val_indices']
test_indices = data['test_indices']
all_indices = np.concatenate((train_indices, val_indices, test_indices))
query_indices = np.concatenate((val_indices, test_indices))

# Feature Vectors
def compute_feature_vector_2D(x_win, feature_selection='all'):
    
    funcs = {
        'mean': lambda x: np.mean(x, axis=0),
        'min': lambda x: np.min(x, axis=0),
        'max': lambda x: np.max(x, axis=0),
        'std': lambda x: np.std(x, axis=0),
        'rms': lambda x: np.mean(x**2, axis=0),
        'energy': lambda x: np.sum(x**2, axis=0)}

    if feature_selection == 'all': feature_selection = list(funcs)
    return np.column_stack([funcs[name](x_win) for name in feature_selection])

def feature_distance(vector1, vector2): return np.linalg.norm(vector1 - vector2, ord='fro')

feature_lookup = {
    int(i): compute_feature_vector_2D(agg_norm[i : i + window_length, 2:])
    for i in all_indices}

feature_lookup_filepath = os.path.join(results_dir, dataset_name + '_feature_lookup.pkl')
with open(feature_lookup_filepath, 'wb') as f: pickle.dump(feature_lookup, f)

# Feature Distances
feature_distance_lookup = {}

for query_idx in tqdm(query_indices, desc='Computing Feature Distances'):
    query_features = feature_lookup[query_idx]

    for template_idx, template_features in feature_lookup.items():
        feature_distance_lookup[(int(template_idx), int(query_idx))] = float(
            feature_distance(template_features, query_features))

feature_distance_lookup_filepath = os.path.join(results_dir, dataset_name + '_feature_distance_lookup.pkl')
with open(feature_distance_lookup_filepath, 'wb') as f: pickle.dump(feature_distance_lookup, f)

# Pearson Correlation
def compute_correlation(frame1, frame2):
    
    correlations = [
        np.corrcoef(frame1[:, i], frame2[:, i])[0, 1]
        for i in range(frame1.shape[1])]

    return np.mean(np.abs(correlations))

pearson_correlation_lookup = {}

for query_idx in tqdm(query_indices, desc='Computing Pearson Correlation'):
    query_frame = agg_norm[query_idx : query_idx + window_length, 2:]

    for template_idx in train_indices:
        template_frame = agg_norm[template_idx : template_idx + window_length, 2:]
        pearson_correlation_lookup[(int(template_idx), int(query_idx))] = float(
            compute_correlation(template_frame, query_frame))

pearson_correlation_lookup_filepath = os.path.join(results_dir, dataset_name + '_pearson_correlation_lookup.pkl')
with open(feature_distance_lookup_filepath, 'wb') as f: pickle.dump(pearson_correlation_lookup, f)

# Dynamic Time Warping
def compute_dtw(frame1, frame2):
    distance, path = fastdtw(
        frame1,
        frame2,
        dist=lambda x, y: np.linalg.norm(x - y))
    return distance, path

# Compare every validation/test frame with every training template
dtw_distance_lookup = {}

for query_idx in tqdm(query_indices, desc='Computing DTW Distance'):
    query_frame = agg_norm[query_idx : query_idx + window_length, 2:]

    for template_idx in train_indices:
        template_frame = agg_norm[template_idx : template_idx + window_length, 2:]
        distance, _ = compute_dtw(template_frame, query_frame)
        dtw_distance_lookup[(int(template_idx), int(query_idx))] = float(distance)

dtw_distance_lookup_filepath = os.path.join(results_dir, dataset_name + '_dtw_distance_lookup.pkl')
with open(dtw_distance_lookup_filepath, 'wb') as f: pickle.dump(dtw_distance_lookup, f)

# Consolidate Distance Metrics
combined_lookup = {}

for key in feature_distance_lookup:
    template_idx, query_idx = key
    if key in pearson_correlation_lookup and key in dtw_distance_lookup:
        combined_lookup[key] = {
            "feature_distance": float(feature_distance_lookup[key]),
            "pearson_correlation": float(pearson_correlation_lookup[key]),
            "dtw_distance": float(dtw_distance_lookup[key])}

combined_lookup_filepath = os.path.join(results_dir, dataset_name + '_combined_distance_metric_lookup.pkl')
with open(combined_lookup_filepath, 'wb') as f: pickle.dump(combined_lookup, f)
