import os
import numpy as np
from scipy.io import loadmat

# Data Generation Settings
T_limit = 1000
train_val_test_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 30
chunk_size = 1440 # One-Day
target_appliances = ['CWE', 'DWE', 'FRE', 'HPE', 'WOE', 'CDE', 'FGE', 'HTE']
rand_seed = 42
dataset_name = 'ampds_pattern-matching_mini'

pattern_matching_mini_settings = {
    'T_limit': 3000,
    'train_val_test_split': [0.7, 0.15, 0.15],
    'window_length': 30,
    'stride': 30,
    'chunk_size': 300,
    'target_appliances': ['CWE', 'DWE', 'FRE', 'HPE', 'WOE', 'CDE', 'FGE', 'HTE'],
    'rand_seed': 42,
    'dataset_name': 'ampds_pattern-matchin_mini'}

def generate_data(T_limit, train_val_test_split, window_length, stride, chunk_size, target_appliances, rand_seed, dataset_name):

    # Directories
    scripts_dir = os.path.dirname(__file__)
    base_dir = os.path.join(scripts_dir, '..')
    data_dir = os.path.join(base_dir, 'data')
    processed_data_dir = os.path.join(base_dir, 'processed_data')

    smart_meter_data_folderpath = os.path.join(data_dir, 'smart_meter_data')
    ampds_folderpath = os.path.join(smart_meter_data_folderpath, 'ampds')
    ampds_filepath = os.path.join(ampds_folderpath, 'ampds2.mat')

    # Fixed Settings - Constant for AMPds2 Data
    time_resolution = 60
    time_units = 'seconds'

    # Load Data
    data = loadmat(ampds_filepath)
    agg_data, apps_data = data['input'], data['output']
    agg_labels, apps_labels = data['labelInp'], data['labelOut']

    electrical_quantities = [label.strip() for label in agg_labels.flatten()]
    electrical_quantities = [quantity for quantity in electrical_quantities if quantity not in ("time", "id")]
    appliance_names = [label.strip() for label in apps_labels.flatten()]
    appliance_names = [name for name in appliance_names if name not in ("time", "id")]

    # Trim Data
    T = agg_data.shape[0]
    T = min(T, T_limit) if T_limit is not None else T
    agg_data, apps_data = agg_data[:T_limit], apps_data[:T_limit]
    apps_data = apps_data[:,:,0] # For appliances, keep only real power P

    # Select Target Appliances Only
    appliance_names = target_appliances if target_appliances is not None else appliance_names
    appliance_indices = [appliance_names.index(name) for name in appliance_names]
    column_indices = [0, 1] + [i + 2 for i in appliance_indices]

    apps_data = apps_data[:, column_indices]
    apps_labels = apps_labels[column_indices]

    # Get Appliance Statistics
    def determine_on_threshold(apps, upper_percentile=99):

        thresholds = np.zeros(apps.shape[1], dtype=float)

        for i in range(apps.shape[1]):
            values = apps[:, i]

            # For constant profile, keep threshold as zero.
            if np.all(values == values[0]): continue

            # Limit the influence of rare high readings when selecting the split.
            cap = np.percentile(values, upper_percentile)
            values = np.minimum(values, cap)

            levels, counts = np.unique(values, return_counts=True)
            weights_left = np.cumsum(counts)[:-1]
            weights_right = values.size - weights_left

            means_left = np.cumsum(levels * counts)[:-1] / weights_left
            means_right = (np.sum(levels * counts) - np.cumsum(levels * counts)[:-1]) / weights_right

            between_class_variance = (weights_left * weights_right * (means_left - means_right) ** 2)
            
            split = np.argmax(between_class_variance)
            thresholds[i] = max(0.0, (levels[split] + levels[split + 1]) / 2)

        return thresholds

    def on_percentages(apps, thresholds):
        is_on = apps > thresholds[np.newaxis, :]
        return is_on.mean(axis=0) * 100

    def get_appliance_metrics(apps, on_thresholds):
        return {
            "apps_min": np.min(apps, axis=0),
            "apps_max": np.max(apps, axis=0),
            "apps_mean": np.mean(apps, axis=0),
            "apps_median": np.median(apps, axis=0),
            "apps_std": np.std(apps, axis=0),
            'apps_on_threshold': on_thresholds,
            'apps_on_percentrage': on_percentages(apps, on_thresholds)}

    appliance_on_thresholds = determine_on_threshold(apps_data[:, 2:]) # Exclude 'id' and 'time' columns.
    appliance_metrics = get_appliance_metrics(apps_data[:, 2:], appliance_on_thresholds)

    # Normalize Data
    def normalize_data(x):
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    # Separate ID and time columns, normalize appliance columns, then reattach.
    agg_norm = np.column_stack((agg_data[:, :2], normalize_data(agg_data[:, 2:])))
    apps_norm = np.column_stack((apps_data[:, :2], normalize_data(apps_data[:, 2:])))

    # Create Training, Validation, and Testing Datasets
    num_chunks = T // chunk_size

    def precompute_indices(
        num_timesteps,
        window_length,
        stride,
        train_val_test_split,
        num_chunks,
        seed=42):

        expected_num_windows = max(0, (num_timesteps - window_length) // stride + 1)

        # Divide the timeline into contiguous, non-overlapping chunks.
        boundaries = np.linspace(0, num_timesteps, num_chunks + 1, dtype=int)
        chunk_ids = np.random.default_rng(seed).permutation(num_chunks)

        # Allocate whole chunks to splits, as close as possible to the requested ratios.
        exact_counts = np.asarray(train_val_test_split) * num_chunks
        chunk_counts = np.floor(exact_counts).astype(int)
        remainder = num_chunks - chunk_counts.sum()
        chunk_counts[np.argsort(-(exact_counts - chunk_counts))[:remainder]] += 1

        split_names = ("train", "val", "test")
        idx_dict = {}
        chunk_offset = 0

        for split_name, chunk_count in zip(split_names, chunk_counts):
            split_chunk_ids = chunk_ids[chunk_offset:chunk_offset + chunk_count]
            chunk_offset += chunk_count

            inp_parts = []
            out_parts = []

            for chunk_id in split_chunk_ids:
                chunk_start = boundaries[chunk_id]
                chunk_end = boundaries[chunk_id + 1]

                # Keep only windows fully contained in this chunk.
                starts = np.arange(
                    chunk_start,
                    chunk_end - window_length + 1,
                    stride,
                )
                inp_parts.append(starts)
                out_parts.append(starts + window_length // 2)

            inp_idx = np.concatenate(inp_parts) if inp_parts else np.array([], dtype=int)
            out_idx = np.concatenate(out_parts) if out_parts else np.array([], dtype=int)

            # Shuffle examples within each split.
            perm = np.random.default_rng(seed + len(idx_dict)).permutation(len(inp_idx))
            idx_dict[split_name + '_indices'] = inp_idx[perm]

        num_windows = sum(len(indices) for indices in idx_dict.values())
        return expected_num_windows, num_windows, idx_dict

    expected_num_windows, num_windows, idx_dict = precompute_indices(T, window_length, stride, train_val_test_split, num_chunks, seed=rand_seed)

    # Recompute the train/validation/test proportions from the assigned windows.
    split_names = ("train_indices", "val_indices", "test_indices")
    total = sum(len(idx_dict[name]) for name in split_names)

    real_train_val_test_split = [len(idx_dict[name]) / total for name in split_names]

    # calculate real number of unique timestamps used in whole dataset
    all_starts = np.concatenate([idx_dict[name] for name in split_names])
    all_timestamps = np.unique(
        np.concatenate([
            np.arange(start, start + window_length)
            for start in all_starts]))
    real_num_unique_timestamps = len(all_timestamps)

    # Save Dataset
    dataset_path = os.path.join(processed_data_dir, dataset_name + '.npz')

    np.savez_compressed(
        dataset_path,
        agg_labels=agg_labels,
        apps_labels=apps_labels,
        electrical_quantities=electrical_quantities,
        appliance_names=appliance_names,
        agg_data=agg_data,
        apps_data=apps_data,
        agg_norm=agg_norm,
        apps_norm=apps_norm,
        **appliance_metrics,
        **idx_dict,
        train_val_test_split=real_train_val_test_split,
        T=real_num_unique_timestamps,
        time_resolution=time_resolution,
        time_units=time_units,
        window_length=window_length,
        stride=stride,
        chunk_size=chunk_size,
        num_chunks=num_chunks)

generate_data(**pattern_matching_mini_settings)