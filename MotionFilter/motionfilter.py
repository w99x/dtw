import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def get_chars(np_marray):
    # maximum position
    features_dict = {}
    features_dict["std"] = np.std(np_marray)
    features_dict["var"] = np.var(np_marray)
    import scipy.stats as stats
    features_dict["iqr"] = stats.iqr(np_marray)
    features_dict["skew"] = stats.skew(np_marray)
    features_dict["kurtosis"] = stats.kurtosis(np_marray)
    features_dict["entropy"] = stats.entropy(np_marray)
    features_dict["zero_crossing"] = len(np.nonzero(np.diff(np_marray > 0))[0])
    features_dict["mean_crossing"] = len(np.nonzero(np.diff((np_marray - np.mean(np_marray)) > 0))[0])
    # Pairwise Correlation

    return features_dict

def filter_by_chars(signal, in_pattern, mindistances, transform=None):
    if transform is None:
        transform = lambda x: x
    filtered_dist = []

    pattern = transform(in_pattern)
    pattern_features = [get_chars(p) for p in pattern]

    for dist in mindistances:
        signal_len = int(dist[1])
        signal_start = int(dist[2])
        signal_candidate = transform(signal[:, signal_start:signal_start + signal_len])

        sig_chars = [get_chars(s) for s in signal_candidate]

        coeff = 1.2
        positive = 0
        for i, char in enumerate(sig_chars):
            if abs(char["std"] * coeff) >= abs(pattern_features[i]["std"]):
                positive += 1

        if positive >= len(sig_chars):
            filtered_dist.append(dist)
    return filtered_dist

def remove_cross(signal, pattern, ranges):
    ranges_local = ranges
    if len(ranges_local) <= 1:
        return ranges_local
    crossed = True
    filtered_indexes = []

    while crossed:
        crossed = False
        filtered_indexes = []
        zipped = list(zip(ranges_local[:-1], ranges_local[1:])) + [(ranges_local[-1], ranges_local[-1])]
        skip_next = False
        for range_pair in zipped:
            if skip_next:
                skip_next = False
                continue
            dist_val, signal_len, idx = range_pair[0]
            signal_len = int(signal_len)
            idx = int(idx)

            dist_val_next, signal_len_next, idx_next = range_pair[1]
            signal_len_next = int(signal_len_next)
            idx_next = int(idx_next)

            if idx + signal_len > idx_next:
                if (idx_next + signal_len_next <= idx + signal_len) or (
                        idx + signal_len - idx_next > int(0.5 * signal_len) + 1):
                    if dist_val_next < dist_val:
                        continue
                    else:
                        skip_next = True
                else:
                    crossed = True
                    signal_len -= idx + signal_len - idx_next

            filtered_indexes.append((dist_val, signal_len, idx))
        # print(filtered_indexes)
        ranges_local = filtered_indexes

    return filtered_indexes


class MotionFilter():
    pattern_t = None
    pattern_s = None

    signal_t = None
    signal_s = None

    pattern_features = None
    signal_features = None

    features_dict = {}

    window_deviation = 0.0
    distances = None
    significant_coords = [0]

    def __init__(self, pattern, signal, significant_coords=None, window_deviation=0.0):
        if significant_coords is not None:
            self.significant_coords = significant_coords

        self.pattern_t, self.pattern_s = pattern
        self.pattern_s = self.pattern_s[np.r_[significant_coords]]

        self.signal_t, self.signal_s = signal
        self.signal_s = self.signal_s[np.r_[significant_coords]]

        self.window_deviation = window_deviation
        self.pattern_features = dict(zip(self.significant_coords, [get_chars(p) for p in self.pattern_s]))
        self.features_dict["pattern"] = self.pattern_features

        self.signal_transform_cb = lambda x: x
        self.calc_distance_cb = lambda p, s: fastdtw(p, s)[0]
        self.filters_chain_list = [filter_by_chars, remove_cross]

    def set_filters_chain(self, filters_list):
        if filters_list is None:
            return
        self.filters_chain_list = filters_list

    def add_filter_to_chain(self, filter_cb):
        self.filters_chain_list.append(filter_cb)

    def set_signal_transform_cb(self, cb):
        if cb is None:
            return
        if isinstance(cb, list):
            def __transform_chain(signal):
                s = signal
                for t in cb:
                    s = t(s)
                return s

            self.signal_transform_cb = __transform_chain
        else:
            self.signal_transform_cb = cb

    def set_calc_distance_cb(self, cb):
        if cb is None:
            return
        self.calc_distance_cb = cb

    def __interpolate_list(xlist, ylist, xnew):
        from scipy import interpolate
        tck = interpolate.splrep(xlist, ylist, s=0)
        ynew = interpolate.splev(xnew, tck, der=0)
        return [xnew, ynew.tolist()]

    def calc_distances(self, signal, patterns_orig):
        transformed_patterns = self.signal_transform_cb(patterns_orig)

        orig_pattern_len = patterns_orig.shape[1]
        signal_len = signal.shape[1]
        distances = []

        sample_deviation = int(orig_pattern_len * self.window_deviation)
        for i in range(signal_len - int(orig_pattern_len * (1 + self.window_deviation))):
            distances_in_window = []
            for d in range(-sample_deviation, sample_deviation + 1):
                window_size = orig_pattern_len - d
                window = self.signal_transform_cb(signal[:, i:window_size + i])

                if window_size != orig_pattern_len:  # window_len != signal_len ??
                    told = [x / window_size for x in range(window_size)]
                    tnew = [x / orig_pattern_len for x in range(orig_pattern_len)]
                    window = np.array([__interpolate_list(told, dw, tnew)[1] for dw in window])

                combined_pattern = transformed_patterns.ravel()
                combined_data = window.ravel()
                distance = self.calc_distance_cb(combined_pattern, combined_data)
                distances_in_window.append((distance, window_size, i))

            dist_min = np.argmin(np.array([x[0] for x in distances_in_window]))
            distances.extend([tuple(x) for x in np.array(distances_in_window)[np.r_[dist_min]]])
        return np.array(distances)

    def get_signal(self):
        return self.signal_t, self.signal_s

    def get_pattern(self):
        return self.pattern_t, self.pattern_s

    def get_distances(self):
        return self.distances

    def get_features(self):
        return self.features_dict

    def __get_min_in_range(self, min_idxs, vals, min_range=1):
        minvals = vals[np.r_[min_idxs]]
        mins = min_idxs
        diffs = np.diff(mins) < min_range
        while diffs.any():
            new_mins = []
            for idx in list(range(len(diffs))):
                d = diffs[idx]
                if d:
                    new_mins.append(mins[idx]) if vals[mins[idx]] <= vals[mins[idx + 1]] else new_mins.append(
                        mins[idx + 1])
                else:
                    new_mins.append(mins[idx])
            mins = np.array(np.unique(new_mins))
            diffs = np.diff(mins) < min_range
            if len(diffs) == 0:
                break
        return mins

    def calc_features(self, distances):
        features = []
        for dist in distances:
            signal_len = int(dist[1])
            signal_start = int(dist[2])
            signal_candidate = self.signal_s[:, signal_start:signal_start + signal_len]
            sig_chars = [get_chars(s) for s in signal_candidate]
            features.append(sig_chars)
        return features

    def do_filter(self):
        self.distances = self.calc_distances(self.signal_s, self.pattern_s)
        distances_vals = np.array([x[0] for x in self.distances])
        dtw_mins = argrelextrema(distances_vals, np.less)[0]
        # dtw_mins = self.__get_min_in_range(dtw_mins, distances_vals, min_range=5)

        self.features_dict = dict()
        result = self.distances[np.r_[dtw_mins]]

        for i, filter in enumerate(self.filters_chain_list):
            result = filter(self.signal_s, self.pattern_s, result)
            self.features_dict["filter_" + filter.__name__ + "_" + str(i)] = self.calc_features(result)

        founds_pos = [(self.signal_t[int(dist[2])], int(dist[1])) for dist in result]
        return founds_pos
