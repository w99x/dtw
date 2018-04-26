import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# fh_loop_character_filter_map_hardcoded = {
#        0:[3.5, 1.5], #x
#        1:[3.5, 1.3], #y
#        4:[1.5, 2], #y
#        5:[9, 1.2] #z
#    }

def read_and_prepare(filenames):
    data_arrays = []
    maxtime = 0
    for filename in filenames:
        data_array = np.array(list(
            map(lambda l: list(map(float, list(filter(lambda x: len(x) > 0, re.split('\\s+', l))))), open(filename)))).T
        data_array[0, :] += maxtime
        maxtime = np.max(data_array[0, :])
        data_arrays.append(data_array)
    concated = np.concatenate(data_arrays, axis=1)
    return concated[0, :], concated[1:, :]


def get_pattern(patternfilename):
    pattern_data = np.array(
        list(map(lambda l: list(map(float, list(filter(lambda x: len(x) > 0, re.split('\\s+', l))))),
                 open(patternfilename)))).T
    pattern_t, pattern = pattern_data[0, :], pattern_data[1:, :]
    pattern_t -= pattern_t[0]
    return pattern_t, pattern


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


def interpolate_list(xlist, ylist, xnew):
    from scipy import interpolate
    tck = interpolate.splrep(xlist, ylist, s=0)
    ynew = interpolate.splev(xnew, tck, der=0)
    return [xnew, ynew.tolist()]


def filter_extr_plain_draw(distances, values, threshold=0.8):
    extr = argrelextrema(np.array(distances), np.less)
    extrvals = extr[0]

    def filter_extr_plain(extrspos, values, threshold=threshold):
        filtered = [[values[extrspos[0]], None]]
        for i in range(len(extrvals) - 1):
            last = filtered[-1][0]
            if (values[extrspos[i + 1]] - last >= threshold):
                filtered[-1][-1] = values[extrvals[i + 1]]
                filtered.append([values[extrvals[i + 1]], None])
        from functools import reduce
        return list(reduce(lambda res, x: res + x, filtered, []))

    return filter_extr_plain(extrvals, values)


def draw_signals(filteredt, datat, datas, label=""):
    minmaxdist = [[-50, 100]] * len(filteredt)
    
    from functools import reduce
    newdatasarr = list(reduce(lambda res, x: res + [datat, x], datas, []))
    plt.plot(*newdatasarr, label=label)
    for i in range(len(filteredt)):
        plt.plot([filteredt[i][0], filteredt[i][0]], minmaxdist[i], '-r')
        plt.plot([filteredt[i][0] + 0.1 * filteredt[i][1], filteredt[i][0] + 0.1 * filteredt[i][1]], minmaxdist[i],
                 '--g', linewidth=2)

    plt.grid()


def draw_patterns(time, sig, name=""):
    labeled = []
    for pat in sig:
        plt.figure("pattern_" + name)
        labeled += plt.plot(time, pat, label="blah" + str(len(labeled) + 1))
    plt.grid()
    plt.legend(handles=labeled)


def filter_by_chars(signal, pattern, mindistances):
    filtered_dist = []

    pattern_features = [get_chars(p) for p in pattern]

    for dist in mindistances:
        signal_len = int(dist[1])
        signal_start = int(dist[2])
        signal_candidate = signal[:, signal_start:signal_start + signal_len]

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
        self.signal_transform_cb = cb

    def set_calc_distance_cb(self, cb):
        if cb is None:
            return
        self.calc_distance_cb = cb

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
                    window = np.array([interpolate_list(told, dw, tnew)[1] for dw in window])

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
        dtw_mins = argrelextrema(distances_vals, np.less, order=3)[0]
        # dtw_mins = self.__get_min_in_range(dtw_mins, distances_vals, min_range=5)

        self.features_dict = dict()
        result = self.distances[np.r_[dtw_mins]]
        for i, filter in enumerate(self.filters_chain_list):
            result = filter(self.signal_s, self.pattern_s, result)
            self.features_dict["filter_" + filter.__name__ + "_" + str(i)] = self.calc_features(result)

        founds_pos = [(self.signal_t[int(dist[2])], int(dist[1])) for dist in result]
        return founds_pos


def draw_features(features_map, label=""):
    pattern_key = 'pattern'
    pattern_features = features_map[pattern_key]

    for feature_key, signal_features in features_map.items():
        if feature_key == pattern_key:
            continue
        plt.figure("features_" + feature_key + label)
        features_t = [x[0] for x in signal_features]
        fs_keys = list(range(len(signal_features[0])))
        features_s_map = dict(zip(fs_keys, [[] for _ in fs_keys]))
        for f in signal_features:
            for k, v in enumerate(f):
                features_s_map[k].append(v)

        # for fs in fs_keys:
        #    features_s_map[fs] = list(zip(*features_s_map[fs]))

        for k, vals in pattern_features.items():
            features_number = len(vals)
            subplot_index = 0
            for vkey, val in vals.items():
                sublabel = str(vkey)
                plt.subplot(features_number * 100 + 11 + subplot_index)
                subplot_index += 1
                labeled = []
                labeled += plt.plot([data_t[0], data_t[-1]], [val, val], '-', label=sublabel)
                features_to_draw = []
                for feature in features_s_map[k]:
                    features_to_draw.append(feature[vkey])
                labeled += plt.plot(features_t, features_to_draw, '*', label=sublabel)
                plt.grid()
                plt.legend(handles=labeled)


kkk = 0


def get_rose_diagram(signal):
    from scipy.spatial.distance import cosine
    dimension = len(signal)
    from itertools import product
    basis = list(product([0, 1, -1], repeat=dimension))
    basis.remove(tuple([0] * dimension))
    norm_signal = np.array([s / np.amax(np.abs(s)) for s in signal])

    diagram_map = dict(zip(basis, [0] * len(basis)))

    def push_sample(vect):
        cosines = {x: 1 - cosine(x, vect) for x in basis}
        angles = {x: np.arccos(v) for x, v in cosines.items()}

        nearest_two_angles = sorted([a for a in angles.values()])[0:2]
        planes = {x: v for x, v in angles.items() if v in nearest_two_angles}

        vect_len = np.linalg.norm(vect)
        for k in planes.keys():
            cos = cosines[k]
            diagram_map[k] += vect_len * cos

    [push_sample(s) for s in norm_signal.T[:]]

    # [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    weights = [10.0, 10.0, 10.0, 1.5, 1, 1.5, 1.3, 1]
    diagram = []
    sorted_keys = sorted(diagram_map.keys())
    for i, k in enumerate(sorted_keys):
        v = diagram_map[k] * weights[i]
        vect_len = np.linalg.norm(k)
        coeff = v / vect_len
        x = k[0] * coeff
        y = k[1] * coeff

        check = np.linalg.norm([x, y]) == v

        diagram.append([(x, y), v])

    global kkk
    kkk = kkk + 1
    # if kkk != 1:
    #    draw_rose_vectors([k for k,v in diagram], num=str(kkk))
    return diagram


def signal_modul(signal):
    lengths = np.array([np.linalg.norm(s) for s in signal.T])
    return np.ndarray(shape=(1, len(lengths)), buffer=lengths)


def get_rose_lengths(signal):
    lengths = np.array([v for k, v in get_rose_diagram(signal)])
    return np.ndarray(shape=(1, len(lengths)), buffer=lengths)


def draw_rose_vectors(vectors, num=""):
    maxpoint = np.amax(np.abs(vectors)) + 1
    point_num = len(vectors)
    dimension = len(vectors[0])
    draw = list(zip([[0] * dimension] * point_num, vectors))
    plt.figure('rose' + num)
    start_point = [0] * dimension
    for d in vectors:
        plt.plot(*zip(start_point, d), '-*', linewidth=5)

    plt.xlim(-maxpoint, maxpoint)
    plt.ylim(-maxpoint, maxpoint)
    plt.grid()


def tresholded_euclidean(u, v):
    threshold = 5.0
    e = euclidean(u, v)
    if e > threshold:
        e = 100.0
    return e

def draw_processed_subplot(data, time, signal):
    plt.figure("ranges")
    subplot_index = 0
    for k, (filteredt, distances, features) in data.items():
        plt.subplot(len(data) * 100 + 11 + subplot_index)
        subplot_index += 1
        draw_signals(filteredt, time, signal[np.r_[k]])

    plt.figure("distances")
    subplot_index = 0
    for k, (filteredt, distances, features) in data.items():
        plt.subplot(len(data) * 100 + 11 + subplot_index)
        subplot_index += 1
        plt.plot(range(len(distances)), [x[0] for x in distances])
        plt.grid()

    #[draw_features(features) for k, (filteredt, distances, features) in data.items()]


if __name__ == "__main__":
    def find_pattern(motion_filter):
        filteredt = motion_filter.do_filter()
        distances = motion_filter.get_distances()
        features = motion_filter.get_features()

        return filteredt, distances, features


    # pattern_stas_t, pattern_stas = get_pattern("pattern_stas.csv")
    # pattern_stas = pattern_stas[np.r_[significant_coords]]

    pattern_t, pattern = get_pattern("pattern.csv")
    data_t, data_s = read_and_prepare([  # 'max_fh_slice.csv',
        'merged.csv',
        # 'iracsv/merged.csv',
        # 'stasdrivecsv/merged.csv',
        # 'stasloopcsv/merged.csv'
    ])


    def append_result(proc_data, sign_coords, transform_cb=None):
        for significant_coords in sign_coords:
            motion_filter = MotionFilter((pattern_t, pattern), (data_t, data_s), significant_coords)
            motion_filter.set_signal_transform_cb(transform_cb)
            proc_data[tuple(significant_coords)] = find_pattern(motion_filter)


    processed_data = dict()
    #append_result(processed_data, [[0], [1], [2], [3], [4], [5]])
    append_result(processed_data, [[0, 1, 2], [3, 4, 5]])
    #append_result(processed_data, [[0, 1]], get_rose_lengths)

    # diagram = get_rose_diagram(motion_filter.get_pattern()[1])
    # rose_pattern_vectors = [k for k,v in diagram]
    # draw_rose_vectors(rose_pattern_vectors, num="pattern")
    # motion_filter.set_signal_transform_cb(signal_modul)
    # motion_filter.set_calc_distance_cb(tresholded_euclidean)

    draw_patterns(pattern_t, pattern, "max")
    draw_processed_subplot(processed_data, data_t, data_s)
    plt.show()
