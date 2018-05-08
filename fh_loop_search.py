import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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



def draw_signals(filteredt, datat, datas, label=""):
    minmaxdist = [[np.amin(datas) - 1, np.amax(datas) + 1]] * len(filteredt)
    
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
        adjusted_fig("pattern_" + name)
        labeled += plt.plot(time, pat, label="blah" + str(len(labeled) + 1))
    plt.grid()
    plt.legend(handles=labeled)




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

    #        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    weights = [10.0,     10.0,     10.0,   1,       1,      1,       1,      1]
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
    #kkk = kkk + 1
    #if kkk != 1:
    #    draw_rose_vectors([k for k,v in diagram], num=str(kkk))
    return diagram


def signal_modul(signal):
    lengths = np.array([np.linalg.norm(s) for s in signal.T])
    return np.ndarray(shape=(1, len(lengths)), buffer=lengths)


def get_rose_lengths(signal):
    lengths = np.array([v for k, v in get_rose_diagram(signal)])
    return np.ndarray(shape=(1, len(lengths)), buffer=lengths)

def get_rose_lengths_from_diagram(diagram):
    lengths = np.array([v for k, v in diagram])
    return np.ndarray(shape=(1, len(lengths)), buffer=lengths)

def get_rose_coords(diagram):
    return [k for k,v in diagram]



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

def adjusted_fig(title):
    return plt.figure(title).subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0, wspace=0, hspace=0)

def draw_processed_subplot(data, time, signal, name=""):
    adjusted_fig("ranges" + name)
    subplot_index = 0
    for k, (filteredt, distances, features) in data:
        plt.subplot(len(data) * 100 + 11 + subplot_index)
        plt.title(str(k), x=0.1, y=0.5, fontsize=10)
        subplot_index += 1
        if signal.shape[0] < max(k):
            draw_signals(filteredt, time, signal)
        else:
            draw_signals(filteredt, time, signal[np.r_[k]])

    adjusted_fig("distances" + name)
    subplot_index = 0
    for k, (filteredt, distances, features) in data:
        plt.subplot(len(data) * 100 + 11 + subplot_index)
        plt.title(str(k), x=0.1, y=0.5, fontsize=10)
        subplot_index += 1
        plt.plot(range(len(distances)), [x[0] for x in distances])
        plt.grid()

    #[draw_features(features) for k, (filteredt, distances, features) in data]

def normalize_window(signal):
    return np.array([s / np.amax(np.abs(s)) for s in signal])

def filter_by_chars_normaized(p,s,m):
    return filter_by_chars(p,s,m, transform=normalize_window)

def filter_by_chars_moduled(p,s,m):
    return filter_by_chars(p,s,m, transform=signal_modul)

def get_distances(pattern_in, signal_in):
    norm_pattern = normalize_window(pattern_in)
    norm_signal = normalize_window(signal_in)
    distances_dict = {}
    distances_dict["dtw_coords"] = [fastdtw(norm_pattern[i], norm_signal[i])[0] for i, _ in enumerate (norm_pattern)]

    norm_module_pattern = normalize_window(signal_modul(pattern_in))
    norm_module_signal = normalize_window(signal_modul(signal_in))
    distances_dict["dtw_modul"] = fastdtw(norm_module_pattern, norm_module_signal)[0]
    if signal_in.shape[0] >= 2:
        pattern_diagram = get_rose_diagram(pattern_in[:2])
        signal_diagram = get_rose_diagram(signal_in[:2])
        distances_dict["rose"] = euclidean(get_rose_lengths_from_diagram(pattern_diagram), get_rose_lengths_from_diagram(signal_diagram))
        #draw_rose_vectors(get_rose_coords(signal_diagram), num=str(signal_in[:2]))
    return distances_dict

def filter_by_distances(signal, pattern, mindistances):
    thresholds = dict()
    thresholds["dtw_modul"] = 3
    thresholds["rose"] = 6
    filtered_dist = []
    for dist in mindistances:
        signal_len = int(dist[1])
        signal_start = int(dist[2])
        dist_value = dist[0]
        signal_candidate = signal[:, signal_start:signal_start + signal_len]

        sig_chars = get_distances(pattern, signal_candidate)
        if all([sig_chars[k] <= t for k,t in thresholds.items()]):
            filtered_dist.append(dist)
    return filtered_dist

def filter_by_dist_treshold(signal, pattern, mindistances):
    dist_val_threshold = 6.0
    filtered_dist = []
    for dist in mindistances:
        if dist[0] <= dist_val_threshold:
            filtered_dist.append(dist)
    return filtered_dist

def scale(s):
    from sklearn import preprocessing
    return preprocessing.scale(s.T).T

def ravel_distance(s, p):
    return fastdtw(s.ravel(), p.ravel())[0]

if __name__ == "__main__":
    def find_pattern(motion_filter):
        filteredt = motion_filter.do_filter()
        distances = motion_filter.get_distances()
        features = motion_filter.get_features()

        return filteredt, distances, features


    pattern_stas_t, pattern_stas = get_pattern("pattern_stas.csv")

    pattern_t, pattern = get_pattern("pattern.csv")

    data_t, data_s = read_and_prepare([  # 'max_fh_slice.csv',
        'merged.csv',
#        'iracsv/merged.csv',
#        'stasdrivecsv/merged.csv',
#        'stasloopcsv/merged.csv'
    ])

    l = 0
    h = len(data_t)#l+26

    data_t = data_t[l:h]
    data_s = data_s[:, l:h]
    def append_result(proc_data, sign_coords, transform_cb=None, measure=None, window_deviation=0.0, filters=None):
        from MotionFilter import motionfilter
        for significant_coords in sign_coords:
            motion_filter = motionfilter.MotionFilter((pattern_t, pattern), (data_t, data_s), significant_coords, window_deviation=window_deviation)
            motion_filter.set_signal_transform_cb(transform_cb)
            motion_filter.set_calc_distance_cb(measure)
            motion_filter.set_filters_chain(filters)
            proc_data.append((significant_coords, find_pattern(motion_filter)))


    processed_data = list()
    append_result(processed_data, [[0]])

    draw_patterns(pattern_t, scale(pattern), "max")
    draw_patterns(pattern_stas_t, scale(pattern_stas), "stas")
    draw_processed_subplot(processed_data, data_t, scale(data_s))

    plt.show()
