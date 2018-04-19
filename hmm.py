import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#fh_loop_character_filter_map_hardcoded = {
#        0:[3.5, 1.5], #x
#        1:[3.5, 1.3], #y
#        4:[1.5, 2], #y
#        5:[9, 1.2] #z
#    }

def read_and_prepare(filenames):
    data_arrays = []
    maxtime = 0
    for filename in filenames:
        data_array = np.array(list(map(lambda l: list(map(float, list(filter(lambda x: len(x) > 0, re.split('\\s+', l))))), open(filename)))).T
        data_array[0, :] += maxtime
        maxtime = np.max(data_array[0, :])
        data_arrays.append(data_array)
    concated = np.concatenate(data_arrays, axis=1)
    return concated[0,:], concated[1:,:]


def get_pattern(patternfilename):
    pattern_data = np.array(
        list(map(lambda l: list(map(float, list(filter(lambda x: len(x) > 0, re.split('\\s+', l))))), open(patternfilename)))).T
    pattern_t, pattern = pattern_data[0, :], pattern_data[1:, :]
    pattern_t -= pattern_t[0]
    return pattern_t, pattern

def get_chars(np_marray):
    #maximum position
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

    #Pairwise Correlation

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
                filtered[-1][-1] = values[extrvals[i+1]]
                filtered.append([values[extrvals[i+1]], None])
        from functools import reduce
        return list(reduce(lambda res, x: res + x, filtered, []))

    return filter_extr_plain(extrvals, values)

def draw_signals(filteredt, datat, datas, name=""):
    minmaxdist = [[-50, 100]] * len(filteredt)

    plt.figure("signal_" + name)
    from functools import reduce
    newdatasarr = list(reduce(lambda res, x: res + [datat, x], datas, []))
    plt.plot(*newdatasarr)
    for i in range(len(filteredt)):
        plt.plot([filteredt[i][0], filteredt[i][0]], minmaxdist[i], '-r')
        plt.plot([filteredt[i][0] + 0.1*filteredt[i][1], filteredt[i][0] + 0.1*filteredt[i][1]], minmaxdist[i], '--g',linewidth=2)

    plt.grid()

def draw_patterns(time, sig, name=""):
    labeled = []
    for pat in sig:
        plt.figure("pattern_" + name)
        labeled += plt.plot(time, pat, label="blah" + str(len(labeled) + 1))
    plt.grid()
    plt.legend(handles=labeled)


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

    def set_signal_transform_cb(self, cb):
        self.signal_transform_cb = cb

    def set_calc_distance_cb(self, cb):
        self.calc_distance_cb = cb

    def calc_distances(self, signal, patterns_orig):
        patterns = self.signal_transform_cb(patterns_orig)

        transformed_len = len(patterns)
        pattern_len = patterns_orig.shape[1]
        signal_len = signal.shape[1]
        distances = []

        sample_deviation = int(pattern_len * self.window_deviation)
        for i in range(signal_len - int(pattern_len * (1 + self.window_deviation)) - 1):
            distances_in_window = []
            for d in range(-sample_deviation, sample_deviation + 1):
                window_size = pattern_len - d
                window = self.signal_transform_cb(signal[:, i:window_size + i])
                window_len = len(window)

                if window_size != pattern_len: #window_len != signal_len ??
                    told = [x / window_size for x in range(window_size)]
                    tnew = [x / pattern_len for x in range(pattern_len)]
                    window = np.array([interpolate_list(told, dw, tnew)[1] for dw in window])

                combined_pattern = np.reshape(patterns, 1 * transformed_len)#len(patterns) * pattern_len)
                combined_data = np.reshape(window, 1 * transformed_len)#len(patterns) * pattern_len)
                distance= self.calc_distance_cb(combined_pattern, combined_data)
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

    def do_filter(self):
        self.distances = self.calc_distances(self.signal_s, self.pattern_s)
        dtw_mins = argrelextrema(np.array([x[0] for x in self.distances]), np.less, order=4)[0]
        def filter_by_chars(signal, mindistances):
            filtered_dist = []
            filtered_chars = []

            for dist in mindistances:
                signal_len = int(dist[1])
                signal_start = int(dist[2])
                signal_candidate = signal[:, signal_start:signal_start + signal_len]

                sig_chars = [get_chars(s) for s in signal_candidate]
                if all(sig_chars):
                    sig_chars_map = dict(zip(self.significant_coords, sig_chars))
                else:
                    continue

                print("sig " + str(signal_start) + " " + str(
                    dict(zip(self.significant_coords, [get_chars(p) for p in signal_candidate]))))

                coeff = 1.2
                positive = 0
                for k in sig_chars_map.keys():
                    if abs(sig_chars_map[k]["std"] * coeff) >= abs(self.pattern_features[k]["std"]):
                        positive += 1
                        # if abs(sig_chars_map[k][1] * coeff) >= abs(fh_loop_character_filter_map[k][1]):
                        #    positive += 1

                if positive >= len(sig_chars_map):
                    filtered_dist.append(dist)
                filtered_chars.append((sig_chars_map, int(dist[2])))
            return filtered_dist, filtered_chars

        def remove_cross(ranges):

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

            print("LAST")
            # print(filtered_indexes)
            return filtered_indexes

        result = self.distances[np.r_[dtw_mins]]
        #result, features_first = filter_by_chars(self.signal_s, result)
        #result, features_second = filter_by_chars(self.signal_s, result)
        #result = remove_cross(result)
        #result, features_second = filter_by_chars(self.signal_s, result)
        #self.features_dict["filtered_signal_first"] = [(self.signal_t[f[1]], f[0]) for f in features_first]
        #self.features_dict["filtered_signal_second"] = [(self.signal_t[f[1]], f[0]) for f in features_second]

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
        fs_keys = signal_features[0][1].keys()
        features_s_map = dict(zip(fs_keys, [[]] * len(fs_keys)))
        for f in signal_features:
            for k, v in f[1].items():
                features_s_map[k] += [v]

        #for fs in fs_keys:
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
    basis = list(product([0,1,-1], repeat=dimension))
    basis.remove(tuple([0]*dimension))
    norm_signal = np.array([s / np.amax(s) for s in signal])

    diagram_map = dict(zip(basis,[0]*len(basis)))
    def push_sample(vect):
        cosines = { x: 1 - cosine(x, vect) for x in basis}
        angles =  { x: np.arccos(v) for x,v in cosines.items()}

        nearest_two_angles = sorted([a for a in angles.values()])[0:2]
        planes = { x:v for x,v in angles.items() if v in nearest_two_angles }

        vect_len = np.linalg.norm(vect)
        for k in planes.keys():
            cos = cosines[k]
            diagram_map[k] += vect_len * cos

    [push_sample(s) for s in norm_signal.T[:]]

    diagram = []
    for k in sorted(diagram_map.keys()):
        v = diagram_map[k]
        vect_len = np.linalg.norm(k)
        coeff = v / vect_len
        x = k[0] * coeff
        y = k[1] * coeff

        check = np.linalg.norm([x,y]) == v

        diagram.append([(x,y), v])

    global kkk
    kkk = kkk + 1
    draw_rose_vectors([k for k,v in diagram], num=str(kkk))
    return diagram

def get_rose_lengths(signal):
    return np.array([v for k,v in get_rose_diagram(signal)])

def draw_rose_vectors(vectors, num=""):
    maxpoint = np.amax(np.abs(vectors)) + 1
    point_num = len(vectors)
    dimension = len(vectors[0])
    draw = list(zip([[0]*dimension]*point_num, vectors))
    plt.figure('rose'+ num)
    start_point = [0]*dimension
    for d in vectors:
        plt.plot(*zip(start_point, d), '-*', linewidth=5)

    plt.xlim(-maxpoint, maxpoint)
    plt.ylim(-maxpoint, maxpoint)
    plt.grid()


if __name__ == "__main__":
    def find_pattern_and_draw(motion_pattern, motion_signal, significant_coords, label=""):
        motion_filter = MotionFilter(motion_pattern, motion_signal, significant_coords)
        diagram = get_rose_diagram(motion_filter.get_pattern()[1])
        rose_pattern_vectors = [k for k,v in diagram]
        draw_rose_vectors(rose_pattern_vectors)
        motion_filter.set_signal_transform_cb(get_rose_lengths)
        motion_filter.set_calc_distance_cb(euclidean)

        filteredt = motion_filter.do_filter()
        distances = motion_filter.get_distances()
        features = motion_filter.get_features()
        draw_features(features, label=label)
        draw_patterns(*motion_filter.get_pattern(), "max" + label)
        plt.figure("distances" + label)
        plt.plot(range(len(distances)), [x[0] for x in distances])
        plt.grid()
        print(len(filteredt))
        draw_signals(filteredt, *motion_filter.get_signal(), name="filtered" + label)


    significant_coords_list = [[0,1]]

    # pattern_stas_t, pattern_stas = get_pattern("pattern_stas.csv")
    # pattern_stas = pattern_stas[np.r_[significant_coords]]

    for significant_coords in significant_coords_list:
        pattern_t, pattern = get_pattern("pattern.csv")
        data_t, data_s = read_and_prepare(['max_fh_slice.csv',
                                            'merged.csv',
                                           # 'iracsv/merged.csv',
                                           # 'stasdrivecsv/merged.csv',
                                           # 'stasloopcsv/merged.csv'
                                           ])
        low = 1118
        high = 1140
        find_pattern_and_draw((pattern_t, pattern), (data_t[low:high], data_s[:, low:high]), significant_coords, label=str(significant_coords))

    plt.show()