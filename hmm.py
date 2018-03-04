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
    minimumsidx = argrelextrema(np_marray, np.less)
    maximumsidx = argrelextrema(np_marray, np.greater)

    def get_vals_by_idx(idxs):
        return [np_marray[x] for x in idxs]

    minimums = get_vals_by_idx(minimumsidx[0])
    maximums = get_vals_by_idx(maximumsidx[0])
    if len(maximums) == 0 or len(minimums) == 0:
        return None

    minmax = np.max(minimums)
    minmin = np.min(minimums)

    maxmin = np.min(maximums)
    maxmax = np.max(maximums)

    maxmax -= minmin
    maxmin -= minmin
    minmin -= maxmax
    minmax -= maxmax
    maxratio = maxmax / maxmin
    minratio = minmin / minmax


    std = np.std(np_marray)
    var = np.var(np_marray)

    return [std, var]

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

def get_dtw_in_window(window, patterns, window_deviation=0.0):
    pattern_len = len(patterns[0])
    windows_len = len(window[0])
    distances = []

    sample_deviation = int(pattern_len*window_deviation)
    for i in range(windows_len - int(pattern_len * (1 + window_deviation)) - 1):
        distances_in_window = []
        for d in range(-sample_deviation, sample_deviation+1):
            window_size = pattern_len - d
            datawin = window[:, i:window_size + i][0]

            told = [x/window_size for x in range(window_size)]
            tnew = [x/pattern_len for x in range(pattern_len)]
            datawin_resampled = interpolate_list(told,datawin, tnew)[1]

            distance, path = fastdtw(patterns[0], datawin)
            distances_in_window.append((distance, window_size, i))

        #dist_mins = argrelextrema(np.array([x[0] for x in distances_in_window]), np.less)[0]
        dist_min = np.argmin(np.array([x[0] for x in distances_in_window]))
        distances.extend([tuple(x) for x in np.array(distances_in_window)[np.r_[dist_min]]])
    return distances


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
    for pat in sig:
        plt.figure("pattern_" + name)
        plt.plot(time, pat)
        plt.grid()


fh_loop_character_filter_map_hardcoded = {
        0:[3.5, 1.5], #x
        1:[3.5, 1.3], #y
        4:[1.5, 2], #y
        5:[9, 1.2] #z
    }

significant_coords = [0]

pattern_t, pattern = get_pattern("pattern.csv")
pattern = pattern[np.r_[significant_coords]]

#pattern_stas_t, pattern_stas = get_pattern("pattern_stas.csv")
#pattern_stas = pattern_stas[np.r_[significant_coords]]

fh_loop_character_filter_map = dict(zip(significant_coords, [get_chars(p) for p in pattern]))

def filter_extr(distances, window_t, window_s):
    founds_pos = []

    dtw_mins = argrelextrema(np.array([x[0] for x in distances]), np.less)[0]
    for dtw_min in zip(dtw_mins[:-1], dtw_mins[1:]):
        dist_val = distances[dtw_min[0]][0]
        signal_len = int(distances[dtw_min[0]][1])
        idx = int(distances[dtw_min[0]][2])


        dist_val_next = distances[dtw_min[1]][0]
        idx_next = int(distances[dtw_min[1]][2])
        signal_len_next = int(distances[dtw_min[1]][1])

        if idx + signal_len > idx_next:
            if (idx_next + signal_len_next < idx + signal_len) or (idx + signal_len - idx_next > int(0.5 * signal_len)):
                if dist_val_next < dist_val:
                    idx = idx_next
                    signal_len = signal_len_next
            else:
                signal_len -= idx + signal_len - idx_next


        time = window_t[idx]
        signal = window_s[:, idx:idx + signal_len]

        sig_chars = [get_chars(s) for s in signal]
        if all(sig_chars):
            sig_chars_map = dict(zip(significant_coords, sig_chars))
        else:
            continue

        coeff = 1.2
        positive = 0
        for k in sig_chars_map.keys():
            if abs(sig_chars_map[k][0]*coeff) >= abs(fh_loop_character_filter_map[k][0]):
                positive += 1
                #if abs(sig_chars_map[k][1] * coeff) >= abs(fh_loop_character_filter_map[k][1]):
                #    positive += 1

        if positive >= len(sig_chars_map):
            founds_pos.append((time, signal_len))
    return founds_pos

draw_patterns(pattern_t, pattern, "max")
#draw_patterns(pattern_stas_t, pattern_stas)

newdatat, newdatas = read_and_prepare(['merged_42_44.csv',
                                       #'iracsv/merged.csv',
                                       #'stasdrivecsv/merged.csv',
                                       #'stasloopcsv/merged.csv'
                                       ])

newdatas = newdatas[np.r_[significant_coords]]

distances = get_dtw_in_window(newdatas, pattern, window_deviation=0.0)
plt.figure("distances")
plt.plot(range(len(distances)), [x[0] for x in distances])
plt.grid()

filteredt = filter_extr(distances, newdatat, newdatas)
draw_signals(filteredt, newdatat, newdatas, name="filtered")

plt.show()