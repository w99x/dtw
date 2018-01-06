import os
from scipy import interpolate

def interpolate_lists(xlist, ylists, xnew):
    interpolated = map(lambda x: interpolate_list(xlist, x, xnew), ylists)
    return interpolated


def interpolate_list(xlist, ylist, xnew):
    tck = interpolate.splrep(xlist, ylist, s=0)
    ynew = interpolate.splev(xnew, tck, der=0)
    return [xnew, ynew.tolist()]

def read_csv(filename, ticks_per_sec=1000000000, col_start=0, cols_size=4):
    import csv
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=' ', lineterminator='\n')
        list_csv = list(reader)
        starttime = float(list_csv[col_start][0])
        return map(lambda x: [int(ticks_per_sec * (float(x[col_start]) - starttime))] + map(float, x[col_start + 1:col_start + cols_size]), list_csv)

def merge_list_pair(merge_to, merge_from):
    if merge_to is None:
        return merge_from

    merge_from_map = {x[0]: list(x[1:]) for x in merge_from}
    merged = []
    for m in merge_to:
        merged.append(list(m) + merge_from_map[m[0]])
    return merged

def merge_lists_zip(lists):
    return reduce(lambda res, x: merge_list_pair(res, zip(*x)), lists, None)

def merge_lists(lists):
    return reduce(lambda res, x: merge_list_pair(res, x), lists, None)

def merge_files_content(files, folder, ticks_per_sec=1000000000, col_start=0, cols_size=4):
    files = map(lambda x: os.path.join(folder, x), files)

    csv_maps_list = map(lambda x: read_csv(x, ticks_per_sec, col_start, cols_size), files)
    csv_maps_list = map(lambda x: map(list, zip(*x)), csv_maps_list)
    xnew = csv_maps_list[0][0]
    scale = 100000000
    newticks = range(0, (xnew[-1] - xnew[0]) / scale)
    newticks = [x*scale + xnew[0] for x in newticks]
    interpolated = map(lambda x: interpolate_lists(x[0], x[1:], newticks), csv_maps_list)
    interpolated = map(merge_lists_zip, interpolated)
    return merge_lists(interpolated)


folder = "stasloopcsv"
signal_files2 = ['Accelerometer_export2.csv', 'Gyroscope_export2.csv']
merged_list2 = merge_files_content(signal_files2, folder)

import numpy
np_merged = numpy.matrix(merged_list2)
numpy.divide(np_merged[:, 0], 1000000000.0, np_merged[:, 0])
numpy.savetxt(os.path.join(folder, 'merged.csv'), np_merged, fmt='%.10f')

def get_chars(np_marray, sample_rate=10):
    np_array = numpy.array(np_marray.T)[0]
    w = numpy.fft.fft(np_array)
    freqs = numpy.fft.fftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = freqs[idx]
    freq_in_herz = abs(freq * sample_rate)
    correlate = numpy.correlate(np_array, np_array, mode='full')
    return numpy.mean(np_array), freq_in_herz, numpy.std(np_array), numpy.median(np_array), correlate[len(correlate)/2:]

merged_chars = [get_chars(np_merged[:, i+1]) for i in range(1)]
from pprint import pprint
pprint(merged_chars[0][:])
import matplotlib.pyplot as plt

def draw_graphs():
    def draw_array(x, y):
        plt.figure()
        plt.plot(x, y)
        plt.grid(True)
        plt.ylabel('A')
        plt.xlabel('time')

    signals_size = len(signal_files2) * 3
    for i in range(signals_size):
        draw_array(np_merged[:, 0], np_merged[:, i+1])
    draw_array(np_merged[:, 0], merged_chars[0][-1])
    plt.show()

draw_graphs()

