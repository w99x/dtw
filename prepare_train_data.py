import numpy as np
import matplotlib.pyplot as plt
from hmm import MotionFilter
from hmm import read_and_prepare
from hmm import get_pattern

def prepare_labeled(filteredt, data):
    def find_start_idx(t):
        for idx,v in enumerate(data[0]):
            if abs(v - t) < 0.0001:
                return idx
        return None
    out = []
    for t in filteredt:
        start_idx = find_start_idx(t[0])
        tsize = 11
        d = data[1][start_idx:start_idx + tsize]
        raveled = np.ravel(d.T)

        out.append(raveled)
    return out

def dump_labeled(data, filename):
    d = np.array(data)
    fmt = '%10.5f ' * len(data[0])
    np.savetxt('{}.csv'.format(filename), d, fmt=fmt)


pattern_t, pattern = get_pattern("pattern.csv")
def get_and_dump(data_t, data_s, filename):
    motion_filter = MotionFilter((pattern_t, pattern), (data_t, data_s), [0])
    filteredt = motion_filter.do_filter()
    signal = motion_filter.get_signal()

    labeled = prepare_labeled(filteredt, (signal[0], data_s[np.r_[[0,1]]].T))
    print(len(labeled))
    dump_labeled(labeled, filename)

    def draw_signals(filteredt, datat, datas, name=""):
        minmaxdist = [[-50, 100]] * len(filteredt)

        plt.figure("signal_" + name)
        from functools import reduce
        newdatasarr = list(reduce(lambda res, x: res + [datat, x], datas, []))
        plt.plot(*newdatasarr)
        for i in range(len(filteredt)):
            plt.plot([filteredt[i][0], filteredt[i][0]], minmaxdist[i], '-r')
            plt.plot([filteredt[i][0] + 0.1 * filteredt[i][1], filteredt[i][0] + 0.1 * filteredt[i][1]], minmaxdist[i],
                     '--g', linewidth=2)

        plt.grid()

    draw_signals(filteredt, *signal, name="filtered")

data_t, data_s = read_and_prepare(['max_fh_slice.csv',
                                   #'merged.csv',
                                    'iracsv/merged.csv',
                                    'stasdrivecsv/merged.csv',
                                    #'stasloopcsv/merged.csv'
                                ])


get_and_dump(data_t, data_s, "data_train_negative_1")
plt.show()