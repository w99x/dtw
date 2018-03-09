import csv

start_time_poimt = 0.0
signal_len = 1300.0

def get_data(filename, bound=(start_time_poimt, start_time_poimt + signal_len)):
    tonanosec = 1000000000.0
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        starttime = next(spamreader)[1]
        rows = [values for values in spamreader]

    def extract_timescale(t):
        return str((float(t) - float(starttime)) / tonanosec)

    series = [[extract_timescale(row[1])] + row[2:5] for row in rows]
    out = filter(lambda x: bound[0] < float(x[0]) < bound[1], series)
    out = [map(float, x) for x in out]
    firsttime = out[0][0]
    return [[x[0] - firsttime] + x[1:] for x in out]

def get_series(series_gen, title=''):
    series_list = list(series_gen)
    return series_list

def dump_csv(file, title_series):
    import csv
    with open('{}_{}.csv'.format(file, "xyz"), 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', escapechar='\n', quoting=csv.QUOTE_NONE)
        map(spamwriter.writerow, title_series)


def prepare(folder, file):
    import os.path
    filename = os.path.join(folder, file)

    #bounds = [(0.3, 1.45), (1.45, 2.56), (2.56, 3.69), (3.69, 4.9), (4.9, 5.95), (5.95, 7.1), (7.1, 8.42), (8.42, 9.63), (9.63, 10.88)]
    bounds = [(0, 100500)]
    outfolder = "output"
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    map(lambda x: dump_csv(os.path.join(outfolder, file + '_export_' + str(x)), get_data(filename, bound=x)), bounds)

if __name__ == "__main__":
    import sys
    files = ["Gyroscope_export.txt"]
    map(lambda x: prepare('max_fh_slice', x), files)
