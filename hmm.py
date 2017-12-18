import numpy as np
import matplotlib.pyplot as plt
import re

ira_merged=np.array(map(lambda l: map(float,filter(lambda x: len(x)>0,re.split('\\s+',l))),open('iracsv/merged.csv'))).T
merged_data=np.array(map(lambda l: map(float, filter(lambda x: len(x) > 0, re.split('\\s+', l))), open('merged.csv'))).T

merged_timet = merged_data[0, :]
merged_datat = merged_data[1:, :]

newdatat, newdatas = merged_timet, merged_datat#interpolate(timet, datat)

#newdatas /= np.max(newdatas)

dk = 11
offset = 480
pattern_t = np.copy(newdatat[offset:offset + dk])
pattern = np.copy(newdatas[offset:offset + dk])
pattern_t -= pattern_t[0]

#plt.figure()
#plt.plot(pattern_t, pattern)
#plt.grid()


ira_merged[0, :] += np.max(merged_timet)
newdatat, newdatas = np.concatenate((merged_timet, ira_merged[0, :])), np.concatenate((merged_datat, ira_merged[1:, :]))

newpatt, newpats = pattern_t, pattern #interpolate(pattern_t, pattern)

plt.figure()
plt.plot(newpatt, newpats)
plt.grid()

from scipy.spatial.distance import euclidean

from scipy.signal import argrelextrema

from fastdtw import fastdtw

#for x in range(1):
x = 0
distances = []
for i in range(len(newdatas) - dk - 1 - x):
    datawin = newdatas[i:dk + i + x]
    distance, path = fastdtw(pattern, datawin, dist=euclidean)
    print distance, path
    distances.append(distance)

extr = argrelextrema(np.array(distances), np.less)
extrvals = extr[0]

def fiilter_extr(extrspos, values, threshold=0.8):
    filtered = [[values[extrspos[0]], None]]
    for i in range(len(extrvals) - 1):
        last = filtered[-1][0]
        if (values[extrspos[i + 1]] - last >= threshold):
            filtered[-1][-1] = values[extrvals[i+1]]
            filtered.append([values[extrvals[i+1]], None])
    return reduce(lambda res, x: res + x, filtered, [])#map(lambda x: values[x], extrspos)#

filteredt = fiilter_extr(extrvals, newdatat)
plt.figure()
plt.plot(range(len(distances)), distances)
plt.grid()

minmaxdist = [[np.min(newdatas), np.max(newdatas)]] * len(filteredt)
plt.figure()
plt.plot(newdatat, newdatas)
for i in range(len(filteredt)):
    plt.plot([filteredt[i],filteredt[i]], minmaxdist[i], '-r')

plt.grid()
plt.show()
raw_input()