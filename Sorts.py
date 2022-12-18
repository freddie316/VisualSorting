# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:11:48 2022
Main Visulization and sorting code
@author: Freddie
"""
import sys
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TrackedArray():

    def __init__(self, arr, kind="full"):
        self.arr = np.copy(arr)
        self.kind = kind
        self.reset()

    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []

    def track(self, key, access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)
        if self.kind == "full":
            self.full_copies.append(np.copy(self.arr))

    def GetActivity(self, idx=None):
        if isinstance(idx, type(None)):
            return [(i, op) for (i, op) in zip(self.indices, self.access_type)]
        else:
            return (self.indices[idx], self.access_type[idx])

    def __delitem__(self, key):
        self.track(key, "del")
        self.arr.__delitem__(key)

    def __getitem__(self, key):
        self.track(key, "get")
        return self.arr.__getitem__(key)

    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)
        self.track(key, "set")

    def __len__(self):
        return self.arr.__len__()

    def __str__(self):
        return self.arr.__str__()

    def __repr__(self):
        return self.arr.__repr__()
    
    def __add__(self,other):
        arr = np.append(self.arr, other.arr)
        new = TrackedArray(arr)
        new.indices.append(self.indices)
        new.indices.append(other.indices)
        new.values.append(self.values)
        new.values.append(other.values)
        new.access_type.append(self.access_type)
        new.access_type.append(other.access_type)
        new.full_copies.append(self.full_copies)
        new.full_copies.append(other.full_copies)
        return new

def freq_map(x, x_min = 0, x_max = 1000, freq_min = 120, freq_max = 1200):
    return np.interp(x, [x_min,x_max], [freq_min,freq_max])

def freq_sample(freq,dt=1./60., samplerate=44100, oversample=2):
    mid_samples = int(dt * samplerate)
    pad_samples = int((mid_samples*(oversample-1)/2))
    total_samples = mid_samples + 2 *pad_samples
    
    y = np.sin(2 * np.pi * freq * np.linspace(0,dt,total_samples))
    
    y[:pad_samples] = y[:pad_samples] * np.linspace(0,1,pad_samples)
    y[-pad_samples:] = y[len(y) - pad_samples:] * \
        np.linspace(1, 0, pad_samples)
    
    return y

def bubble_sort(arr):
    unsorted = True
    n = 1
    while unsorted:
        unsorted = False
        for i in range(0,len(arr)-n):
            if arr[i] > arr[i+1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                unsorted = True
        n += 1
    return arr

def cocktail_sort(arr):
    unsorted = True
    n = 0
    m = len(arr)-1
    while unsorted:
        unsorted = False
        for i in range(n,m):
            if arr[i] > arr[i+1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                unsorted = True
        if unsorted == False:
            break
        m -= 1
        for i in reversed(range(n,m)):
            if arr[i] > arr[i+1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                unsorted = True
        n += 1
    return arr

def insertion_sort(arrIn):
    arr = arrIn
    for i in range(1,len(arr)):
        if arr[i] > arr[i-1]:
            continue
        else:
            ind = i
            while ind > 0 and arr[ind] < arr[ind-1]:
                arr[ind], arr[ind - 1] = arr[ind - 1], arr[ind]
                ind -= 1
    return arr

def quick_sort(A, lo=0, hi=0):
    if lo == hi == 0:
        hi = len(A)-1
    if lo < hi:
        p = partition(A, lo, hi)
        quick_sort(A, lo, p - 1)
        quick_sort(A, p + 1, hi)


def partition(A, lo, hi):
    pivot = A[hi]
    i = lo
    for j in range(lo, hi):
        if A[j] < pivot:
            temp = A[i]
            A[i] = A[j]
            A[j] = temp
            i += 1
    temp = A[i]
    A[i] = A[hi]
    A[hi] = temp
    return i

def merge_sort(arr,tracked=0,slicer=0):
    if type(arr) == TrackedArray:
        tracked = arr
        arr = arr.arr.copy()
    if len(arr) > 1:
        half = len(arr)//2
        
        L = arr[:half].copy()
        R = arr[half:].copy()
        
        if tracked != 0:
            if slicer == 0:
                slicer = list(range(0,len(tracked)))
            
            slicerL = slicer[:half]
            slicerR = slicer[half:]
        
            merge_sort(L,tracked,slicerL)
            merge_sort(R,tracked,slicerR)
        
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    tracked[slicer[k]] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    tracked[slicer[k]] = R[j]
                    j += 1
                k += 1
            
            while i < len(L):
                arr[k] = L[i]
                tracked[slicer[k]] = L[i]
                i += 1
                k += 1
                
            while j < len(R):
                arr[k] = R[j]
                tracked[slicer[k]] = R[j]
                j += 1
                k += 1
        else:
            merge_sort(L)
            merge_sort(R)
        
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            
            while i < len(L):
                arr[k] = L[i]
                i += 1
                k += 1
                
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1

def countingSort(arr,expo):
    n = len(arr)
    
    output = [0] * (n)
    count = [0] * (10)
    
    for i in range(0,n):
        ind = arr[i] // expo
        count[int(ind % 10)] += 1
    
    for i in range(1,10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        ind = arr[i] // expo
        output[count[int(ind % 10)] - 1] = arr[i]
        count[int(ind % 10)] -= 1
        i -= 1
    
    i = 0
    for i in range(0,len(arr)):
        arr[i] = output[i]

def radix_sort(arr):
    max1 = max(arr)
    expo = 1
    while max1 / expo >= 1:
        countingSort(arr,expo)
        expo *= 10
    
    

# =============================================================================
# methods = [
#     "bubble",
#     "cocktail",
#     "insertion",
#     "quick",
#     "merge"
#     ]
# 
# print("Available sorting methods:")
# for i in methods:
#     print(i)
# sorter = input("Choose sorting method: ").lower()
# 
# =============================================================================
try:
    sorter = sys.argv[1]
except:
    sorter = "quick"

# Params ########
try:
    N = int(sys.argv[2])
except:
    N = 30
FPS = 60.0
F_SAMPLE = 44100
OVERSAMPLE = 2
#################

arr = np.round(np.linspace(10,1000,N),0)
np.random.shuffle(arr)
arr = TrackedArray(arr)

if sorter == "quick":
    t0 = time.perf_counter()
    quick_sort(arr)
    t = time.perf_counter() - t0
elif sorter == "merge":
    t0 = time.perf_counter()
    merge_sort(arr)
    t = time.perf_counter() - t0
elif sorter == "bubble":
    t0 = time.perf_counter()
    bubble_sort(arr)
    t = time.perf_counter() - t0
elif sorter == "cocktail":
    t0 = time.perf_counter()
    cocktail_sort(arr)
    t = time.perf_counter() - t0
elif sorter == "insertion":
    t0 = time.perf_counter()
    insertion_sort(arr)
    t = time.perf_counter() - t0
elif sorter == "radix":
    t0 = time.perf_counter()
    radix_sort(arr)
    t = time.perf_counter() - t0
else:
    print("Error: Unknown method, defaulting to merge")
    t0 = time.perf_counter()
    merge_sort(arr)
    t = time.perf_counter() - t0

plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams["font.size"] = 16
fig, ax = plt.subplots()
container = ax.bar(np.arange(0, len(arr),1), arr.full_copies[0], align="edge")
ax.set_xlim([0, N])
ax.set(xlabel="Index", ylabel="Value")
ax.set(title=f"{sorter.capitalize()} sort execution time: {t*1E3:.2f} ms to sort {N} elements")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
txt = ax.text(0.02,0.97,"", transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

wav_data = np.zeros(int(F_SAMPLE*len(arr.values)*1./FPS), dtype=float)
dN = int(F_SAMPLE * 1./FPS)

for i, value in enumerate(arr.values):
    freq = freq_map(value)
    sample = freq_sample(freq, 1./FPS, F_SAMPLE, OVERSAMPLE)
    
    idx_0 = int((i+0.5)*dN - len(sample)/2)
    idx_1 = idx_0 + len(sample)
    try:
        wav_data[idx_0:idx_1] += sample
    except ValueError:
        pass
        #print(f"Failed to generate {i} sample")

sp.io.wavfile.write(f"sound/{sorter}_sound.wav", F_SAMPLE, wav_data)

def update(frame):
    if frame == len(arr.full_copies):
        plt.close()
    txt.set_text(f"Array Accesses: {frame}")
    for (rectangle, height) in zip(container.patches, arr.full_copies[frame]):
        rectangle.set_height(height)
        rectangle.set_color("#1f77b4")
        #rectangle.set_edgecolor("k")
    idx, op = arr.GetActivity(frame)
    if idx != len(container.patches):
        if op == "get":
            container.patches[idx].set_color("magenta")
            #container.patches[idx].set_edgecolor("magenta")
        elif op == "set":
            container.patches[idx].set_color("red")
            #container.patches[idx].set_edgecolor("red")
    
    fig.savefig(f"frames/{sorter}_frame{frame:05.0f}.png")
    
    return (txt, *container)

print("Rendering frames, close the plot when done")

ani = FuncAnimation(fig, update, frames=range(len(arr.full_copies)), 
                    blit=False, interval=1000./FPS, repeat=False)
plt.show()
