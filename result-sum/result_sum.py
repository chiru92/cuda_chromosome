import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import glob, os
import os, os.path
import csv
import operator 
import numpy as np
import pandas as pd

#For Module 1------->
cpu_result_sum = np.loadtxt("similarity_cpu.txt")
print("Similarity CPU Result read Done!")
p = np.shape(cpu_result_sum)[0]
q = np.shape(cpu_result_sum)[1]
cpu_row = np.int32(p)
cpu_col = np.int32(q)

np.set_printoptions(precision=3)
print(cpu_result_sum)

cpu_sum = 0.0
gpu_sum1 = 0.0
gpu_sum2 = 0.0

for i in xrange(0,cpu_row):
	for j in xrange(0,cpu_col):
		cpu_sum = cpu_sum + cpu_result_sum[i][j]

print("")
print("CPU Result Sum = ",cpu_sum)

#For Module 2 ------------>
#For Row wise calculation
gpu_result_sum1 = np.loadtxt("similarity_gpu-1.txt")
print("")
print("Similarity Row wise Result read Done!")
r = np.shape(gpu_result_sum1)[0]
s = np.shape(gpu_result_sum1)[1]
gpu_row = np.int32(r)
gpu_col = np.int32(s)

np.set_printoptions(precision=3)
print(gpu_result_sum1)


for i in xrange(0, gpu_row):
	for j in xrange(0, gpu_col):
		gpu_sum1 = gpu_sum1 + gpu_result_sum1[i][j]

print("")
print("GPU Result Row wise Sum = ",gpu_sum1)

#For Grid wise calculation
gpu_result_sum2 = np.loadtxt("similarity_gpu-2.txt")
print("")
print("Similarity Grid wise Result read Done!")
x = np.shape(gpu_result_sum2)[0]
y = np.shape(gpu_result_sum2)[1]
gpu_row2 = np.int32(x)
gpu_col2 = np.int32(y)

np.set_printoptions(precision=3)
print(gpu_result_sum2)


for i in xrange(0, gpu_row2):
	for j in xrange(0, gpu_col2):
		gpu_sum2 = gpu_sum2 + gpu_result_sum2[i][j]

print("")
print("GPU Grid wise Result Sum = ",gpu_sum2)
print("")