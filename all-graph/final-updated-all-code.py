#
#
#
#		**************************************************************************
#
#  					This Machine GPU having following details : 
#							Device Name : GeForce GTX 650
#							Total Global Memory : 979 MB
#							maximum Shared memory in a block : 48 KB
#							Maximum Threads per Block : 1024
#
# 		***************************************************************************
#
#
#	
#		In this machine GPU having maximum 1024 threads in a single block.  
#
#		The CPU vs GPU execution Graph having 640x480 pixels and it's 'X' axis represent the No. of Data(in unit of 50K) that executed 
#			and 'Y' axis represent the Execution Time(in unit of Sec.).
#
#		The CPU execution line represented by "Black" line, the GPU1 or Row Wise Kernel execution line represented by "Red" line
#			and the GPU2 or Grid Wise Kernel execution line represent by "Blue" line.
#
#
#

from __future__ import print_function
from __future__ import absolute_import
from pycuda.compiler import SourceModule
from myScatter import * 

import pycuda.driver as drv
import pycuda.autoinit  
import numpy as np
import time
import pandas as pd
import glob, os
import os, os.path
import csv
import operator
import re

mod = SourceModule("""
	//Row-Wise Kernels
	__global__ void MinMax(double *a1, double *a2, double *mini_gpu, double *maxi_gpu, int max_threads_int)
	{
		int idx = threadIdx.x;
		int pos = 0;
		double min = 0.0, max = 0.0;
		if(idx < max_threads_int
		)
		{
			for(int i=0; i<8; i++)
			{	
				pos = idx*8 + i;
				if( a1[pos] <= a2[pos])
				{
					min = min + a1[pos];
					max = max + a2[pos];
				}
				else
				{
					min = min + a2[pos];
					max = max + a1[pos];
				}
			}
			mini_gpu[idx] = mini_gpu[idx] + min;
			maxi_gpu[idx] = maxi_gpu[idx] + max;
		}
	}
	__global__ void MinMaxSum(double *mini_gpu, double *maxi_gpu, double *mini_tmp_gpu, double *maxi_tmp_gpu, int next_protein_int, int max_threads)
	{
		int idx = threadIdx.x;
		double min = 0.0, max = 0.0;
		if(idx < 1)
		{
			for(int i=0; i<max_threads; i++)
			{
				min = min + mini_gpu[i];
				max = max + maxi_gpu[i];
			}
			mini_tmp_gpu[next_protein_int] = min;
			maxi_tmp_gpu[next_protein_int] = max;
		}

	}
	__global__ void result(double *mini_tmp_gpu, double *maxi_tmp_gpu, double *res_gpu,int current_protein, int no_of_protein)
	{
		int idx = threadIdx.x;
		int position1 = current_protein*no_of_protein + idx;
		int position2 = idx*no_of_protein + current_protein;
		if(idx>=current_protein && idx<no_of_protein)
		{
			res_gpu[position1] = (double)mini_tmp_gpu[idx]/(double)maxi_tmp_gpu[idx];	
			res_gpu[position2] = (double)mini_tmp_gpu[idx]/(double)maxi_tmp_gpu[idx];
			
		}
	}

	//Grid-Wise Kernels 

	__global__ void MinMax2(double *a_gpu, double *mini2_gpu, double *maxi2_gpu, int current_protein, int no_of_protein, int no_of_columns_int)
	{
		int idx = threadIdx.x;
		if(idx<no_of_protein)
		{	
			double min = 0.0, max = 0.0;
			for(int j=0; j<no_of_columns_int; j++)
			{
				int position1 = current_protein*no_of_columns_int + j;
				int position2 = idx*no_of_columns_int + j;
				if(a_gpu[position1]<=a_gpu[position2])
				{
					min = min + a_gpu[position1];
					max = max + a_gpu[position2];
				}
				else
				{
					min = min + a_gpu[position2];
					max = max + a_gpu[position1];
				}
			}
			mini2_gpu[idx] = mini2_gpu[idx] + min;
			maxi2_gpu[idx] = maxi2_gpu[idx] + max;
		}
	}

	__global__ void result2(double *mini2_gpu, double *maxi2_gpu, double *res2_gpu, int current_protein, int no_of_protein)
	{
		int idx = threadIdx.x;
		int position1 = current_protein*no_of_protein + idx;
		int position2 = idx*no_of_protein + current_protein;
		if(idx>=current_protein && idx<no_of_protein)
		{
			res2_gpu[position1] = (double)mini2_gpu[idx]/(double)maxi2_gpu[idx];
			res2_gpu[position2] = (double)mini2_gpu[idx]/(double)maxi2_gpu[idx];
		}
	}
	""")




#Getting Maximum no of Threads details:

max_threads = 0												# Initialize max_threads to 0. It's holds the maximum Threads per Block for GPU

for devicenum in range(drv.Device.count()):
    device=drv.Device(devicenum)
    attrs=device.get_attributes()

    #Beyond this point is just pretty printing
    print("\n===Attributes for device %d"%devicenum)
    for (key,value) in attrs.iteritems():
    	if(str(key)=="MAX_THREADS_PER_BLOCK"):
	        max_threads = value								# Assign the Maximum no of Threads value to max_threads

print('Max Threads: ',max_threads)

max_threads_int = np.int32(max_threads)						# Convert max_threads variable to numpy int32
print(max_threads_int)
max_threads = np.int(max_threads_int)						# Convert max_threads variable to numpy int


#Set the ProteinKet data set folder path

path = "/home/pycuda/Desktop//pycuda/Old/ProteinKey3/"			# home path for all Protein-Key folders
dirs = os.listdir( path )										# Create a list of all folders that present in Protein-Key folder
dirs.sort()

cpu = np.zeros(len(dirs))										# Define a array that holds the cpu execution time
gpu1 = np.zeros(len(dirs))										# Define a array that holds the gpu1 or Row Wise execution time
gpu2 = np.zeros(len(dirs))										# Define a array that holds the gpu2 or Grid Wise execution time
no_of_data = np.zeros(len(dirs))								# Define a array that holds the Total No. of Data that will be executed for each folder 
i = 0															# initialize i that define the position of the arrays

LTMargin = 100													# For Scatter Graph define the Left Margin with 100 pixel
TPMargin = 100													# For Scatter Graph define the Top Margin with 100 pixel

for d in dirs:
	os.chdir("/home/pycuda/Desktop/pycuda/Old/ProteinKey3/")
	print(d)													# Print the folder name that currently executed 
	os.chdir(d)													# Set the child directory with current executed folder 
	Protein=[]													# Initialize a list that holds the Protein Names
	for file in glob.glob("*.keys"):
		Protein.append(file)									# Save all Protein file names into the Protein list

	no_of_proteins = len(Protein)								# Define the No. of Protein presented in the current folder
	print("#Proteins  = ", no_of_proteins)

	#Sorting all Protein files ------------->
	Protein.sort()
	print("FileSorting Done!")
	#print(Protein)	

	#Getting all unique keys in a sorted list 'Keys[]' ------------>
	keys = []
	for pt in Protein:
		with open(pt, "r") as p_file:
			filename = ""+p_file.name
			#print(filename)
			f = open(filename, "r")
			for line in f:
				cntnt = line.split()	
				res = list(map(int, cntnt))
				item = res[0]
				keys.append(item)	#add only 1st number of each line of the files as keys
			f.close()

	keys = np.unique(keys)
	no_unq_keys = keys.shape[0]
	np.save("keys", keys)	#save the keys into Keys file
	print("#Unique Keys = ", no_unq_keys)
	#print(Keys)
	print("Getting Unique keys Done!")

	#Forming Protein-Key Matrix ------------------------>
	PKmat_gpu = np.zeros(shape=(no_of_proteins,no_unq_keys))
	for p in Protein:
		with open(p, "r") as pt_file:
			fname = ""+pt_file.name
			fl = open(fname, "r")
			for line in fl:
				content = line.split()
				results = list(map(int, content))
				r = Protein.index(p)
				var = np.where(keys==results[0])
				c = var[0][0]
				PKmat_gpu[r][c] = results[1]
			f.close()
	print("PKmat_gpu Done!")
	#print(PKmat[14][8])
	np.savetxt("PKmat_gpu.txt", PKmat_gpu, delimiter=' ')
	df = pd.DataFrame(PKmat_gpu, columns=keys)
	df.to_csv('pkmat_gpu.csv')
	print("Protein-Key matrix fromulation Done!")

	#<<--------------------This is the CPU code section ------------------>>

	r = np.shape(PKmat_gpu)[0]		#No. of Proteins
	total_protein = np.int32(r)			
	c = np.shape(PKmat_gpu)[1]		#No. of Keys
	unq_keys = np.int32(c)

	similariy = np.zeros(shape=(no_of_proteins,no_of_proteins), dtype=np.float64)

	print("Entering in the calculation section!")

	start_time = time.clock()

	for protein in xrange(0,total_protein):
		current_protein = np.int32(protein)
		for next_protein in xrange(current_protein,total_protein):				
			minsum = 0.0												#initialize minsum with 0.0
			maxsum = 0.0												#initialize maxsum with 0.0
			for unique_key in xrange(0,unq_keys):
				if(PKmat_gpu[protein][unique_key] <= PKmat_gpu[next_protein][unique_key]):		#Compare ProteinKey value with next ProteinKey value
					minsum = minsum + PKmat_gpu[protein][unique_key]						# minsum adding itself with PKmat[protein][unique_key]
					maxsum = maxsum + PKmat_gpu[next_protein][unique_key]					# maxsum, adding itself with PKmat[next_protein][unique_key]
				else:
					minsum = minsum + PKmat_gpu[next_protein][unique_key]
					maxsum = maxsum + PKmat_gpu[protein][unique_key]
			res = minsum/maxsum
			similariy[protein][next_protein] =  res
			similariy[next_protein][protein] =  res
		#print(x)

	#print ("Total Code exection took :",time.clock() - start_time, "seconds.")
	cpu_time = (time.clock() - start_time)				# Calculate the CPU code execution time in multiply of 10 Sec
	cpu[i] = cpu_time 										# Save the CPU code execution time in cpu array

	print ("Similarity calculation on CPU Done!")

	np.savetxt("similarity_cpu.txt", similariy, delimiter=' ')

	np.set_printoptions(precision=3)	#for print Similarity upto 3 decimal
	print (similariy)

	df2 = pd.DataFrame(similariy)		#Save Result in .csv file
	df2.to_csv('similarity_cpu.csv')

	#<<--------------------This is the Device Kernel Section that calculate the Similarity value------------------->>

	#*****************This Kernel Section calculate the Similarity matrix in Row wise*****************


	total_protein = np.shape(PKmat_gpu)[0]
	no_of_protein = np.int32(total_protein)
	total_keys = np.shape(PKmat_gpu)[1]
	no_unq_keys = np.int32(total_keys)

	print("")
	print("")
	print("*************Row Wise Calculation*************")
	print("")
	print("")


	mini = np.zeros(max_threads)							# Create Mini Column martix on Host Memory
	mini_gpu = drv.mem_alloc(mini.nbytes)					# Allocate Device memory for Mini Column matrix
	maxi = np.zeros(max_threads)							# Create Maxi Column matrix on Host Memory
	maxi_gpu = drv.mem_alloc(maxi.nbytes)					# Allocate Device memory for Maxi Column matrix
	res = np.zeros(shape=(no_of_protein,no_of_protein))		# Create Result matrix on Host Memory to store the Similarirty value
	res_gpu = drv.mem_alloc(res.nbytes)						# Allocate Device memory for Result matrix

	drv.memcpy_htod(res_gpu, res)						#Set Device Result matrix with 0's

	no_of_element_temp_array =	max_threads*8; 			# This variable for temp array size that will calculate for result

	#Create two Temp row matrix for GPU calculation and allocate there corrosponding memory into Device
	a1 = np.zeros(no_of_element_temp_array)			
	a2 = np.zeros(no_of_element_temp_array)
	a1_gpu = drv.mem_alloc(a1.nbytes)
	a2_gpu = drv.mem_alloc(a2.nbytes)

	# It create a variable loop that indicated each Row divided into how many small part
	loop = 0
	lp = np.float32(no_unq_keys)/np.float32(no_of_element_temp_array)
	lp1 = lp % 1
	if(lp1 > 0):
		loop = lp + 1
	loop = np.int32(loop)


	no_of_protein_as_thread = np.int(no_of_protein)				#Assume no of Proteins < maximum Threads

	print("Entering in the kernel section!")
	start_time = time.clock()									#Start the timer

	mini_tmp = np.zeros(no_of_protein)									# Create mini_tmp array on Host Memory and fill 0's into it
	maxi_tmp = np.zeros(no_of_protein)									# Create maxi_tmp array on Host Memory and fill 0's into it
	mini_tmp_gpu = drv.mem_alloc(mini_tmp.nbytes)						# Allocate Device memory for mini_tmp array
	maxi_tmp_gpu = drv.mem_alloc(maxi_tmp.nbytes)						# Allocate Device memory for maxi_tmp array

	for protein in xrange(0,total_protein):								# it's indicated the Protein no.
		current_protein = np.int32(protein)								# Convert current_protein in numpy int32 format
		#print(current_protein)
		drv.memcpy_htod(mini_tmp_gpu, mini_tmp)							# Copy the data from (mini_tmp) host memory to (mini_tmp_gpu) device memory
		drv.memcpy_htod(maxi_tmp_gpu, maxi_tmp)							# Copy the data from (maxi_tmp) host memory to (maxi_tmp_gpu) device memory
		for next_protein in xrange(protein, total_protein):				# it's indicated the Next Protein no.
			#print(current_protein," ",next_protein)
			next_protein_int = np.int32(next_protein)					# Convert next_protein in numpy int32 format
			drv.memcpy_htod(mini_gpu, mini)								# Copy the data from (mini) host memory to (mini_gpu) device memory
			drv.memcpy_htod(maxi_gpu, maxi)								# Copy the data from (maxi) host memory to (maxi_gpu) device memory
			position = 0												# Initialize the position with 0
			for grd in xrange(0, loop):											# it's indicated each Row divided into how many small part
				for keys_pos in xrange(0, no_of_element_temp_array):			# Fill each temp a1 and a2 arrays
					if(position < total_keys):									# if position < total keys then fill a1 and a2 array with PKMa_gpu matrix data else fill with 0's
						a1[keys_pos] = PKmat_gpu[current_protein][position]		# Fill the a1 array with PKMat_gpu matrix data
						a2[keys_pos] = PKmat_gpu[next_protein][position]		# Fill the a2 array with PKMat_gpu matrix data
					else:
						a1[keys_pos] = 0
						a2[keys_pos] = 0
					position = position + 1				# Increment the position by 1

				drv.memcpy_htod(a1_gpu, a1)				# Copy the data from (a1) host memory to (a1_gpu) device memory
				drv.memcpy_htod(a2_gpu, a2)				# Copy the data from (a2) host memory to (a2_gpu) device memory

				MinMax = mod.get_function("MinMax")		# Create MinMax function in host that call the MinMax Kernel on GPU or Device
				MinMax(a1_gpu, a2_gpu, mini_gpu, maxi_gpu, max_threads_int, block = (max_threads,1,1), grid = (1,1))	# Send MinMax Kernel Arguments and Kernel Structure(as block, grid)
	
			MinMaxSum = mod.get_function("MinMaxSum")
			MinMaxSum(mini_gpu, maxi_gpu, mini_tmp_gpu, maxi_tmp_gpu, next_protein_int, max_threads_int, block =(1,1,1), grid = (1,1))
		
		result = mod.get_function("result")			# Create result function in host that call the result kernel on GPU or Device
		result(mini_tmp_gpu, maxi_tmp_gpu, res_gpu,current_protein, no_of_protein, block = (no_of_protein_as_thread,1,1), grid = (1,1))			#Send result Kernel Arguments and Kernel Structure(as block, grid)
	

	#print ("Total GPU Code execution took :",time.clock() - start_time, "seconds.")		# Stop timer and Calculate total time
	gpu1_time = (time.clock() - start_time)			# Calculate the GPU1 or Row Wise Kernel code execution time in multiply of 10 Sec
	gpu1[i] = gpu1_time 								# Save the GPU1 or Row Wise Kernel code execution time in gpu1 array

	print ("Similarity calculation on GPU Done!")

	drv.memcpy_dtoh(res, res_gpu)				# Copy data of (res_gpu) device memory to (res) host memory  
	np.savetxt("similarity_gpu-1.txt", res, delimiter=' ')		# Save results in similarity_gpu.txt file 

	np.set_printoptions(precision=3)
	print(res)

	df1 = pd.DataFrame(res)		# Save Result in .csv file
	df1.to_csv('similarity_gpu-1.csv')	



	#***************** This Kernel Section calculate the Similarity matrix in Grid wise **********************
	#	In each grid having total no. of protein as rows and some keys as columns


	print("")
	print("")
	print("*************Grid Wise Calculation*************")


	no_of_columns = max_threads/4					# Calculate the no of Columns for each grid
	no_of_columns_int = np.int32(no_of_columns)		# Convert the no of Columns in numpy int32 format
	mini2 = np.zeros(no_of_proteins)				# Create Mini2 Column martix on Host Memory and fill 0's into it
	mini2_gpu = drv.mem_alloc(mini2.nbytes)			# Allocate Device memory for Mini2 Column matrix
	maxi2 = np.zeros(no_of_proteins)				# Create Maxi2 Column matrix on Host Memory and fill 0's into it
	maxi2_gpu = drv.mem_alloc(maxi2.nbytes)			# Allocate Device memory for Maxi2 Column matrix

	a_cpu = np.zeros(shape=(no_of_proteins,no_of_columns))			# Create temp Grid on host memory
	a_gpu = drv.mem_alloc(a_cpu.nbytes)			# Allocate Device memory for temp Grid (a_gpu) 

	res2 = np.zeros(shape=(no_of_proteins,no_of_proteins))			# Create Result2 matrix on Host Memory to store the Similarirty value
	res2_gpu = drv.mem_alloc(res2.nbytes)		# Allocate Device memory for Result2 matrix
	drv.memcpy_htod(res2_gpu, res2)				# Set Device Result2 matrix with 0's

	# Calculate the no of Grids needed for getting result Similarity matrix
	gd = np.float32(no_unq_keys)/np.float32(no_of_columns)
	gd1 = gd % 1
	#print(gd1)
	if(gd1 > 0):
		grd = gd + 1
	grd = np.int32(grd)


	no_of_protein_as_thread = np.int(no_of_protein)				#Assume max Threads 1024 and assign thread with no of Proteins

	print("Entering in the kernel section!")
	start_time = time.clock()						# Start the timer
	position2 = 0
	for protein in xrange(0,total_protein):			#For each protein
		current_protein = np.int32(protein)
		drv.memcpy_htod(mini2_gpu, mini2)		# Set device (mini2_gpu) matrix with 0's 
		drv.memcpy_htod(maxi2_gpu, maxi2)		# Set device (maxi2_gpu) matrix with 0's 
		for grd_no in xrange(0,grd):			# Total no of Grids
			a_cpu = np.zeros(shape=(no_of_proteins,no_of_columns))	# Create temp Grid on host memory
			for pt in xrange(current_protein,total_protein):		
				y = np.int32(pt)
				for key_pos in xrange(0,no_of_columns):
					position2 = key_pos+(grd_no*no_of_columns)
					if(position2<total_keys):
						a_cpu[pt][key_pos] = PKmat_gpu[pt][position2]	# Fill the temp Grid (a) on host with value of PKMatrix
					else:
						a_cpu[pt][key_pos] = 0.0

			drv.memcpy_htod(a_gpu,a_cpu)		# Copy the data from (a) host memory to (a_gpu) device memory 

			MinMax2 = mod.get_function("MinMax2")	# Create MinMax2 function in host that call the MinMax2 Kernel on GPU or Device
			MinMax2(a_gpu, mini2_gpu, maxi2_gpu, current_protein, no_of_protein, no_of_columns_int, block = (no_of_protein_as_thread,1,1), grid = (1,1))		# Send MinMax2 Kernel Arguments and Kernel Structure(as block, grid)
		# print(protein)

		result2 = mod.get_function("result2")	# Create result2 function in host that call the result kernel on GPU or Device
		result2(mini2_gpu, maxi2_gpu, res2_gpu, current_protein, no_of_protein, block = (no_of_protein_as_thread,1,1), grid = (1,1))	# Send result2 Kernel Arguments and Kernel Structure(as block, grid)

	# print ("Total GPU Code execution took :",time.clock() - start_time, "seconds.")		#Stop timer and Calculate total time
	gpu2_time = (time.clock() - start_time)				# Calculate the GPU2 or Grid Wise code execution time in multiply of 10 Sec
	gpu2[i] = gpu2_time 									# Save the GPU2 or Grid Wise Kernel code execution time in gpu2 array

	print ("Similarity calculation on GPU-2 Done!")

	drv.memcpy_dtoh(res2, res2_gpu)		# Copy data of (res2_gpu) device memory to (res2) host memory  
	np.savetxt("similarity_gpu-2.txt", res2, delimiter=' ')		# Save results in similarity_gpu-2.txt file 

	np.set_printoptions(precision=3)
	print(res2)						# Print Result2 Matrix

	df2 = pd.DataFrame(res2)		# Save Result2 in .csv file
	df2.to_csv('similarity_gpu-2.csv')

	no_of_data[i] = (no_of_proteins * no_unq_keys)		# Calculate the Total no of data that was executed in current folder

	i = i + 1											# Increment the i value that need to poinout the array position


print("")
print("")
print(cpu)						# Print the CPU array that holds the CPU code execution time details in Sec.
print(gpu1)						# Print the GPU1 or Row Wise Kernel code execution time details in Sec.
print(gpu2)						# Print the GPU2 or Grid Wise Kernel code execution time details in Sec.
print(no_of_data)				# Print No. of Data details of each folders

# Save the execution details in .txt files in the last folder of the directory that holds the ProteinKey folders
np.savetxt("CPU_Time.txt", cpu, delimiter = ' ')				
np.savetxt("GPU1_Time.txt", gpu1, delimiter = ' ')
np.savetxt("GPU2_Time.txt", gpu2, delimiter = ' ')
np.savetxt("No_of_Data.txt", no_of_data, delimiter = ' ')



# ************************ Graphics Output Section *******************

rect(no_of_data[len(no_of_data)-1])

# Create the outer grids of the graphics output
j=0
for i in xrange(0,len(no_of_data)/3):
	s = no_of_data[j]
	c = cpu[j]
	outGrid(s,c)
	j = j+3
# End of grid creation

lineDraw(no_of_data, cpu, 'black')		# Create Graph Line for CPU execution time details which stored in cpu array
lineDraw(no_of_data, gpu1, 'red')		# Create Graph Line for GPU! or Row Wise execution time details which stored in gpu1 array
lineDraw(no_of_data, gpu2, 'blue')		# Create Graph Line for GPU2 or Grid Wise execution time details which stored in gpu2 array

message()								# Show the inbuilt messages
input()


