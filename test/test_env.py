'''
Test environment for all unit tests

author: @ugurc
201109 
'''
from aedat_file import AedatFile
import aedat_file as af
import glob
import os 
import shutil
import numpy as np

def process_aedat_env():
	'''
	'''
	test_path = os.path.join(os.getcwd(), 'test_files')
	print(test_path)
	if not os.path.exists(test_path):
		os.makedirs(test_path)

	file1 = aedat1()
	af.pickle_aedat(file1,test_path)
	af.save_aedat_file(file1,test_path)

def aedat_file_env():
	'''
	'''

def hdf5_dbs_aedat_env():
	'''
	'''

	test_path = os.path.join(os.getcwd(), 'test_files')

	if not os.path.exists(test_path):
		os.makedirs(test_path)

	file1 = aedat1()
	af.pickle_aedat(file1,test_path)
	af.save_aedat_file(file1,test_path)

	file2 = aedat2()
	af.pickle_aedat(file2,test_path)
	af.save_aedat_file(file2,test_path)

	file3 = aedat3()
	af.pickle_aedat(file3,test_path)
	af.save_aedat_file(file3,test_path)

	filelist = [file1, file2, file3]

	test_path = 'test_files'
	config_path = 'extract_event_test.txt'
	with open(os.path.join(test_path, config_path),'w') as f:
		for file in filelist:
			f.write(file.filepath+'.aedat'+'\n')

def aedat1():
	'''
	In the mock file there are 3 event records.

	seq[0] includes 5 events and it's 
	first and last events are to be clipped. 

	seq[1] includes 4 events and labels chosen in a way 
	to show that the interval to be considered is [startTime_usec, endTime_usec)

	seq[2] includes 3 events and all will be clipped

	Event headers are filled to represent sequences carefully.
	'''
	
	filename = 'aedat1_for_testing'
	heads = [[1, 1, 8, 4, 0, 5, 5, 5],
					 [1, 1, 8, 4, 0, 4, 4, 4],
					 [1, 1, 8, 4, 0, 3, 3, 3]]

	seq = [[[10,0,0,1],[25,10,20,0],[38,120,59,1],[48,5,9,0],[59,9,5,1]],
				 [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1],[126562,45,59,1]],
				 [[2068981,127,127,0],[2157619,0,15,1],[2178812,1,1,1]]]


	labels = [[1,24,50],
	          [5,102865,126562],
	          [11,2157620,2178812]]

	seq_clipped = [[[25,10,20,0],[38,120,59,1],[48,5,9,0]],
								 [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1]],
								 []]

	file = af.AedatFile(heads,seq,labels,seq_clipped,filename)
	return file;

def aedat2():
	'''
	In the mock file there are 4 event records.

	seq[0] includes 5 events and it's 
	first and last events are to be clipped. 

	seq[1] includes 4 events and labels chosen in a way 
	to show that the interval to be considered is [startTime_usec, endTime_usec)

	seq[2] includes 3 events and all will be clipped

	Event headers are filled to represent sequences carefully.
	'''
	filename = 'aedat2_for_testing'
	heads = [[1, 1, 8, 4, 0, 6, 6, 6],
					 [1, 1, 8, 4, 0, 4, 4, 4],
					 [1, 1, 8, 4, 0, 5, 5, 5],
					 [1, 1, 8, 4, 0, 3, 3, 3]]

	seq = [[[10,0,0,1],[25,10,20,0],[38,120,59,1],[48,5,9,0],[59,9,5,1],[68,127,115,0]],
				 [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1],[126562,45,59,1]],
				 [[126587,0,0,1],[126918,10,20,0],[127100,120,59,1],[127120,5,9,0],[127230,9,5,1]],
				 [[2068981,127,127,0],[2157619,0,15,1],[2178812,1,1,1]]]

	labels = [[2,40,59],
	          [3,119579,119580],
	          [8,126918,127220],
	          [7,2068981,2178818]]

	seq_clipped = [[[48,5,9,0]],
								 [[119579,32,44,1]],
								 [[126918,10,20,0],[127100,120,59,1],[127120,5,9,0]],
								 [[2068981,127,127,0],[2157619,0,15,1],[2178812,1,1,1]]]

	file = af.AedatFile(heads,seq,labels,seq_clipped,filename)
	return file;

def aedat3():
	'''
	In the mock file there are 4 event records.

	seq[0] includes 5 events and it's 
	first and last events are to be clipped. 

	seq[1] includes 4 events and labels chosen in a way 
	to show that the interval to be considered is [startTime_usec, endTime_usec)

	seq[2] includes 3 events and all will be clipped

	Event headers are filled to represent sequences carefully.
	'''
	filename = 'aedat3_for_testing'
	heads = [[1, 1, 8, 4, 0, 2, 2, 2],
					 [1, 1, 8, 4, 0, 3, 3, 3],]

	seq = [[[15,127,126,1],[88,18,98,1]],
				 [[1254,15,19,0],[1265,28,19,1],[1288,17,17,1]]]

	labels = [[4,15,90],
	          [6,1254,1300]]

	seq_clipped = [[[15,127,126,1],[88,18,98,1]],
			 					 [[1254,15,19,0],[1265,28,19,1],[1288,17,17,1]]]

	file = af.AedatFile(heads,seq,labels,seq_clipped,filename)
	return file;

def clear_env():
	'''
	Clear all aedat, csv and pickle files and test_files folder
	'''
	dir_path = 'test_files'
	cleared = glob.glob('*.aedat')
	cleared += glob.glob('*_labels.csv')
	cleared += glob.glob('*.pickle')
	cleared += glob.glob('*.txt')

	for file in cleared:
		try:
			os.remove(file)
		except:
			print(file, ' removal is unsuccesful')
		else:
			print(file, ' has cleared!')

	try:
		shutil.rmtree(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))
	else:
		print(dir_path, ' has cleared including its content!')

	print('\n')







