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

def process_aedat_env():
	'''
	'''
	file = aedat1()
	af.save_aedat_file(file)

def aedat_file_env():
	'''
	'''

def hdf5_dbs_aedat_env():
	'''
	'''

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
	if not os.path.exists('test_files'):
		os.makedirs('test_files')
	filename = os.path.join('test_files','aedat1_for_testing')
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

	file = af.AedatFile(heads,seq,labels,seq_clipped,filename,dump=True)
	return file;

def clear_env():
	'''
	Clear all aedat, csv and pickle files and test_files folder
	'''
	dir_path = 'test_files'
	cleared = glob.glob('*.aedat')
	cleared += glob.glob('*_labels.csv')
	cleared += glob.glob('*.pickle')

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







