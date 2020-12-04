'''
Test environment for all unit tests

author: @ugurc
201109 
'''
from aedat_file import AedatFile
import aedat_file as af
import process_aedat as pa
import glob
import os 
import shutil
import numpy as np

def process_aedat_env():
	'''
	Creates an aedat file for testing the process_aedat file. 
	It generates one aedat file, it's labels, and it's pickle.
	'''
	test_path = os.path.join(os.getcwd(), 'test_files')
	print(test_path)
	if not os.path.exists(test_path):
		os.makedirs(test_path)

	file1 = aedat1()
	af.pickle_aedat(file1,test_path)
	af.save_aedat_file(file1,test_path)

	print('TEST ENVIRONMENT HAS BEEN SET!!!\n\n\n')

def aedat_file_env():
	'''
	'''

def hdf5_dbs_aedat_env():
	'''
	Creates the necessary files for testing the hdf5_dbs_aedat.py file
	It generates 3 aedat files, their labels, their pickles, and
	3 text files for configuration: one for listing all the filenames of the 
	artificially created aedat files, one for listing the ones to be used as 
	training set data, one for listing the ones to be used as test set data.
	It also copies gesture_mapping.csv from the original dataset folder. 
	'''
	root_dir = '/home/ugurc/drive/data/DvsGesture'
	gesture_mapping='gesture_mapping.csv'
	
	test_path = os.path.join(os.getcwd(), 'test_files')

	if not os.path.exists(test_path):
		os.makedirs(test_path)

	shutil.copy(os.path.join(root_dir,gesture_mapping),test_path)

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

	config_path_train = 'train_events_test.txt'
	config_path_test = 'test_events_test.txt'

	with open(os.path.join(test_path, config_path),'w') as f:
		for file in filelist:
			f.write(file.filepath+'.aedat'+'\n')

	with open(os.path.join(test_path, config_path_train),'w') as f:
		for file in filelist[0:2]:
			f.write(file.filepath+'.aedat'+'\n')

	with open(os.path.join(test_path, config_path_test),'w') as f:
		for file in filelist[2:]:
			f.write(file.filepath+'.aedat'+'\n')

	print('TEST ENVIRONMENT HAS BEEN SET!!!\n\n\n')

def dvs_dataloader_env():
	'''
	'''
	root_dir = '/home/ugurc/drive/data/DvsGesture'
	trials_to_train = 'trials_to_train.txt'
	trials_to_test = 'trials_to_test.txt'
	gesture_path = 'gesture_mapping.csv'

	test_path = os.path.join(os.getcwd(), 'test_files')

	if not os.path.exists(test_path):
		os.makedirs(test_path)

	trainMissing = os.path.join(test_path, 'trainMissing')
	testMissing = os.path.join(test_path, 'testMissing')
	gestureMissing = os.path.join(test_path, 'gestureMissing')
	randomMissing = os.path.join(test_path, 'randomMissing')

	missingList = [trainMissing,testMissing,gestureMissing,randomMissing]

	for _dir in missingList:
		if not os.path.exists(_dir):
			os.makedirs(_dir)

	path_list = pa.get_filelist(root_dir,trials_to_train) + pa.get_filelist(root_dir,trials_to_test)
	csv_list = [path.replace('.aedat', '_labels.csv') for path in path_list]
	path_list = path_list+csv_list
	path_list.append(os.path.join(root_dir,gesture_path))
	path_list.append(os.path.join(root_dir,trials_to_train))
	path_list.append(os.path.join(root_dir,trials_to_test))

	trainMissing_paths = [path.replace(root_dir, trainMissing) for path in path_list]
	testMissing_paths = [path.replace(root_dir, testMissing) for path in path_list]
	gestureMissing_paths = [path.replace(root_dir, gestureMissing) for path in path_list]
	randomMissing_paths = [path.replace(root_dir, randomMissing) for path in path_list]

	testMissing_paths.pop(-1)
	trainMissing_paths.pop(-2)
	gestureMissing_paths.pop(-3)
	randomMissing_paths.pop(np.random.randint(0,len(randomMissing_paths)-3))

	missing_paths = [trainMissing_paths,testMissing_paths,gestureMissing_paths,randomMissing_paths]

	for _dir in missing_paths:
		for path in _dir:
			with open(path, 'wb') as fp: 
				pass

	shutil.copy(os.path.join(root_dir,trials_to_train),randomMissing)
	shutil.copy(os.path.join(root_dir,trials_to_test),randomMissing)

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
	filename = 'aedat2_led'
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
	filename = 'aedat3_fluorescent'
	heads = [[1, 1, 8, 4, 0, 2, 2, 2],
					 [1, 1, 8, 4, 0, 3, 3, 3],]

	seq = [[[15,127,126,1],[88,18,98,1]],
				 [[1254,15,19,0],[1265,28,19,1],[1288,17,17,1]]]

	labels = [[4,89,90],
						[6,1254,1300]]

	seq_clipped = [[],
								 [[1254,15,19,0],[1265,28,19,1],[1288,17,17,1]]]

	file = af.AedatFile(heads,seq,labels,seq_clipped,filename)
	return file;

def clear_env(keepHDF5=False):
	'''
	Clear all aedat, csv and pickle files and test_files folder
		Arguments:
			keepHDF5: 
				real hdf5 dbs requires takes at least 2 minutes to create.
				So make it true to keep a real hdf5 dbs

	'''
	dir_path = 'test_files'

	try:
		shutil.rmtree(dir_path)
	except OSError as e:
		print("Error: %s : %s" % (dir_path, e.strerror))
	else:
		print(dir_path, ' has cleared including its content!')

	print('\n')







