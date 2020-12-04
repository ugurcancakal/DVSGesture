'''
Utility functions to create a hdf5 database organized well to perform 
training and test on the DVS128 Gesture dataset

All functions have been tested(only proof of concept not an extensive test)
and the test file is stored in the test folder.

author: @ugurc
201110
'''
import process_aedat as pa
import time 
import numpy as np
import progressbar
import h5py
import os
import glob
import csv

def create_dbs(root_dir, 
							 hdf5_dir='dvs_gestures.hdf5',
							 trials_to_train='trials_to_train.txt', 
							 trials_to_test='trials_to_test.txt',
							 gesture_mapping='gesture_mapping.csv'):
	'''
	Create a hdf database in the given hierarchical structure and including attributes like:
										attrs {'1': 'hand_clapping', '10': 'air_guitar'... # gesture labels
													 'time' : us
													 'link' : 'https://www.research.ibm.com/dvsgesture/'
													 'root_dir' : root_dir
													 'trails_to_train': list of .aedat files processed for train_set
													 'trials_to_test': list of .aedat files processed for test_set
		./train_set 		attrs {'instances' 				: (int) # of instances}
			./0 					attrs	{'light'						: (str) lightening condition,
			./1						 			 'subject'					: (str) name of the subject}
			./2
			...
				./label
				./pos
				./time
		./test_set 			attrs {'instances' 				: (int) # of instances}
			./0 					attrs	{'light'						: (str) lightening condition,
			./1						 			 'subject'					: (str) name of the subject}
			./2
			...
				./label
				./pos
				./time

		Arguments:
			root_dir(str): 
				path to the directory of the file to be processed

			hdf5_dir(str): 
				filepath for database to be created. Default is 'dvs_gestures.hdf5',
				which means the hdf5 database will be created in the current working directory

			trials_to_train(str):
				filepath to the trials_to_train.txt folder expected to be in the root directory.
				Default is 'trials_to_train.txt'

			trials_to_test(str):
				filepath to the trials_to_test.txt folder expected to be in the root directory
				Default is 'trials_to_test.txt'
			
			gesture_mapping(str):
				filepath to the gesture_mapping.csv folder expected to be in the root directory
				Default is 'gesture_mapping.csv'
	'''
	print(f'{hdf5_dir} database is begin created...')
	tic = time.perf_counter()
	train_filelist = pa.get_filelist(root_dir,trials_to_train)
	test_filelist = pa.get_filelist(root_dir,trials_to_test)
	gesture_map = get_gesture_mapping(root_dir,gesture_mapping)

	try:
		with h5py.File(hdf5_dir, 'w') as hdf:
			hdf.attrs['label_idx']=list(gesture_map.keys())
			hdf.attrs['label_name']=list(gesture_map.values())
			hdf.attrs['time'] = 'us'
			hdf.attrs['link'] = 'https://www.research.ibm.com/dvsgesture/'
			hdf.attrs['root_dir'] = root_dir
			hdf.attrs['trials_to_train'] = [reduce_filename(filename) for filename in train_filelist]
			hdf.attrs['trials_to_test'] = [reduce_filename(filename) for filename in test_filelist]
			set_group(hdf, 'train_set', train_filelist)
			set_group(hdf, 'test_set', test_filelist)

	except:
		print("Unable to open the file {}".format(hdf5_dir))

	else:
		toc = time.perf_counter()
		size_byte = os.path.getsize(hdf5_dir)
		print(f'\n{(size_byte/10**9):0.4f} GB of database has been saved in "{hdf5_dir}" in {toc-tic:0.4f} seconds!')

def set_group(hdf, setname, filelist, keystart=0):
	'''
	Create groups and datasets under the given parent group hdf.
	Extract event sequences from given filelist and create a group 
	under the given setname. Event sequence will be seperated into 
	3 dataset structure as time, pos and label for the sake of simplicity
	in further processing. 

	The name of the subject and lightening condition is added as meta 
	attribute	for each instances. 

	The number of instances are added as meta attribute only for the set.

	IMPORTANT: Extracted events may have empty lists because
	event clipping depended on the label may result in empty list creation
	during event extraction. It's redundant and empty sets will be skipped.
	Therefore, if number of inserted event sequences are different than the extracted, 
	that's beceuse there are empty event sequences

		Arguments:

			hdf(parent hdf5 group):
				It may be the root or not. The datasets under the given setname will
				be created under the hdf parent.

			setname(string):
				Name of the group to be created under hdf parent group.

			filelist(string list):
				list of the paths of .aedat files to be processed.
			
			keystart(int):
				starting key for the event sequence instances to be added 
				to the group. Default is 0.

	'''
	count = 0;
	current = 0;
	
	print("\n{} in the database is being created...".format(setname))
	tic = time.perf_counter()
	_set = hdf.create_group(setname)
	events, labels, log = extract_events(filelist)
	print("Insertion has started!")

	subject,light,limit = log[0]

	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(labels), widgets = widgets).start()

	for i,event in enumerate(events, start=keystart):
		if(i>=limit):
			current+=1;
			subject,light,_l = log[current]
			limit += _l

		if (not event.size == 0):
			event_seq = _set.create_group('{}'.format(count))
			event_seq.create_dataset('time', data=event[:,0], dtype=np.uint32)
			event_seq.create_dataset('pos', data=event[:,1:], dtype=np.uint8)
			event_seq.create_dataset('label', data=labels[i], dtype=np.uint8)
			event_seq.attrs['subject']=subject
			event_seq.attrs['light']=light
			count+=1;

		bar.update(i+1)

	_set.attrs['instances']=count
	bar.finish()
	toc = time.perf_counter()
	print(f"\n{count} Event sequences have been inserted succesfully in {toc-tic:0.4f} seconds!")

def extract_events(filenames):
	'''
	Extracts event and label pairs in the form of numpy arrays
	from the given list of filenames. To create filename list,
	process_aedat.get_filelist() function is recommended
	
		Arguments:
			filenames(list of string):
				list of global paths to .aedat files of interest

		Returns:
				events: 
					1D array of 2D variable size arrays consisting of 
					event sequences [[[t,x,y,p],[t,x,y,p]...],[[t,x,y,p],[t,x,y,p]...]...]

				labels: 
					1D array of event labels for the ones in events array.

				log:
				list of subject, light, # events
					includes extraction information for each file processed in the form of a list 
						subject: 
							name of the subject performing the gesture
						light :
							lightening condition
						# events:
							number of event sequences extracted
	'''
	tic = time.perf_counter()
	_events = np.empty(0)
	_labels = np.empty(0, dtype=np.uint8)
	log = []

	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(filenames), widgets = widgets).start()

	for i,name in enumerate(filenames):
		events, labels = pa.aedat_to_np(name)
		log.append((*get_meta(name),len(events))) 
		_events = np.concatenate((_events, events))
		_labels = np.concatenate((_labels, labels))
		bar.update(i+1)
	# print('\n\n\n\n\n',log,'\n\n\n\n\n')
	bar.finish()
	toc = time.perf_counter()
	print(f"\n{len(_labels)} Event sequence&label pairs have been extracted succesfully in {toc-tic:0.4f} seconds!")

	return _events, _labels, log

def get_meta(filepath):
	''' 
	Extracts meta information (subject name and lightening condition)
	from the filepath 

		Arguments:
			filepath(string):
				.aedat filepath including valueable information

		Returns:
			subject(string):
				name of the subject
			light(string):
				lightening condition
	'''
	if '.aedat' in filepath:
		filename = reduce_filename(filepath)
		filename = filename.replace('.aedat', '')
		i = filename.find('_')
		subject = filename[:i]
		light = filename[i+1:]

	return subject, light

def reduce_filename(filepath):
	'''
	Clip the filename from the global filepath

		Arguments:
			filepath(string):
				global filepath
				/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files/user01_fluorescent_led.aedat

			filename(string):
				filename without previous parent directories.	
				user01_fluorescent_led.aedat
	'''
	reverse_path=filepath[::-1]
	i = reverse_path.find(os.path.sep)
	filename=reverse_path[i-1::-1]
	return filename

def get_gesture_mapping(root_dir, file_dir, log=False):
	''' 
	Creates a dictionary of gesture mapping of dataset out of .csv file

	Arguments:
		root_dir(string): parent directory path to the file
		file_dir(string): csv file of interest

	Return:
		mapping(dictionary): dictionary of labels:gesture names
	'''
	path_of_interest = os.path.join(root_dir,file_dir)

	if not os.path.exists(path_of_interest):
		print(f'{file_dir} is missing')
		return {}

	tic = time.perf_counter()
	mapping = {}

	try:
		with open(path_of_interest, 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				if row[1] == 'label':
					continue
				mapping[row[1]] = row[0]

	except:
		print("Unable to process the file {}".format(file_dir))

	else:
		if log:
			toc = time.perf_counter()
			print(f'\n"{path_of_interest}" has been processed in {toc-tic:0.4f} seconds!')

	return mapping
