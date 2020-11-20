'''
Unittest for hdf5_dbs_aedat.py file
The functions are tested using a 3 aedat files created artificially 
and some of them are tested using hardcoded dictionaries and lists. 
Properties of the test file can be investigated in test_env.py file.

author: @ugurc
201110
'''

import sys
sys.path.insert(0,'..')

import unittest
import aedat_file as af
import process_aedat as pa
import hdf5_dbs_aedat as dbs
import test_env as env
import numpy as np
import os
import h5py

class TestHdf5DBSAedat(unittest.TestCase):
	def test1_extract_events(self):
		'''
		Checks if event sequence extraction is being done perfectly.
		It checks the extraction process depending on object pickles
		created during test environment creation. 
		'''
		root_dir = '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files'
		test_config = 'extract_event_test.txt'
		test_filelist = pa.get_filelist(root_dir,test_config)
		pickle_list = [name.replace('.aedat', '.pickle') for name in test_filelist]
		events, labels, log = dbs.extract_events(test_filelist)

		_events = np.empty(0)

		for filename in pickle_list:
			test = af.load_aedat_pickle(filename)
			event = np.asarray([np.asarray(seq,dtype=np.uint32) for seq in test.seq_clipped])
			_events = np.concatenate((_events, event))
			
		for event1, event2 in zip(events, _events):
			self.assertTrue((event1 == event2).all())

		self.assertEqual(events.shape,(9,))

		print('File extraction Test on artificial aedat files has passed!\n')

	# def test_1_1_extract_events_test(self):
	# 	'''
	# 	Event sequence extraction applied on real test set data.
	# 	Test function only checks the resulting shape.
	# 	'''
	# 	root_dir = '/home/ugurc/drive/data/DvsGesture'
	# 	test_config = 'trials_to_test.txt'
	# 	test_filelist = pa.get_filelist(root_dir,test_config)
	# 	events, labels, log = dbs.extract_events(test_filelist)
	# 	print('File extraction Test on real test set has passed!\n')
	# 	self.assertEqual(events.shape,(288,))

	# def test_1_2_extract_events_train(self):
	# 	'''
	# 	Event sequence extraction applied on real train set data.
	# 	Test function only checks the resulting shape.
	# 	!It takes TOO MUCH TIME, so better to comment while not using.
	# 	'''
	# 	root_dir = '/home/ugurc/drive/data/DvsGesture'
	# 	test_config = 'trials_to_train.txt'
	# 	train_filelist = pa.get_filelist(root_dir,test_config)
	# 	events, labels, log = dbs.extract_events(train_filelist)
	# 	print('File extraction Test on training set has passed!\n')
	# 	self.assertEqual(events.shape,(1176,))


	def test2_set_group(self):
		'''
		Group creation test done by using 3 artificially created files and
		check done by depending on event pickles created during test environement 
		creation.
		Check if 
			time dataset
			pos dataset
			label dataset
			subject meta information
			light meta information
			instances meta information
		are stored as expected. 
		'''
		hdf5_dir = 'test_files/dvs_gestures_unittest.hdf5'
		root_dir = '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files'
		test_config = 'extract_event_test.txt'
		test_filelist = pa.get_filelist(root_dir,test_config)
		events, labels,log = dbs.extract_events(test_filelist)

		# Eliminate the effect of empty event sequences
		_,_,limit = log[0]
		current = 0
		_events = []
		_labels = []

		for i,event in enumerate(events):
			if(i>=limit):
					current+=1;
					_,_,_l = log[current]
					limit += _l
			if not event.size == 0:
				_events.append(event)
				_labels.append(labels[i])
			else:
				log[current] = (log[current][0], log[current][1], log[current][2]-1)

		subject,light,limit = log[0]
		current = 0

		# Create a unit_test group
		try:
			with h5py.File(hdf5_dir, 'w') as hdf:
				dbs.set_group(hdf, 'unit_test', test_filelist)
		except:
			print("Failed to create unit_test set")

		# Check if all created correctly
		with h5py.File(hdf5_dir,'r') as hdf:
			for i, (group, event, lab) in enumerate(zip(hdf['unit_test'].keys(), _events, _labels)):
				if(i>=limit):
					current+=1;
					subject,light,_l = log[current]
					limit += _l

				time = hdf.get('unit_test/{}/time'.format(group))
				pos = hdf.get('unit_test/{}/pos'.format(group))
				label = hdf.get('unit_test/{}/label'.format(group))
				self.assertTrue((event[:,0]==np.array(time)).all())
				self.assertTrue((event[:,1:]==np.array(pos)).all())
				self.assertEqual(lab, np.array(label))
				self.assertEqual(hdf['unit_test/{}'.format(group)].attrs['subject'], subject)
				self.assertEqual(hdf['unit_test/{}'.format(group)].attrs['light'], light)
				self.assertEqual(hdf['unit_test'].attrs['instances'], len(_events))

		print('Group creation test has passed!\n')

	def test3_create_dbs(self):
		'''
		Test for the hierarchical structure of the database if it's the same as expected
		Tests done using hardcoded lists and dictionaries. 
		Checks if 
			dataset names are equal
			dataset attributes are the same
			dbs attributes are the same
		'''
		root_dir = '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files'
		hdf5_dir = 'test_files/dbs_unittest.hdf5'
		config_path_train = 'train_events_test.txt'
		config_path_test = 'test_events_test.txt'

		dbs.create_dbs(root_dir, hdf5_dir, config_path_train, config_path_test)
		groups = ['train_set', 'test_set']
		train_sub = [i for i in range(6)]
		test_sub = [i for i in range(1)]
		data_sub = ['time', 'pos', 'label']

		datasets=[]
		_datasets = ['test_set/0/label', 'test_set/0/pos', 'test_set/0/time', 
								 'train_set/0/label', 'train_set/0/pos', 'train_set/0/time', 
								 'train_set/1/label', 'train_set/1/pos', 'train_set/1/time', 
								 'train_set/2/label', 'train_set/2/pos', 'train_set/2/time', 
								 'train_set/3/label', 'train_set/3/pos', 'train_set/3/time', 
								 'train_set/4/label', 'train_set/4/pos', 'train_set/4/time', 
								 'train_set/5/label', 'train_set/5/pos', 'train_set/5/time']
		attributes={}
		_attributes = {'test_set': {'instances': 1}, 
									 'test_set/0': {'light': 'fluorescent', 'subject': 'aedat3'}, 
									 'train_set': {'instances': 6}, 
									 'train_set/0': {'light': 'for_testing', 'subject': 'aedat1'}, 
									 'train_set/1': {'light': 'for_testing', 'subject': 'aedat1'}, 
									 'train_set/2': {'light': 'led', 'subject': 'aedat2'}, 
									 'train_set/3': {'light': 'led', 'subject': 'aedat2'}, 
									 'train_set/4': {'light': 'led', 'subject': 'aedat2'}, 
									 'train_set/5': {'light': 'led', 'subject': 'aedat2'}}
		_dbs_attrs = {'1': 'hand_clapping', 
									'10': 'air_guitar', 
									'11': 'other_gestures', 
									'2': 'right_hand_wave', 
									'3': 'left_hand_wave', 
									'4': 'right_arm_clockwise ', 
									'5': 'right_arm_counter_clockwise ', 
									'6': 'left_arm_clockwise ', 
									'7': 'left_arm_counter_clockwise ', 
									'8': 'arm_roll', 
									'9': 'air_drums', 
									'link': 'https://www.research.ibm.com/dvsgesture/', 
									'root_dir': '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files', 
									'time': 'us', 
									'trials_to_test': np.array(['aedat3_fluorescent.aedat'], dtype=object), 
									'trials_to_train': np.array(['aedat1_for_testing.aedat', 'aedat2_led.aedat'], dtype=object)}

		def visitor_func(name, node):
			if isinstance(node, h5py.Dataset):
				datasets.append(name)
			else:
				attributes[name]=dict(node.attrs)

		# Check if all created correctly
		with h5py.File(hdf5_dir,'r') as hdf:
			hdf.visititems(visitor_func)
			self.assertEqual(datasets, _datasets)
			self.assertEqual(attributes, _attributes)
			# dbs attributes
			for attr, _attr in zip(dict(hdf.attrs).items(), _dbs_attrs.items()):
				if (not isinstance(_attr[1],(np.ndarray, np.generic))):
					self.assertEqual(attr, _attr)
				else:
					self.assertEqual(attr[0], _attr[0])
					self.assertTrue((attr[1] == _attr[1]).all())

		print('DBS structure test has passed!\n')

	def test4_get_meta(self):
		'''
		Test for the get_meta function. It's expected to extract
		the subject name and the lightening condition information from 
		a perfectly formed  global path 
		'''
		filepath = '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files/user01_fluorescent_led.aedat'
		subject, light = dbs.get_meta(filepath)
		self.assertEqual(subject,'user01')
		self.assertEqual(light,'fluorescent_led')
		print('Meta information extraction test has passed!\n')
		
	def test5_get_getsure_mapping(self):
		'''
		Test for the get_gesture_mapping function.
		The expected map is hardcoded. 
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		gesture_mapping='gesture_mapping.csv'

		_map = {'1': 'hand_clapping', 
						'2': 'right_hand_wave', 
						'3': 'left_hand_wave', 
						'4': 'right_arm_clockwise ', 
						'5': 'right_arm_counter_clockwise ', 
						'6': 'left_arm_clockwise ', 
						'7': 'left_arm_counter_clockwise ', 
						'8': 'arm_roll', 
						'9': 'air_drums', 
						'10': 'air_guitar', 
						'11': 'other_gestures'}

		gesture_map = dbs.get_gesture_mapping(root_dir,gesture_mapping)
		self.assertEqual(_map, gesture_map)
		print('Gesture mapping test has passed!\n')

	# def test6_create_dbs_real(self):
	# 	'''
	# 	DBS creation test for real data
	# 	!It takes TOO MUCH TIME, comment if unnecessary
	# 	'''
	# 	root_dir = '/home/ugurc/drive/data/DvsGesture'
	# 	dbs.create_dbs(root_dir)
	# 	print('DBS Creation test has passed!\n')
		
if __name__=='__main__':
	env.clear_env()
	env.hdf5_dbs_aedat_env()
	unittest.main()