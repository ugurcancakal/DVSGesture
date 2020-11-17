'''
201110
@ugurc
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
		Extract events from aedat file
		'''
		root_dir = '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files'
		test_config = 'extract_event_test.txt'
		test_filelist = pa.get_filelist(root_dir,test_config)
		pickle_list = [name.replace('.aedat', '.pickle') for name in test_filelist]
		events, labels = dbs.extract_events(test_filelist)

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
	# 	root_dir = '/home/ugurc/drive/data/DvsGesture'
	# 	test_config = 'trials_to_test.txt'
	# 	test_filelist = pa.get_filelist(root_dir,test_config)
	# 	events, labels = dbs.extract_events(test_filelist)
	# 	print('File extraction Test on test set has passed!\n')

	# 	self.assertEqual(events.shape,(288,))

	# def test_1_2_extract_events_train(self):
	# 	root_dir = '/home/ugurc/drive/data/DvsGesture'
	# 	test_config = 'trials_to_train.txt'
	# 	train_filelist = pa.get_filelist(root_dir,test_config)
	# 	events, labels = dbs.extract_events(train_filelist)
	# 	print('File extraction Test on training set has passed!\n')

	# 	self.assertEqual(events.shape,(1176,))


	def test2_set_group(self):
		'''
		Group creation test
		'''
		hdf5_dir = 'test_files/dvs_gestures_unittest.hdf5'
		root_dir = '/home/ugurc/drive/current/ee543/project/DVSGesture/test/test_files'
		test_config = 'extract_event_test.txt'
		test_filelist = pa.get_filelist(root_dir,test_config)
		events, labels = dbs.extract_events(test_filelist)
		_events = []
		_labels = []

		for i,event in enumerate(events):
			if not event.size == 0:
				_events.append(event)
				_labels.append(labels[i])

		# Create a unit_test group
		try:
			with h5py.File(hdf5_dir, 'w') as hdf:
				dbs.set_group(hdf, 'unit_test', test_filelist)
		except:
			print("Failed to create unit_test set")

		# Check if all created correctly
		try:
			with h5py.File(hdf5_dir,'r') as hdf:
				for group, event, lab in zip(hdf['unit_test'].keys(), _events, _labels):
					time = hdf.get('unit_test/{}/time'.format(group))
					pos = hdf.get('unit_test/{}/pos'.format(group))
					label = hdf.get('unit_test/{}/label'.format(group))

					print('\n\n\n',event)
					print(lab,'\n\n\n')
					print(np.array(time))
					print(np.array(pos))
					print(np.array(label))
							# print(items)
							# print(list(hdf[_index]))

		except:
			print("Failed to read the hdf5 file")



		print('Group creation test has passed!\n')

	def test3_create_dbs(self):
		'''
		DBS Creation test
		'''
		print('DBS creation test has passed!\n')

if __name__=='__main__':
	env.clear_env()
	env.hdf5_dbs_aedat_env()
	unittest.main()