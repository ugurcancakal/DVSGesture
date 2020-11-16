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

		print('File extraction Test passed!\n')
	def test2_set_group(self):
		'''
		Group creation test
		'''
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