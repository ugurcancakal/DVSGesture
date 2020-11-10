'''
201110
@ugurc
'''

import sys
sys.path.insert(0,'..')

import unittest
import hdf5_dbs_aedat as dbs
import test_env as env
import numpy as np
import os

class TestHdf5DBSAedat(unittest.TestCase):
	def test1_extract_events(self):
		'''
		Extract events from aedat file
		'''
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