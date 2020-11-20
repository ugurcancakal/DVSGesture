'''
Unittest for aedat_file.py file
The functions are tested using only one piece of possible 
input. The empty cases and big chunk of data cases remains untested. 

201110
@ugurc
'''

import sys
sys.path.insert(0,'..')

import unittest
import aedat_file as af
import process_aedat as pa
from aedat_file import AedatFile
import test_env as env
import numpy as np
import os
from datetime import datetime,timezone,timedelta

class TestAedatFile(unittest.TestCase):
	def test1_start_time(self):
		''' 
		Check if the start time is in the correct format and 
		try constructing the object with user defined start time
		VISUAL INSPECTION REQUIRED!
		Check if the time is correct!
		'''
		time_defined = datetime(1995,5,9,14,20,48,tzinfo=timezone(timedelta(hours=3)))
		start_now = AedatFile([],[],[],[],'')
		start_defined = AedatFile([],[],[],[],'', time_defined)
		print('NOW:', start_now._start_time)
		print('@ugurc:',start_defined._start_time)
		print('Start Time Check Test has been passed!\n')

	def test2_load_store_aedat(self):
		'''
		Check if the object is correctly pickled and what is necessary to
		dump and load the AedatFile object
		AedatFile constructor IMPORT is a must!

		Functions under test:
		save_aedat_file
		pickle_aedat
		load_aedat_pickle
		'''
		if not os.path.exists('test_files'):
			os.makedirs('test_files')

		filename = 'aedat_for_testing'
		dir_path = os.path.join(os.getcwd(), 'test_files')

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

		file = AedatFile(heads,seq,labels,seq_clipped,filename)
		af.pickle_aedat(file,dir_path)
		af.save_aedat_file(file,dir_path)

		file_check = af.load_aedat_pickle(os.path.join(dir_path,filename)+'.pickle')

		print('\n')
		self.assertEqual(heads, file_check.event_head)
		print('Heads are equal(1/5)')
		self.assertEqual(seq, file_check.event_seq)
		print('Sequences are equal(2/5)')
		self.assertEqual(labels, file_check.event_labels)
		print('Labels are equal(3/5)')
		self.assertEqual(seq_clipped, file_check.seq_clipped)
		print('Clipped Sequences are equal(4/5)')
		self.assertEqual(filename, file_check.filepath)
		print('Filenames are equal(5/5)')
		print('Pickle Aedat Test has been passed(load/store)!\n')

	def test3_create_file_header(self):
		'''
		Binary header file test using artificial file header in the form of a list
		containing the start time as @ugurc's birthday.
		'''
		if not os.path.exists('test_files'):
			os.makedirs('test_files')

		head_path = os.path.join('test_files','head_test')

		time_defined = datetime(1995,5,9,14,20,48,tzinfo=timezone(timedelta(hours=3)))
		start_defined = AedatFile([],[],[],[],'', time_defined)

		head_list = [b'#!AER-DAT3.1\r\n', 
								 b'#Format: RAW\r\n', 
								 b'#Source 1: Test\r\n', 
								 b'#Start-Time: 1995-05-09 14:20:48 (TZ+0300)\r\n', 
								 b'#!END-HEADER\r\n']
	 	# Write
		try:
			with open(head_path, 'ab') as f:
				af.create_file_header(f,start_defined.header_line)
		except:
			print(f'Unable to create {head_path}!')

		else:
			print(f'{head_path} has been created succesfully!')

		# Read
		try:
			with open(head_path, 'rb') as f:
				header_line = pa.get_header_line_aedat(f)
		except:
			print(f'Unable to read {head_path}')

		self.assertEqual(head_list, header_line)
		print('File header test has been passed!\n')

	def test4_create_event_header(self):
		'''
		Event header creation test using an articifially created proper event header.
		The event header is created and stored in an aedat file. Nothing else. 
		'''
		if not os.path.exists('test_files'):
			os.makedirs('test_files')

		header_path = os.path.join('test_files','header_test')
		
		head_keys = ['eventType', 'eventSource', 'eventSize',
                 'eventTSOffset', 'eventTSOverflow',
             		 'eventCapacity', 'eventNumber', 'eventValid']
		
		head = [1, 1, 8, 4, 0, 5, 5, 5]             		 	
		header_check = dict(zip(head_keys, head))

		# Write
		try:
			with open(header_path, 'ab') as f:
				af.create_event_header(f,head)
		except:
			print(f'Unable to create {header_path}!')

		else:
			print(f'{header_path} has been created succesfully!')

		# Read
		try:
			with open(header_path, 'rb') as f:
				header_line = pa.get_event_header_aedat(f)
		except:
			print(f'Unable to read {head_path}')

		self.assertEqual(header_check, header_line)
		print('Event header test has been passed!\n')

	def test5_create_polarity_event(self):
		'''
		Polarity event creationtest using an articifially created proper event.
		The event is created and stored in an aedat file. Nothing else. 
		'''
		if not os.path.exists('test_files'):
			os.makedirs('test_files')

		polar_path = os.path.join('test_files','polarity_test')		
		_event = [19950509, 59, 22, 1]         		 	
		
		# Write
		try:
			with open(polar_path, 'ab') as f:
				af.create_polarity_event(f,_event)
		except:
			print(f'Unable to create {polar_path}!')

		else:
			print(f'{polar_path} has been created succesfully!')

		# Read
		try:
			with open(polar_path, 'rb') as f:
				event = pa.get_polarity_event_aedat(f,1,8)
		except:
			print(f'Unable to read {polar_path}')

		_event = [np.array([element], dtype = np.uint32) for element in _event]
		self.assertEqual(event, _event)
		print('Polarity event test has been passed!\n')

if __name__=='__main__':
	env.clear_env()
	env.aedat_file_env()
	unittest.main()