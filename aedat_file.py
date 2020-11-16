'''
Utility functions and AedatFile class definition to create and save articial 
aedat files for test purposes. 

All functions have been tested(only proof of concept not an extensive test)
and the test file is stored in the test folder.

The datasheet of the AEDAT file format is in 
https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html

author: @ugurc
201030
'''

import numpy as np
from datetime import datetime, timezone
import os
import pickle

class AedatFile():
	'''
	Aedat File class definition.
	Used for creating and saving aedat files for test purposes.
	One can save the aedat file by pickling the object. 
	In that case, event headers, event sequences, labels, and clipped sequences are 
	easy to use. One can see directly the lists including all the elements. 
	One can save the aedat file in the AEDAT 3.1 format. In that case, 
	proper processing of the compressed form is required. 
	'''
	def __init__(self, event_head, event_seq, event_labels, seq_clipped, filepath, start_time=None):
		'''
		Aedat File object designed to replicate an original aedat record.
			Arguments:
				event_head(list of list of int):
					event headers in the form a list
						example : [[1, 1, 8, 4, 0, 5, 5, 5],
	                     [1, 1, 8, 4, 0, 4, 4, 4],
	                     [1, 1, 8, 4, 0, 3, 3, 3]]

				event_seq(list of list of list of int):
					list of sequences of event records([t,x,y,pol])
						example : [[[10,0,0,1],[25,10,20,0],[38,120,59,1],[48,5,9,0],[59,9,5,1]],
	                     [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1],[126562,45,59,1]],
	                     [[2068981,127,127,0],[2157619,0,15,1],[2178812,1,1,1]]]

				event_labels(list of list of int):
					list of [event label, start_time(usec), end_time(usec)] for events to be recorded
						example : [[1,24,50],
	                     [5,102865,126562],
	                     [11,2157620,2178812]]

				event_seq(list of list of list of int):
					list of clipped sequences of event records([t,x,y,pol]) according to labels
						example : [[[25,10,20,0],[38,120,59,1],[48,5,9,0]],
											 [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1]],
											 []]

				filepath(string):
					filepath of the future .aedat and _labels.csv file

				start_time(datetime object):
					datatime object to be used as start time
		'''
		self._version = b'#!AER-DAT3.1\r\n'
		self._format = b'#Format: RAW\r\n'
		self._source = b'#Source 1: Test\r\n'
		self._start_time = self.set_start_time(start_time)
		self._end_line = b'#!END-HEADER\r\n'
	
		self.header_line = self._version+self._format+self._source+\
											 self._start_time+self._end_line

		self.event_head_keys = ['eventType', 'eventSource', 'eventSize',
			                      'eventTSOffset', 'eventTSOverflow',
			                      'eventCapacity', 'eventNumber', 'eventValid']

		self.label_header = "class,startTime_usec,endTime_usec"

		self.event_head = event_head
		self.event_seq = event_seq
		self.seq_clipped = seq_clipped
		self.event_labels = event_labels
		self.filepath = filepath 

	def set_start_time(self, start_time):
		'''
		Sets start time of the event record which is to be reside in the 
		file header. It could either be defined by the user or could be 
		left to the computer to stamp the current time and date.
			Arguments:
				start_time(datetime object):
					datetime object containing the time to be stamped into the
					header line. It could be None! In that case, the current time
					is used. 
			Returns:
				_start_time(binary array):
					byte array of UTF-8 chars representing the start time line 
		'''
		if not start_time:
			today = datetime.now(timezone.utc).astimezone()
			current_time = today.strftime("%Y-%m-%d %H:%M:%S (TZ%z)")
		else:
			current_time = start_time.strftime("%Y-%m-%d %H:%M:%S (TZ%z)")
		
		_start_time = b'#Start-Time: '
		_start_time += bytearray(current_time, "utf8")
		_start_time += b'\r\n'
		return _start_time

def pickle_aedat(aedat_file, filepath=None):
	'''
	Utility funciton to dump the object in the form of a pickle.
	In that case one can easily load and use the data inside the 
	object created. It won't create the .pickle file if it already 
	exists.

	Arguments:
		aedat_file(AedatFile object):
			aedat file to be saved in AEDAT 3.1 format
	'''
	if not filepath:
		pickle_path = os.path.join(os.getcwd(),aedat_file.filepath+'.pickle')
	else:
		pickle_path = os.path.join(filepath,aedat_file.filepath+'.pickle')
	if os.path.exists(pickle_path):
		print(f'{pickle_path} already exist!')
		return
	try:
		with open(pickle_path,'wb') as f:
			pickle.dump(aedat_file, f, pickle.HIGHEST_PROTOCOL)
	except:
		print(f'AedatFile object could not be stored in {pickle_path}')
	else:
		print(f'AedatFile object has been succesfully pickled in {pickle_path}\n')

def save_aedat_file(aedat_file, filepath=None):
	'''
	Creates an aedat file and it's labels to test the functions 
	defined to read a real one. The function won't create files 
	if they are already exist.

	Arguments:
		aedat_file(AedatFile object):
			aedat file to be saved in AEDAT 3.1 format
	'''
	if not filepath:
		aedat_path = os.path.join(os.getcwd(),aedat_file.filepath+'.aedat')
		label_path = os.path.join(os.getcwd(),aedat_file.filepath+'_labels.csv')

	else:
		aedat_path = os.path.join(filepath,aedat_file.filepath+'.aedat')
		label_path = os.path.join(filepath,aedat_file.filepath+'_labels.csv')

	# EVENTS
	if os.path.exists(aedat_path):
		print(f'{aedat_path} has already exist!')

	else:
		if (len(aedat_file.event_head) != len(aedat_file.event_seq)):
			print(f'length of event header ({len(aedat_file.event_head)}) is \
							different from length of event sequence{len(aedat_file.event_seq)}')
			print("Failure!")
			return
		event_it = zip(aedat_file.event_head,aedat_file.event_seq)
		try:
			with open(aedat_path, 'ab') as f:
				create_file_header(f,aedat_file.header_line)
				for head, seq in event_it:
					create_event_header(f,head)
					for event in seq:
						create_polarity_event(f,event)
		except:
			print(f'Unable to create {aedat_path}!')

		else:
			print(f'{aedat_path} has been created succesfully!')

	# LABELS
	if os.path.exists(label_path):
		print(f'{label_path} has already exist!')
	else:
		try:
			np.savetxt(label_path, np.asarray(aedat_file.event_labels), 
								 fmt='%u', delimiter=',', 
								 header=aedat_file.label_header, comments='')
		except:
			print(f'Unable to create {label_path}!')

		else:
			print(f'{label_path} has been created succesfully!')

def create_file_header(_f,file_header):
	'''
	Writes the header line to the given file.
	It's a slave function called by create_test_files()

		Arguments:
			_f(output file): 
				file to be written
			file_header(byte array):
				header of the aedat file
	'''
	try:
		_f.write(file_header)
	except:
		print('Unable to write the header line!')

def create_event_header(_f,header):
	'''
	Writes the given event header to the given file.
	It's a slave function called by create_test_files()

	Converts integer values to their little endian binary representation

		Arguments:
			_f(output file): 
				file to be written
			header(int list):
				header in the form of an integer list
	'''

	eventType = header[0].to_bytes(2,byteorder='little')
	eventSource = header[1].to_bytes(2,byteorder='little')
	eventSize = header[2].to_bytes(4,byteorder='little')
	eventTSOffset = header[3].to_bytes(4,byteorder='little')
	eventTSOverflow = header[4].to_bytes(4,byteorder='little')
	eventCapacity = header[5].to_bytes(4,byteorder='little') 
	eventNumber = header[6].to_bytes(4,byteorder='little')
	eventValid = header[7].to_bytes(4,byteorder='little')

	try:
		_f.write(eventType+eventSource+eventSize+\
						 eventTSOffset+eventTSOverflow+eventCapacity+\
						 eventNumber+eventValid) 
	except:
		print(f'Unable to write the event header {header}')

def create_polarity_event(_f,seq):
	'''
	Writes the given event to the given file 
	It's a slave function called by create_test_files()

	Using the compression method defined in 
	https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html#aedat-31
	and 
	https://www.research.ibm.com/dvsgesture/,

	express 
	32 bits time step, 
	15 bits x position, 
	15 bits y position, 
	1 bit polarization,
	1 bit validity check
	in 64 bits  

		Arguments:
			_f(output file): 
				file to be written
			seq(int list):
				[time, xpos, ypos, pol] in the form of an integer list
	'''
	time = seq[0]
	data = 0

	data = data | seq[1]<<17
	data = data | seq[2]<<2
	data = data | seq[3]<<1

	try:
		_f.write(data.to_bytes(4,byteorder='little')+time.to_bytes(4,byteorder='little'))
	except:
		print(f'Unable to write polarity_event {seq}')

def load_aedat_pickle(filepath):
	'''
	Utility function to load the pickled AedatFile object 
	in given filepath.

		Arguments:
			filepath(string):
				filepath to the .pickle file including aedat object

		Returns:
			aedat(AedatFile object):
				the unpickled AedatFile object which was stored in the given filepath. 
	'''
	try:
		with open(filepath, 'rb') as f:
			aedat = pickle.load(f)
	except:
		print(f'Unable to load {filepath}!')
	else:
		print(f'{filepath} pickle succesfully loaded')
		return aedat