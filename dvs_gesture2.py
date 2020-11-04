import numpy as np
import progressbar
import time
import os
import struct

def get_filelist(root_dir, file_dir):
  ''' 
  Creates a list of filenames of .aedat files which are listed in a .txt file
  Also reports the time passed between start and the end

  Arguments:
    file_dir(string): text file of interest

  Return:
    file_list(string): list of global paths to .aedat files 
  '''

  if not os.path.exists(root_dir+file_dir):
	  print(f'{file_dir} does not exist')
	  return

  tic = time.perf_counter()
  file_list = []

  try:
    with open(root_dir+file_dir, 'r') as f:
      filenames = f.readlines()

      for name in filenames:
        if '.aedat' in name:
          file_list.append(root_dir+'/'+name.replace('\n',''))

  except:
    print("Unable to open the file {}".format(file_dir))

  else:
    toc = time.perf_counter()
    print(f'\n"{root_dir+file_dir}" has been processed in {toc-tic:0.4f} seconds!')

  return file_list

def aedat_to_np(filepath):
	'''
	Extract labeled event sequences from the given .aedat file
	each event is given in the form of 
	[timestep(us), x_position, y_position, polarization]

		Arguments:
			filepath(string): path to .aedat file

		Returns:
			events, labels tuple:
				events: 1D array of 2D variable size arrays consisting of 
					event sequences [[[t,x,y,p],[t,x,y,p]...],[[t,x,y,p],[t,x,y,p]...]...]
				labels: 1D array of event labels for the ones in events array.

		NOTE: the .csv file which is reserved for labels of the .aedat 
		has to be in the same root directory.
	'''

	# Initial requirement satisfaction
	if not os.path.exists(filepath):
		print(f'{filepath} does not exist')
		return np.array([]),np.array([])

	print(f'"{filepath}" is being processed...', end=' ')
	tic = time.perf_counter()


	label_filepath = filepath.replace('.aedat', '_labels.csv')

	if not os.path.exists(label_filepath):
		print(f'{label_filepath} does not exist')
		return np.array([]),np.array([])

	labels = np.genfromtxt(label_filepath, delimiter=',', dtype=np.uint32)

	# File Reading
	try:
		events = []
		with open(filepath, 'rb') as f:
			header_line = get_header_line_aedat(f)
			while True:
				event_header = get_event_header_aedat(f)
				if (len(event_header)==0):
					break
				if (event_header['eventType'] == 1): #polarity_event
					event = get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
					events.append(event)
				else:
					f.read(event_header['eventNumber']*event_header['eventSize'])

		events, labels = arrange_aedat_events(events,labels)

	except:
		print("Unable to read the file {}".format(filepath))
		return np.array([]),np.array([])

	else:
		toc = time.perf_counter()
		print(f'in {toc-tic:0.4f} seconds!')
		return events, labels
  
def get_header_line_aedat(aedat_file):
	'''
	AEDAT 3.1 file format includes 5 special lines at the beginning of the file 
	This function processes the header lines and keeps them in a list.
	The header lines are:

		1 - Version header line : 
			#!AER-DAT3.1\r\n
		2 - Format header line : 
			#Format: <FORMAT>\r\n.
		3 - Source Identifier header line : 
			human-readable identifiers for all event source IDs 
			present in the file are a mandatory part of the header. 
				#Source <ID>: <DESCRIPTION>\r\n
				Example: #-Source 0: DVS128\r\n
		4 - Start Time header line : 
			Encodes the time at which we started transmitting or logging data. 
				#Start-Time: %Y-%m-%d %H:%M:%S (TZ%z)\r\n
		5 - End of header line: 
			#!END-HEADER\r\n

	For more please refer to:
		https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html#aedat-31

		Arguments:
			aedat_file(io_file): file to be read

		Returns:
			list of raw header lines 
			[version,format,source,start_time,end_line]
	'''
	try:
		_version = aedat_file.readline()
		_format = aedat_file.readline()
		_source = aedat_file.readline()
		_start_time = aedat_file.readline()
		_end_line = aedat_file.readline()

	except:
		print("Unable to get the header lines!")
		return []

	else:
		return [_version,_format,_source,_start_time,_end_line]

def get_event_header_aedat(aedat_file):
	'''
	DVS data is stored in the AEDAT 3.1 file format as Polarity Events.
	Like [header] [events] [header] [events] [header] [events]...
	This function process the header data and creates a dictionary.
	The featues in header are:

	Bytes	| Meaning	        | Description
	---------------------------------------------------------------------------------------
	0-1   | eventType	      | Numerical type ID, unique to each event type 
													+ (see ‘Event Types’ table).
	2-3	  | eventSource     |	Numerical source ID, identifies who generated 
													+ the events inside a system.
	4-7   |	eventSize       | Size of one event in bytes.
	8-11	| eventTSOffset   |	Offset from the start of an event, in bytes, at which 
													+ the main 32 bit time-stamp can be found.
	12-15	| eventTSOverflow	| Overflow counter for the standard 32bit event time-stamp. 
													+ Used to generate the 64 bit time-stamp.
	16-19	| eventCapacity   |	Maximum number of events this packet can store. 
													+ **This always equals eventNumber in files and streams, 
													+ It can only have a different value for in-memory packets.
	20-23	| eventNumber	    | Total number of events present in this packet(valid+invalid)
	24-27	| eventValid	    | Total number of valid events present in this packet.

	For more please refer to:
	  https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html#aedat-31
		
		Arguments:
			aedat_file(io_file): file to be read

		Returns:
			dictionary of event features stated above
			the features are eiher uint16 or uint32
	'''
	try:
		raw_header = aedat_file.read(28)
		if (len(raw_header)==0):
			# print("END OF FILE!")
			return {}

		event_header = {}
		event_header['eventType'] = struct.unpack_from('<H', raw_header, 0)[0]
		event_header['eventSource'] = struct.unpack_from('<H', raw_header, 2)[0]
		event_header['eventSize'] = struct.unpack_from('<I', raw_header, 4)[0]
		event_header['eventTSOffset'] = struct.unpack_from('<I', raw_header, 8)[0]
		event_header['eventTSOverflow'] = struct.unpack_from('<I', raw_header, 12)[0]
		event_header['eventCapacity'] = struct.unpack_from('<I', raw_header, 16)[0]
		event_header['eventNumber'] = struct.unpack_from('<I', raw_header, 20)[0]
		event_header['eventValid'] = struct.unpack_from('<I', raw_header, 24)[0]

	except:
		print("Unable to get the event header!")
		return {}

	else:
		return event_header

def get_polarity_event_aedat(aedat_file,eventNumber,eventSize):
	'''
	DVS data is stored in the AEDAT 3.1 file format as Polarity Events.
	Like [header] [events] [header] [events] [header] [events]...

	This function process the events data and creates 
	a 2D array shaped [n, 4] 	each line in the form of 
	[timestep(us), x_position, y_position, polarization]

	The Polarity Event includes:

	Bytes	| Meaning	         | Description
	----------------------------------------------------------------------------
	0-3	  | 32 bit data	     | Holds information on the polarity 
													 + (luminosity change) event.
	4-7	  | 32 bit timestamp | Event-level microsecond timestamp.

	An events block contains eventNumber of events embedded in eventSize bytes.

	For more please refer to:
	  https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html#aedat-31
		
		Arguments:
			aedat_file(io_file): file to be read
			eventNumber: number of event to be read
			eventSize: size of each event in bytes

		Returns:
			2D list of numpy arrays shaped [n, 4] each line in the form of 
			[timestep(us), x_position, y_position, polarization]
	'''
	events = []
	event_bytes = np.frombuffer(aedat_file.read(eventNumber*eventSize), 'uint32')
	event_bytes = event_bytes.reshape(-1,2)

	x = (event_bytes[:,0] >> 17) & 0x00001FFF
	y = (event_bytes[:,0] >> 2 ) & 0x00001FFF
	p = (event_bytes[:,0] >> 1 ) & 0x00000001
	t = event_bytes[:,1]
	return [t,x,y,p]

def arrange_aedat_events(events,labels):
	'''
	Arranges events and labels in such a way that 

	The final events structure
	includes mutually exclusive labeled event sequences with a series of
	timestep, xpos, ypos and polarity. Since each aedat file inlcudes 12 event sequences,
	the events numpy array includes 12 2D variable length numpy arrays.

	The final labels structure
	is a 1D array of labels 

		Arguments:
			events(3D list of event sequences in eventBlocks,4{t,x,y,p},eventNumber):
				eventNumber is variable for all event blocks
			labels(2D list of [event labels, start times, stop times]):


		Returns:
			events(1D np.array of variable size 2D np.arrays of type np.uint32):
				in the shape of 12{number of seperately labeled events},events,4{t,x,y,p}
				events are separated according to labels instead of event block lengths.
				Also unlabeled event samples are excluded.
			labels(1D np array of type np.uint8):
				since the start and stop times of events are clear, only the event labels 
				present. 

	NOTE : The interval is [startTime_usec, endTime_usec)
	'''
	labels = labels[1:]
	events = np.column_stack(events).T
	events = events.astype(np.uint32)
	_events = []

	for l in labels:
	  i_start = np.searchsorted(events[:,0], l[1], side='left')
	  i_end = np.searchsorted(events[:,0], l[2], side='left')
	  event = events[i_start:i_end]
	  _events.append(np.array(event,dtype=np.uint32))

	return np.array(_events), labels[:,0].astype(np.uint8)

# def extract_events(filenames):
# 	'''
# 	TO BE COMMENTED AND TESTED
# 	'''
#   tic = time.perf_counter()
#   _events = []
#   _labels = []

#   widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
#   bar = progressbar.ProgressBar(maxval=len(filenames), widgets = widgets).start()

#   for i,name in enumerate(filenames):
#     events, labels = aedat_to_events(name)
#     for label in labels:
#       i_start = np.searchsorted(events[:,0], label[1], side='left')
#       i_end = np.searchsorted(events[:,0], label[2], side='left')
#       event = events[i_start:i_end]
#       _events.append(np.array(event,dtype=np.uint32))
#       _labels.append(label[0])
#     bar.update(i+1)

#   bar.finish()
#   toc = time.perf_counter()
#   print(f"\n{len(_labels)} Event sequence&label pairs have been extracted succesfully in {toc-tic:0.4f} seconds!")

#   return np.array(_events), np.array(_labels,dtype=np.uint8)

if __name__ == '__main__':
	test_file = '/home/ugurc/drive/data/DvsGesture/user01_fluorescent.aedat'
	
	# events, labels = aedat_to_np(test_file)
	# print(type(events))
	# print(type(labels))
	# print(events.shape)
	# print(events[0].shape)

	# print(events)
	# print(labels)

	# print(np.vstack(events))
