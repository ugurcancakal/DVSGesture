'''
'''
import process_aedat as pa
import time 
import numpy as np
import progressbar

def create_dbs(root_dir, 
							 hdf5_dir='/dvs_gestures.hdf5',
							 trials_to_train='/trials_to_train.txt', 
							 trials_to_test='/trials_to_test.txt'):

	print(f'{hdf_dir} database is begin created...')
	tic = time.perf_counter()
	train_filelist = pa.get_filelist(root_dir,trials_to_train)
	test_filelist = pa.get_filelist(root_dir,trials_to_test)

	try:
	  with h5py.File(hdf5_dir, 'w') as hdf:
	    set_group(hdf, 'train_set', train_filelist)
	    set_group(hdf, 'test_set', test_filelist)

	except:
	  print("Unable to open the file {}".format(hdf5_dir))

	else:
	  toc = time.perf_counter()
	  size_byte = os.path.getsize(hdf5_dir)
	  print(f'\n{(size_byte/10**9):0.4f} GB of database has been saved in \
	    "{hdf5_dir}" in {toc-tic:0.4f} seconds!')

def set_group(hdf, setname, filelist, keystart=0):

	print("\n{} in the database is being created...".format(setname))
	tic = time.perf_counter()
	trainset = hdf.create_group(setname)
	events, labels = extract_events(filelist)
	print("Insertion has started!")

	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(labels), widgets = widgets).start()

	for i,event in enumerate(events, start=keystart):

		event_seq = hdf.create_group('{}/{}'.format(setname,i))
		event_seq.create_dataset('time', data=event[:,0])
		event_seq.create_dataset('pos', data=event[:,1:])
		event_seq.create_dataset('label', data=labels[i])
		bar.update(i+1)

	bar.finish()
	toc = time.perf_counter()
	print(f"\n{len(labels)} Event sequences have been extracted succesfully in {toc-tic:0.4f} seconds!")


def extract_events(filenames):
	'''
	TO BE COMMENTED AND TESTED
	'''
	tic = time.perf_counter()
	_events = np.empty(0)
	_labels = np.empty(0, dtype=np.uint8)

	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(filenames), widgets = widgets).start()

	for i,name in enumerate(filenames):
		events, labels = pa.aedat_to_np(name)
		_events = np.concatenate((_events, events))
		_labels = np.concatenate((_labels, labels))
		bar.update(i+1)

	bar.finish()
	toc = time.perf_counter()
	print(f"\n{len(_labels)} Event sequence&label pairs have been extracted succesfully in {toc-tic:0.4f} seconds!")

	return _events, _labels

if __name__ == '__main__':
	test_file = '/home/ugurc/drive/data/DvsGesture/user01_fluorescent.aedat'
	test_file2 = '/home/ugurc/drive/data/DvsGesture/user02_fluorescent.aedat'
	# _events = np.empty(0)
	# _labels = np.empty(0, dtype=np.uint8)

	# print(type(_events))
	# print(type(_labels))
	# print(_events.shape)
	# print(_events[0].shape)

	# print(_events)
	# print(_labels)

	# events, labels = aedat_to_np(test_file)
	# _events = np.concatenate((_events, events))
	# _labels = np.concatenate((_labels, labels))

	# print(type(_events))
	# print(type(_labels))
	# print(_events.shape)
	# print(_events[0].shape)

	# print(_events)
	# print(_labels)

	# events2, labels2 = aedat_to_np(test_file2)
	# print(type(events2))
	# print(type(labels2))
	# print(events2.shape)
	# print(events2[0].shape)

	# print(events2)
	# print(labels2)

	# events = np.concatenate((events, events2))
	# labels = np.concatenate((labels, labels2))
	# print(type(events))
	# print(type(labels))
	# print(events.shape)
	# print(events[0].shape)

	# print(events)
	# print(labels)

	root_dir = '/home/ugurc/drive/data/DvsGesture'
	trials_to_train = '/trials_to_train.txt'
	trials_to_test = '/trials_to_test.txt'

	test_filelist = pa.get_filelist(root_dir,trials_to_test)
	events, labels = extract_events(test_filelist)

	print(type(events))
	print(type(labels))
	print(events.shape)
	print(events[0].shape)

	print(events)
	print(labels)


	# print(np.vstack(events))
