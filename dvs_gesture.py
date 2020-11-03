import numpy as np
import pickle
import progressbar
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from torchneuromorphic.utils import aedat_to_events


data_dir = '/home/ugurc/drive/data/DvsGesture'
train_dir = '/trials_to_train.txt'
test_dir = '/trials_to_test.txt'
dbs_dir = '/_dbs.pkl'

def filename_list(is_train_set):
	''' 
	Creates a list of filenames of .aedat files to train or
	to test the model

		Arguments:
			is_train_set(bool): interested in train set or the test set

		Return:
			_list(string): list of global paths to .aedat files 
	'''
	file_dir = train_dir if is_train_set else test_dir
	_list = []

	try:
		with open(data_dir+file_dir, 'r') as f:
			filenames = f.readlines()

			for name in filenames:
				if '.aedat' in name:
					_list.append(data_dir+'/'+name.replace('\n',''))

	except:
		print("Unable to open the file {}".format(file_dir))

	return _list

class DVSGesture(Dataset):

  # Initialize your data, download, etc.
  def __init__(self, root_dir, is_train_set=False):
  	self.filenames = self.filename_list(is_train_set)

  # def __getitem__(self, index):

  # def __len__(self):
  #   return self.len

# def event_dbs(load=False, save=False):

# 	dbs = {'train':{'events':[],
# 									'labels':[]},
# 				 'test': {'events':[],
# 									'labels':[]}};

# 	if load:
# 		dbs = load_dbs()

# 	else:
# 		print("Training database creation...")
# 		train_filenames = filename_list(is_train_set=True)
# 		dbs['train'] = extract_events(train_filenames)

# 		print("Test database creation...")
# 		test_filenames = filename_list(is_train_set=False)
# 		dbs['test'] = extract_events(test_filenames)
# 		if save:
# 			save_dbs(dbs)

# 	return dbs;

# def extract_events(filenames):
# 	_events = []
# 	_labels = []

# 	bar = progressbar.ProgressBar(maxval=len(filenames), \
# 		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
# 	bar.start()

# 	for i,name in enumerate(filenames):
# 		events, labels = aedat_to_events(name)
# 		bar.update(i+1)
# 		for label in labels:
# 			i_start = np.searchsorted(events[:,0], label[1], side='left')
# 			i_end = np.searchsorted(events[:,0], label[2], side='left')
# 			event = events[i_start:i_end]
# 			_events.append(event)
# 			_labels.append(label[0])

# 	print("\n{} Event sequence&label pairs have been extracted succesfully!".format(len(_labels)))

# 	return {'events':_events, 'labels':_labels}

# def load_dbs():
# 	events_dir = data_dir+dbs_dir
# 	try:
# 		with open(events_dir, 'rb') as f:
# 			dbs = pickle.load(f)
# 	except:
# 		print("Unable to open the {} file for loading!".format(events_dir))

# 	else:
# 		print("Database has been loaded from {}".format(events_dir))

# 	return dbs

# def save_dbs(dbs):
# 	events_dir = data_dir+dbs_dir
# 	try:
# 		with open(events_dir, 'wb') as f:
# 			pickle.dump(dbs, f, protocol=pickle.HIGHEST_PROTOCOL)

# 	except:
# 		print("Unable to open {} file for writing!".format(events_dir))

# 	else:
# 		size_byte = os.path.getsize(events_dir)
# 		print("{} GB of database has been saved in {}".format(size_byte/1073741824, events_dir))

	

# def event_dbs(filenames, events_dir=None, labels_dir=None, load=False):

# 	if load:
# 		if (not events_dir) or (not labels_dir):
# 			print("If dbs is already exist, events_dir and labels_dir needs to be provided!")
# 			return [],[]

# 		_events, _labels = load_dbs(events_dir, labels_dir)

# 		if (_events.empty() or _labels.empty()):
# 			print("Error while loading the dbs")

# 	else: 
# 		_events, _labels = file_to_dbs(filenames)

# 		if (events_dir) and (labels_dir):
# 			save_dbs(_events, events_dir, _labels, labels_dir)

# 	return _events, _labels

# def file_to_dbs(filenames):
# 	_events = []
# 	_labels = []

# 	for name in filenames:

# 		events, labels = aedat_to_events(name)
# 		for label in labels:
# 			i_start = np.searchsorted(events[:,0], label[1], side='left')
# 			i_end = np.searchsorted(events[:,0], label[2], side='left')
# 			event = events[i_start:i_end]
# 			_events.append(event)
# 			_labels.append(label[0])

# 		return _events, _labels

# def save_dbs(events, events_dir, labels, labels_dir):
# 	try:
# 		with open(events_dir, 'wb') as f:
# 			pickle.dump(events, f)
# 	except:
# 		print("Unable to open the {} file for writing!".format(events_dir))

# 	try:
# 		with open(labels_dir, 'wb') as f:
# 			pickle.dump(labels, f)
# 	except:
# 		print("Unable to open the {} file for writing!".format(labels_dir))

# def load_dbs(events_dir, labels_dir):
# 	_events = []
# 	_labels = []

# 	try:
# 		with open(events_dir, 'rb') as f:
# 			_events = pickle.load(f)
# 	except:
# 		print("Unable to open the {} file for loading!".format(events_dir))

# 	try:
# 		with open(labels_dir, 'rb') as f:
# 			_labels = pickle.load(f)
# 	except:
# 		print("Unable to open the {} file for loading!".format(labels_dir))

# 	return _events, _labels


def extract_events(filenames):
	_events = []
	_labels = []

	bar = progressbar.ProgressBar(maxval=len(filenames), \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	# bar.start()

	for i,name in enumerate(filenames):
		events, labels = aedat_to_events(name)
		# bar.update(i+1)
		print(i)
		for label in labels:
			# print(label)
			i_start = np.searchsorted(events[:,0], label[1], side='left')
			i_end = np.searchsorted(events[:,0], label[2], side='left')
			event = events[i_start:i_end]
			_events.append(event)
			_labels.append(label[0])
		break;

	print("\n{} Event sequence&label pairs have been extracted succesfully!".format(len(_labels)))

	return {'events':_events, 'labels':_labels}

if __name__ == '__main__':
	# for name in filename_list(True):
	# 	print(name)

	# for name in filename_list(False):
	# 	print(name)

	# records, labs = event_dbs(filename_list(False), events_dir='/try_events.pkl', labels_dir='/try_labels.pkl', load=True)

	# for rec in records:
	#   print(rec)

	# for lab in labs:
	#   print(lab)

	test_aedat = filename_list(False)
	x = extract_events(test_aedat)	

	for i,label in enumerate(x['labels']):
		print(i, end=' : ')
		print(label)

	for i,event in enumerate(x['events']):
		print(i, end=' : ')
		print(event)
	# x = event_dbs(load=True, save=False)



	