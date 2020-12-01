'''
Includes dataloader

author: ugurc
201201
'''

import hdf5_dbs_aedat as dbs

import torch
from torch.utils.data import Dataset

import os
import h5py
import numpy as np

class DVSGesture(Dataset):

	# Initialize your data, download, etc.
	def __init__(self, root_dir, hdf5_dir='dvs_gestures.hdf5', is_train_set=False, transform=None):
		'''

			Arguments:
				root_dir(str): 
					path to the directory of the file to be processed
		'''
		self.root_dir = root_dir
		self.hdf5_dir = hdf5_dir
		self.subset = 'train_set' if is_train_set else 'test_set'

		# Check if root_dir includes the dataset
		if not os.path.isdir(root_dir):
			print(f'The path {root_dir} is not valid')
			return

		# Check if hdf created
		if not os.path.isfile(hdf5_dir):
			print(f'The hdf5 database in {hdf5_dir} is missing. It will be created just for once.')
			dbs.create_dbs(root_dir, hdf5_dir)

		self.transform = transform

	def __getitem__(self, idx):
		'''
		Not a good method but I can not see the future so keep it easy for now.
		Returns tuples of lists
		'''
		# if torch.is_tensor(idx):
		# 	idx = idx.tolist()
		# if isinstance(idx, int):
		# 	idx = [idx]

		# time = []
		# pos = []
		# label = []
		# with h5py.File(self.hdf5_dir,'r') as hdf:
		# 	for index in idx:
		# 		time.append(np.array(hdf.get('{}/{}/time'.format(self.subset,index))))
		# 		pos.append(np.array(hdf.get('{}/{}/pos'.format(self.subset,index))))
		# 		label.append(np.array(hdf.get('{}/{}/label'.format(self.subset,index))))

		with h5py.File(self.hdf5_dir,'r') as hdf:
			time = np.array(hdf.get('{}/{}/time'.format(self.subset,idx)))
			pos = np.array(hdf.get('{}/{}/pos'.format(self.subset,idx)))
			label = np.array(hdf.get('{}/{}/label'.format(self.subset,idx)))

		return time, pos, label

	def __len__(self):
		with h5py.File(self.hdf5_dir,'r') as hdf:
			res = hdf[self.subset].attrs['instances']
		return res;


# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths):
	seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max()))
	for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
		seq_tensor[idx, :seq_len] = torch.Tensor(seq)
	return seq_tensor

def collate_fn(batch):
		# time_list, pos_list, label_list = [], [], []
		time_list, label_list = [], []
		lengths = []
		for _time, _pos, _label in batch:
			time_list.append(_time)
			# pos_list.append(_pos)
			label_list.append(int(_label))
			lengths.append(_pos.shape[0])

		time_list = pad_sequences(np.array(time_list,dtype='int32'), np.array(lengths))
		# pos_list = pad_sequences(np.array(pos_list,dtype='int32'), np.array(lengths))
		# return data_list, label_list
		# return time_list, pos_list, torch.Tensor(np.array(label_list, dtype='uint8'))
		return time_list, torch.Tensor(np.array(label_list, dtype='uint8'))