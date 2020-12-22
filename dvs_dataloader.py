'''
Utility functions to create images and video 
using of label, position and time sequences in aedat events 
and also custom pytorch dataset class declaration for DVSGesture.

All functions have been tested(only proof of concept not an extensive test)
and the test file is stored in the test folder.

author: ugurc
201201
'''

import hdf5_dbs_aedat as dbs
import process_aedat as pa

import os
import h5py
import time
import progressbar
import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from cv2 import VideoWriter, VideoWriter_fourcc
from torch.utils.data import Dataset

class DVSGesture(Dataset):
	'''
	Custom pytorch dataset definition for DVSGesture128.
	It can be used to generate a memory efficient dataloader
	'''
	def __init__(self, root_dir, hdf5_path=None, is_train_set=False, transform=None):
		'''
		Initializer for the DVSGesture dataset.

		Checks if 
			-the root file is in the expected structue
			-essential files exists
			-hdf5 database has created

		In the case root file is missing or malstructured, it reports the situation
		and prints out a link to download.

		In the case hdf5 has not been created yet, it creates the dbs inside the
		raw data folder.

			Arguments:
				root_dir(str): 
					path to the directory of the file to be processed
				
				hdf5_path(str):
					directory to existing hdf5 database. default : 'None'

				
				is_train_set(bool):
					flag to decide on train or test set. default : False(test)

				transform(transform functor):
					a transformation function to operate on the data before getting an item
					default : None
		'''
		hdf5_path = hdf5_path if hdf5_path else os.path.join(root_dir, 'dvs_gestures.hdf5')
		self.root_dir = root_dir
		self.hdf5_path = hdf5_path
		self.subset = 'train_set' if is_train_set else 'test_set'

		if not os.path.isdir(root_dir):
			print(f'\nFAIL: The path {root_dir} does not exist!!!')
			self.download_dataset()
			return

		self.dataset_valid_check(root_dir)

		# Check if hdf created
		if not os.path.isfile(hdf5_path):
			print(f'The hdf5 database in {hdf5_path} is missing. It will be created just for once.')
			dbs.create_dbs(root_dir, hdf5_path)

		try:
			with h5py.File(self.hdf5_path,'r') as hdf:
				self.gesture_mapping = dict(zip(hdf.attrs['label_idx'], hdf.attrs['label_name']))
		except:
			print("\nFAIL: Unable to open and retrieve the gesture map from the hdf5 dataset!")
		
		self.transform = transform

	def __getitem__(self, idx):
		'''
		Returns the item stored in given id and apply transformation
		if a transform function is defined

			Arguments:

				idx(int):	
					The id of the sample tuple to be returned

			Returns:

				_time(1D np.ndarray of type np.uint32):
					time line of the event sequence consisting of elements representing
						the times that the event happened in microsecond resolution 

				_pos(2D np.ndarray of type np.uint8):
					2D positions of events in 2D coordinate space [xpos, ypos, pol]
					the same indexing with the time sequence

				_label(np.uint8):
					Label of the event happened. The same indexing with _time and _pos
		'''

		with h5py.File(self.hdf5_path,'r') as hdf:
			_time = np.array(hdf.get('{}/{}/time'.format(self.subset,idx)),dtype=np.uint32)
			_pos = np.array(hdf.get('{}/{}/pos'.format(self.subset,idx)),dtype=np.uint8)
			_label = np.uint8(hdf.get('{}/{}/label'.format(self.subset,idx)))

		if self.transform:
			_time, _pos, _label = self.transform(_time, _pos, _label)

		return _time, _pos, _label

	def __len__(self):
		'''
		Overwrites the len method to get the length value as len(dataset)
			Returns:
				__len__(int): number of samples in the dataset to be used.
		'''
		with h5py.File(self.hdf5_path,'r') as hdf:
			res = hdf[self.subset].attrs['instances']
		return res;

	def dataset_valid_check(self,root_dir):
		'''
		Check if root_dir includes the dataset in the expected structure
		To be a valid DVSGesture dataset, it need to include 
			-'trials_to_train.txt'
			-'trials_to_test.txt'
			-'gesture_mapping.csv'
		files.
		Also the .aedat files refered in 'trials_to_train.txt' and 'trials_to_test.txt'
		need to exist in the directory with their .csv companions

			Arguments:
				root_dir(str):
					the directory of the dataset DVSGesture
		'''
		trials_to_train = 'trials_to_train.txt'
		trials_to_test = 'trials_to_test.txt'
		gesture_path = 'gesture_mapping.csv'

		#check if the files referred in trials_to_train.txt trials_to_test.txt exist
		assert os.path.isfile(os.path.join(root_dir,trials_to_train)), \
			f'FAIL: {trials_to_train} is missing!!!'

		assert os.path.isfile(os.path.join(root_dir,trials_to_test)), \
			f'FAIL: {trials_to_test} is missing!!!'

		path_list = pa.get_filelist(root_dir,trials_to_train) + pa.get_filelist(root_dir,trials_to_test)
		csv_list = [path.replace('.aedat', '_labels.csv') for path in path_list]
		path_list = path_list+csv_list
		path_list.append(os.path.join(root_dir,gesture_path))

		for path in path_list:
			assert os.path.isfile(path), f'FAIL: {path} is missing!!!'

	def download_dataset(self):
		'''
		In the case that the DVSGesture dataset directory is missing,
		print out the method to download the tar.gz file and extract
		'''
		url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/file/211521748942'
		print(f"Download and extract(tar.gz) the dataset manually from: \n{url}")
		print(f"Then use the directory that the tar.gz file has extracted in the constructor like DVSGesture(root_dir='dataset_directory')")
		ibm_url = 'https://www.research.ibm.com/dvsgesture/'
		print(f'If the url above does not work, check: \n{ibm_url}')
	
def pad_sequences(seqs, seq_lengths):
	'''
	Pad sequences with 0 and fill torch tensors 
	increasing dimensionality by one.

		Arguments:
			seqs(list of np.ndarray):
				list of numpy arrays to be padded in one bigger tensor

			seq_lengths(np.ndarray): 
				lengths of sequences to be padded

		Returns:
			seq_tensor(torch.tensor):
				tensor of padded sequences
	'''
	seq_shape = seqs[0].shape[1:]
	_dtype=seqs[0].dtype
	seq_tensor = torch.zeros((len(seqs), seq_lengths.max(), *seq_shape),dtype=eval(f'torch.{_dtype}'))
	for idx, (seq, seq_len) in enumerate(zip(seqs, seq_lengths)):
		seq_tensor[idx, :seq_len] = torch.from_numpy(seq)
	return seq_tensor

def collate_fn(batch):
	'''
	Custom collate_fn to be used in torch.DataLoader. 
	Pad sequences to be able to create a torch tensor. 
	Since torch does not accept uint32, time sequence casted to int32.
	Necessary to use in batch operations 
	because of the variable length input arrays.
		
		Arguments:

			batch(list of (_time, _pos, _label) tuples):
				list created by iterative calling of __getitem__() of dataset

		Returns:

			time_tensor(2D torch.Tensor of dtype torch.int32):
				batch of time lines of the event sequences consisting of elements 
					representing	the times that the event happened in microsecond resolution 

			pos_tensor(3D torch.Tensor of dtype torch.uint8):
				batch of 2D positions of events in 2D coordinate space [xpos, ypos, pol]
					the same indexing with the time sequence

			list_tensor(1D torch.Tensor of dtype torch.uint8):
				batch of labels of the events happened. 
					the same indexing with _time and _pos
	'''
	time_list, pos_list, label_list = [], [], []
	lengths = []
	for _time, _pos, _label in batch:
		time_list.append(_time.astype(np.int32))
		pos_list.append(_pos)
		label_list.append(_label)
		lengths.append(_pos.shape[0])

	time_tensor = pad_sequences(time_list, np.array(lengths))
	pos_tensor = pad_sequences(pos_list, np.array(lengths))
	label_tensor = torch.from_numpy(np.array(label_list, dtype=np.uint8))

	return time_tensor, pos_tensor, label_tensor

def pos_to_frame(pos, scale, label, px_frame_shape=(128,128)):
	'''
	Creates an image out of the input position sequence.
	Use the positions given to create pixel values.
	Positive change(pol=1) in limunosity is represented by green and
	Negative change(pol=0) in limunosity is represented by blue.
	Also prints text on top of each frame.

		Arguments:

			pos(np.ndarray):
				sequence of [xpos, ypos, pol]'s to be converted into an image

			scale(int): 
				scaling coefficient, requires to be an integer to ease the operation

			label(str):
				label to be printed on top of the image 

			px_frame_shape(int,int):
				frame shape(height,width) in pixels. no control, choose carefully. 
				default : 128,128

		Returns:
			frame(3D np.ndarray):
				BGR image of the responsible position sequence
	'''

	# Type Checks
	if not isinstance(pos, np.ndarray):
		pos = np.asarray(pos,dtype=np.int32)
	if not isinstance(scale, int):
		scale = int(scale)
	if not isinstance(label, str):
		label = str(label)

	# Scaling
	px_frame_width = px_frame_shape[1]*scale
	px_frame_height = px_frame_shape[0]*scale
	text_x_pos = 0
	text_y_pos = 10*scale

	# Frame Generation
	frame=np.zeros((px_frame_width,px_frame_height,3),dtype=np.uint8)

	# Indexes
	pol_one = pos[np.asarray(pos[:,2],dtype=bool)]*scale # Green
	pol_zero = pos[np.asarray(1-pos[:,2],dtype=bool)]*scale # Blue

	for i in range(scale):
		for j in range(scale):
			frame[pol_one[:,1]+i,pol_one[:,0]+j,1] = 255 # Green
			frame[pol_zero[:,1]+i,pol_zero[:,0]+j,0] = 255 # Blue
			
	# Uncomment to create a rectangular black space for background of the text
	# cv2.rectangle(frame, (0, 0), (px_frame_width*scale, text_y_pos+10), (0,0,0), -1)
	cv2.putText(frame, label, (text_x_pos,text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.28*scale, (0,255,255), 1+(scale//2))
	return frame

def get_area_index(i, n_frame_width, scale, px_frame_shape=(128,128), px_margin=10):
	'''
	Get the coordinate limits(mask) for given frame width and number of frames
	in x coordinate while creating a grid consisting of multiple images
	with margins. Used in batch_frame function.

		Arguments:
			i(int):
				row-major ordering of the image of interest in the grid.

			n_frame_width(int): 
				number of frames on the width

			scale(int): 
				scaling coefficient, requires to be an integer to ease the operation

			px_frame_shape(int,int):
				frame shape(height,width) in pixels. no control, choose carefully. 
				default : 128,128

			px_margin(int):
				frame margin in pixels, default :10

		Returns:
			y_min,y_max,x_min,x_max(int): coordinate limits
	'''
	# Scaling
	px_frame_width = px_frame_shape[1]*scale
	px_frame_height = px_frame_shape[0]*scale
	px_margin = px_margin*scale

	# Shaping
	offset_width = (i%n_frame_width)*(px_frame_width+px_margin) # 0 1 2 0 1 2 
	offset_height = (i//n_frame_width)*(px_frame_height+px_margin) # 0 0 0 1 1 1 
	y_min = px_margin+offset_height
	y_max = px_margin+px_frame_height+offset_height
	x_min = px_margin+offset_width
	x_max = px_margin+px_frame_width+offset_width

	return y_min,y_max,x_min,x_max

def batch_frame(frames, scale, n_grid_shape=None, px_frame_shape=(128,128), px_margin=10):
	'''
	Merge images created by pos_to_frame(..) into a bigger one
	Try to make the final frame square if the n_grid_shape is not defined by the user.

		Arguments:
			frames(list of 3D np.ndarray):
				list of BGR images of position sequences

			scale(int): 
				scaling coefficient, requires to be an integer to ease the operation
			
			n_grid_shape(int,int):
				the grid shape(n_height,n_width) defined by the user. 
				n_width stands for the number of frames in x axis, and 
				n_height stands for the number of frames in y axis
				default : None 
					-> result in a square like grid as much as possible

			px_frame_shape(int,int):
				frame shape(height,width) in pixels. no control, choose carefully. 
				default : 128,128

			px_margin(int):
				frame margin in pixels, default :10

		Returns:
			frame(3D np.ndarray):
				one combined BGR image of the given batch of position sequences
	'''
	# Scaling
	px_frame_width = px_frame_shape[1]*scale
	px_frame_height = px_frame_shape[0]*scale
	px_margin = px_margin*scale

	# Shaping
	batch_size = len(frames)

	if n_grid_shape:
		n_frame_height,n_frame_width=n_grid_shape
	else:
		n_frame_width=math.ceil(math.sqrt(batch_size))
		n_frame_height=math.ceil(batch_size/n_frame_width)

	px_full_width=(n_frame_width*(px_frame_width+px_margin))+(px_margin)
	px_full_height=(n_frame_height*(px_frame_height+px_margin))+(px_margin)

	# Frame Generation
	frame=np.zeros((px_full_height,px_full_width,3),dtype=np.uint8)
	for i in range(batch_size):
		y_min,y_max,x_min,x_max = get_area_index(i,n_frame_width,scale)
		frame[y_min:y_max,x_min:x_max] = frames[i]

	return frame

def split_time(_time,incr):
	'''
	Take an array of time points and split the whole by incr amount
	Long story short: discretization
	For the list [12, 15, 21, 23, 26, 27, 32] if the incr = 10
	the function will return [0,3,6]

		Arguments:

			_time: (1D np.ndarray of type np.int32):
				the time line to be discretized

			incr(int): the smallest duration possible

		Returns:
			idx(list of int): indexes of the limiting timepoints.

	'''
	# Type Checks
	if isinstance(_time, torch.Tensor):
		_time = _time.numpy()

	if not isinstance(_time, np.ndarray):
		_time = np.array(_time, dtype=np.int32)

	_time = _time[_time != 0]
	limit = _time[-1]
	current = _time[0];

	idx = []

	while(current<limit):
		idx.append(np.searchsorted(_time, current, side='left'))
		current+=incr;

	idx.append(np.searchsorted(_time, current, side='left'))
	return idx;

def loader_video(data_loader,filepath,fps=24,scale=2,n_grid_shape=None,px_frame_shape=(128,128),px_margin=10):
	'''
	Captures the video from batched sequences of merged images of 
	position sequences created by batch_frame(..)
	One run of loader_video(..) takes like 60 seconds for now.

	Each event sequence has different number of events to be
	converted into pixels. To find the correct number, their time
	sequences are processed and splitted by incr amount of time.
	Resulting index arrays are used to split the pos sequences.

		Arguments:
			data_loader(torch.Dataloader):
				dataloader to be used to create the video.

			filepath(str):
				file path for the video to be saved.

			fps(frame per second):
				default : 24

			scale(int): 
				scaling coefficient, requires to be an integer to ease the operation
				default : 2
			
			n_grid_shape(int,int):
				the grid shape(n_height,n_width) defined by the user. 
				n_width stands for the number of frames in x axis, and 
				n_height stands for the number of frames in y axis
				default : None 
					-> result in a square like grid as much as possible

			px_frame_shape(int,int):
				frame shape(height,width) in pixels. no control, choose carefully. 
				default : 128,128

			px_margin(int):
				frame margin in pixels, default :10
	'''
	tic = time.perf_counter()
	# Prerequisites
	batch_size = data_loader.batch_size
	gesture_mapping = data_loader.dataset.gesture_mapping

	# Scaling
	px_frame_width = px_frame_shape[1]*scale
	px_frame_height = px_frame_shape[0]*scale
	incr = int(1e6)//fps

	# Dummy frame to get the of the video
	frames = [np.zeros((px_frame_width,px_frame_height,3),dtype=np.uint8) for i in range(batch_size)]
	frame = batch_frame(frames,scale,n_grid_shape,px_frame_shape,px_margin)
	fourcc = VideoWriter_fourcc(*'MP42')
	video = VideoWriter(filepath, fourcc, float(fps), (frame.shape[1], frame.shape[0]))	

	# Progress Bar 
	print("Video capturing...")
	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(data_loader), widgets = widgets).start()
	counter = 0

	# Video Capturing
	for _time, pos, labels in data_loader:
		time_splitted = []
		for i in range(batch_size):
			time_splitted.append(split_time(_time[i].numpy(),incr))
		
		t0_list = list(zip(*time_splitted))
		t1_list = t0_list[1:]
		for t0,t1 in zip(t0_list,t1_list):
			frames = []
			for j,(t_0,t_1) in enumerate(zip(t0,t1)):
				frames.append(pos_to_frame(pos[j,t_0:t_1],scale,gesture_mapping[str(int(labels[j]))]))
			frame = batch_frame(frames,scale,n_grid_shape,px_frame_shape,px_margin)
			video.write(frame)
		counter+=1
		bar.update(counter)

	# Wrap up
	video.release()
	bar.finish()
	toc = time.perf_counter()
	print(f"\n{len(data_loader.dataset)} event records in {len(data_loader)} batches are recorded as video in {toc-tic:0.4f} seconds in {filepath}!")
