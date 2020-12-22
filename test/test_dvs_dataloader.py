'''
Unittest for dvs_dataloader.py file

The functions are tested using visual inspection with the 
aid of images and videos, and also with simple small hardcoded arrays.

The empty cases and big chunk of data cases remains untested. 

Some test requires pre environment creation and it can 
be investigated in in test_env.py file.

author: ugurc
201201
'''

import sys
sys.path.insert(0,'..')
import dvs_dataloader as load
import test_env as env

import torch
from torch.utils.data import DataLoader

import unittest
import cv2
import numpy as np
import math

import os

class TestDVSDataloader(unittest.TestCase):
	def test1_download_check(self):
		'''
		Check if dataset fails in case the root file does not exist
		'''
		root_dir = os.path.join(os.getcwd(), 'thisFileDoesNotExists')
		dataset = load.DVSGesture(root_dir,'thisFileDoesNotExists.hdf5')
		print('!!!!FAKE FAIL: Manual download check test\n')

	def test2_valid_check_trainset(self):
		'''
		Check if the dataset fails in the case 'trials_to_train.txt' file is missing
		'''
		root_dir = os.path.join(os.getcwd(), 'test_files', 'trainMissing')
		try:
			dataset = load.DVSGesture(root_dir, 'trainMissing.hdf5')
		except AssertionError as e:
			print(repr(e))
			print("!!!!FAKE FAIL: Validity check trainset file missing test passed!\n")

	def test3_valid_check_testset(self):
		'''
		Check if the dataset fails in the case 'trials_to_test.txt' file is missing
		'''
		root_dir = os.path.join(os.getcwd(), 'test_files', 'testMissing')
		try:
			dataset = load.DVSGesture(root_dir, 'testMissing.hdf5')
		except AssertionError as e:
			print(repr(e))
			print("!!!!FAKE FAIL: Validity check testset file missing test passed!\n")

	def test4_valid_check_gesture(self):
		'''
		Check if the dataset fails in the case gesture_mapping.csv' file is missing
		'''
		root_dir = os.path.join(os.getcwd(), 'test_files', 'gestureMissing')
		try:
			dataset = load.DVSGesture(root_dir, 'gestureMissing.hdf5')
		except AssertionError as e:
			print(repr(e))
			print("!!!!FAKE FAIL: Validity check gesture mapping file missing test passed!\n")

	def test5_valid_check_random_file_missing(self):
		'''
		Check if prerequisite files exist but one of the files(randomly selected)
		referenced in 'trials_to_train.txt' or in 'trials_to_test.txt' is missing
		'''
		root_dir = os.path.join(os.getcwd(), 'test_files', 'randomMissing')
		try:
			dataset = load.DVSGesture(root_dir, 'randomMissing.hdf5')
		except AssertionError as e:
			print(repr(e))
			print("!!!!FAKE FAIL: Validity check random file missing test passed!\n")

	def test6_length(self):
		'''
		Check if train and test set lengths are exactly the same as expected
		Train set has 1176 instances and the test set has 288 instances
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir,is_train_set=True)
		self.assertEqual(dataset.__len__(), 1176)

		dataset = load.DVSGesture(root_dir,is_train_set=False)
		self.assertEqual(dataset.__len__(), 288)
		print('Dataset length test passed(1176/288)!\n')

	def test7_getitem_shape(self):
		'''
		Check if __getitem__ returns the data in the correct form.
		It should return a tuple with 3 elements : 
			time(1D np.ndarray)
			pos(2D np.ndarray consisting of [xpos,ypos,pol] elements) 
			label(np.uint8)
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir)
		time, pos, label = dataset[0] # __getitem__

		self.assertTrue(isinstance(time,np.ndarray))
		self.assertTrue(isinstance(pos,np.ndarray))
		self.assertTrue(isinstance(label,np.uint8))

		self.assertEqual(len(time.shape),1)
		self.assertEqual(len(pos.shape),2)
		self.assertEqual(pos.shape[1],3)

		print('Dataset getitem shape test passed!\n')

	def test8_padding(self):
		'''
		Check if pad_sequences(..) works properly.
		It should take a list of arrays and create a +1 dimensional
		tensor with 0-padded forms of the input arrays
		'''
		array1 = np.array([[1,2,3],
											 [5,6,7],
											 [12,22,34],
											 [2,4,5]])

		array2 = np.array([[-1,-2,-3],
											 [-5,-6,-7]])

		array3 = np.array([[-174,-21,-34],
											 [12,-634,72],
											 [23,234,5]])

		seqs = [array1, array2, array3]
		seq_lengths = np.asarray([len(arr) for arr in seqs])
		seq_tensor = load.pad_sequences(seqs,seq_lengths)

		_seq_tensor = torch.tensor([[[   1,    2,    3],
												         [   5,    6,    7],
												         [  12,   22,   34],
												         [   2,    4,    5]],

												        [[  -1,   -2,   -3],
												         [  -5,   -6,   -7],
												         [   0,    0,    0],
												         [   0,    0,    0]],

												        [[-174,  -21,  -34],
												         [  12, -634,   72],
												         [  23,  234,    5],
												         [   0,    0,    0]]])

		self.assertTrue((seq_tensor.numpy()==_seq_tensor.numpy()).all())
		print('Padding test passed!\n')

	def test9_loader(self):
		'''
		Checks if collate_fn(..) work properly.
		It requires manual investigation. An artificial data could
		have been created to check automatically but it takes time.
		Collate function helps the dataloader to convert given data
		to torch tensors. In this case, it also does zero padding.
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		batch = 4
		dataset = load.DVSGesture(root_dir)

		test_loader = DataLoader(dataset=dataset,
														 batch_size=batch,
														 shuffle=True,
														 collate_fn=load.collate_fn)

		for epoch in range(1):
			for i, (time, pos, labels) in enumerate(test_loader):
				print(epoch, i, "\n\ntime :", time, "\n\npos :", pos, "\n\nlabels :", labels)
				break
		print('Manual data loader test!\n')

	def test10_pos_to_frame_basic(self):
		'''
		Basic pos_to_frame test using a small array
		Visual inspection makes it easier to decide. 
		Uncomment to see the resulting picture.
		'''
		pos = [[0,5,1],
					 [3,4,0],
					 [8,15,0],
					 [15,15,1]]
		frame = load.pos_to_frame(pos,2,'',(16,16))

		# # UNCOMMENT to visually inspect
		# cv2.imshow("Test", frame)
		# k = cv2.waitKey(0)

		print("Basic pos to frame test passed(Check the image visually if required)!\n")

	def test11_pos_to_frame(self):
		'''
		Check if pos_to_frame(..) function creates frames out of
		given position sequences correctly.
		Visual inspection makes it easier to decide. 
		Uncomment to see the resulting picture.
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir)

		test_loader = DataLoader(dataset=dataset,
														 batch_size=4,
														 shuffle=True,
														 collate_fn=load.collate_fn)

		for i, (time, pos, labels) in enumerate(test_loader):
			frame = load.pos_to_frame(pos[0],4,'right_arm_counter_clockwise')
			self.assertEqual(frame.shape, (512,512,3))

			# # UNCOMMENT to visually inspect
			# cv2.imshow("Test", frame)
			# k = cv2.waitKey(0)
			break
		print('Pos to frame Test passed(Check the image visually if required)!\n')

	def test12_get_area_index(self):
		'''
		Check if get_area_index(..) returns the correct min, max
		coordinate values to create a valid mask for placing 
		multiple position images in a frame
		'''
		idx = []
		batch_size = 9
		n_frame_width=math.ceil(math.sqrt(batch_size))
		
		px_frame_width=128
		px_frame_height=128
		px_margin=10
		scale=2

		_idx = [(20, 276, 20, 276), (20, 276, 296, 552), (20, 276, 572, 828), (296, 552, 20, 276), (296, 552, 296, 552), (296, 552, 572, 828), (572, 828, 20, 276), (572, 828, 296, 552), (572, 828, 572, 828)]

		for i in range(9):
			idx.append(load.get_area_index(i,n_frame_width,scale,(px_frame_width,px_frame_height),px_margin))

		self.assertEqual(idx,_idx)
		print('Area index test passed!\n')

	def test13_batch_frame(self):
		'''
		Check if batch_frame(..) places the multiple image 
		in a frame correctly. Shape is controlled automatically.
		Visual inspection makes it easier to decide. 
		Uncomment to see the resulting picture.
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir)
		batch = 18
		scale = 2

		test_loader = DataLoader(dataset=dataset,
														 batch_size=batch,
														 shuffle=True,
														 collate_fn=load.collate_fn)
		pos_batch = []
		for i, (time, pos, labels) in enumerate(test_loader):
			for i in range(batch):
				pos_batch.append(load.pos_to_frame(pos[i],scale,str(labels[i])))
			break

		frame_batch=load.batch_frame(pos_batch,scale, (3,6))
		self.assertEqual(frame_batch.shape, (848, 1676, 3))
		# # UNCOMMENT to visually inspect
		# cv2.imshow("Test", frame_batch)
		# k = cv2.waitKey(0)
		print('Batch frame Test passed(Check the image visually if required)!\n')

	def test14_split_time_simple(self):
		'''
		Checks if split_time(..) can discretize a time series properly.
		It should be able to pick the indices of the series in a way 
		to represent the increment amount in the resulting 
		array as close as possible.
		'''
		_time = [12, 15, 21, 23, 26, 27, 32]
		incr = 10
		idx = load.split_time(_time,incr)
		_idx = [0,3,6]
		# resultin array would become 
		# [12, 23, 32]
		self.assertEqual(idx,_idx)
		print('Simple time split test passed!\n')

	def test15_split_time(self):
		'''
		Shape test for the resulting discretization of split_time(..)
		The same idea here applies for video generation also.
		'''
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		fps = 24
		batch = 9
		incr = int(1e6)//fps
		scale = 2

		dataset = load.DVSGesture(root_dir)
		gesture_mapping = dataset.gesture_mapping
		test_loader = DataLoader(dataset=dataset,
														 batch_size=batch,
														 shuffle=True,
														 collate_fn=load.collate_fn)


		for i, (time, pos, labels) in enumerate(test_loader):
			time_splitted = []
			for i in range(batch):
				time_splitted.append(load.split_time(time[i].numpy(),incr))

			# transpose
			t0_list = list(zip(*time_splitted))
			t1_list = t0_list[1:]
			self.assertEqual(len(t0_list[0]),batch)

			for t0,t1 in zip(t0_list,t1_list):
				for j,(t_0,t_1) in enumerate(zip(t0,t1)):
					break
				self.assertEqual(t0,tuple([0]*batch))
				# Check if all 0
				break
			break

		print('Time split shape test has passed!\n')

	# def test16_loader_video(self):
	# 	'''
	# 	Generate a video using a dataloader's returns. 
	# 	See if all the event sequences looks natural and 
	# 	their endings are not far away each others.
	# 	Visual inspection makes it easier to decide. 
	# 	The video will be saved in current directory with name 'test.avi'
	# 	!It takes TOO MUCH TIME and MEMORY,(like 500 MB and 30 secs, not too much) 
	# 	comment if unnecessary
	# 	'''

	# 	root_dir = '/home/ugurc/drive/data/DvsGesture'
	# 	filename = 'test.avi'
	# 	batch = 18

	# 	dataset = load.DVSGesture(root_dir)
	# 	test_loader = DataLoader(dataset=dataset,
	# 													 batch_size=batch,
	# 													 shuffle=True,
	# 													 collate_fn=load.collate_fn)


	# 	load.loader_video(test_loader,filename,n_grid_shape=(3,6))


if __name__=='__main__':
	env.clear_env()
	env.dvs_dataloader_env()
	unittest.main()