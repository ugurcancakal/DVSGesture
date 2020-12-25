'''
Data transformations 

201221
author : @ugurc
'''
import cv2
import utils

import numpy as np
import dvs_dataset as dvs

# template
class template():
	'''
	'''
	def __init__(self,argument):
		'''
		Object Initialization
			Arguments:
		'''
	def __str__(self):
		'''
		Convert the object to a string
		'''
		info = "template(\n"
		info += f"\targument = {argument})\n"
		return info

	def __call__(self,_time,pos,label):
		'''
		Object call transfroms the input data
			Arguments:
				_time(1D np.ndarray of type np.uint32):
					time line of the event sequence consisting of elements representing
						the times that the event happened in microsecond resolution 

				pos(2D np.ndarray of type np.uint8):
					2D positions of events in 2D coordinate space [xpos, ypos, pol]
					the same indexing with the time sequence

				label(np.uint8):
					Label of the event happened. The same indexing with _time and pos

				Return:
					_time(1D np.ndarray of type np.uint32)
					pos(2D np.ndarray of type np.uint8)
					label(np.uint8)
		'''
		return _time,pos,label

class average_stride():
	'''
	Average the time and positions by defined window size
	to reduce the size of the event sequence.

	Throw out the last instances if the sequence length 
	is not a factor window size.

	new seq_length = seq_length/window
	'''
	def __init__(self, window):
		'''
		Object Initialization
			Arguments:
				window(int): averaging window
		'''
		self.window = window

	def __str__(self):
		'''
		Convert the object to a string
		'''
		info = "average_stride(\n"
		info += f"\twindow = {self.window})\n"
		return info

	def __call__(self,_time,pos,label):
		'''
		Object call transfroms the input data

			Arguments:
				_time(1D np.ndarray of type np.uint32):
					time line of the event sequence consisting of elements representing
						the times that the event happened in microsecond resolution 

				pos(2D np.ndarray of type np.uint8):
					2D positions of events in 2D coordinate space [xpos, ypos, pol]
					the same indexing with the time sequence

				label(np.uint8):
					Label of the event happened. The same indexing with _time and pos

			Return:
				Pos and time arrays will be affected

					_time(1D np.ndarray of type np.uint32)
					pos(2D np.ndarray of type np.uint16)
					label(np.uint8)
		'''
		cut = (_time.shape[0]//self.window)*self.window
		_time = _time[:cut]
		pos = pos[:cut]

		# Create a new dimension(including window size elements) for averaging 
		t_shape = (_time.shape[0]//self.window, self.window)
		p_shape = (pos.shape[0]//self.window, self.window, pos.shape[1])
		time_shaped = _time.reshape(t_shape)
		pos_shaped = pos.reshape(p_shape)

		# Mean operation on the new dimension(reduced)
		_time = np.asarray(_time.reshape(t_shape).mean(axis=1).round(),dtype=np.uint32)
		pos = np.asarray(pos_shaped.mean(axis=1).round(),dtype=np.uint8)
		return _time,pos,label

class seq_to_video():
	'''
	Represent event sequences in the form of a time sequence of images
	The sampling rate is determined by the fps(frame per second) parameter.
	'''
	def __init__(self, fps=24, px_frame_shape=(128,128,3), binary=False, frame_limit=None):
		'''
		Object Initialization
			Arguments:

				fps(int):
					frames per second
					The sampling rate is determinded by int(1e6)//fps
					1e6 is because the time measurement is in microseconds 
					When we choose a sampling rate int(1e6)//fps, it means that
					a frame will represent all the events between happended between
					t0 and t0+int(1e6)//fps
					Default : 24 

				px_frame_shape(int,int,int):
					IMPORTANT : (height,width,channel)
					frame shape in pixels. no control, choose carefully. 
					Default : (128,128,3)

				binary(bool):
					Determines whether or not the pixel values be 0 and 1 
					or 0 and 255. 0&1 is used when True. Keep it False
					using for cv2 video representation and True for network training
					Default : False 

				frame_limit(int):
					The maximum number of frames allowed in the return
					Default : None
		'''
		self.fps = fps
		self.limit = frame_limit
		self.incr = int(1e6)//fps
		self.px_frame_shape = px_frame_shape
		self.px_val = 1 if binary else 255

	def __str__(self):
		''''
		Convert the object to a string
		'''
		info = "seq_to_video(\n"
		info += f"\tfps = {self.fps},\n"
		info += f"\tpx_frame_shape = {self.px_frame_shape},\n"
		info += f"\tpx_val = {self.px_val},\n"
		info += f"\tframe_limit = {self.limit})\n"
		return info
		
	def __call__(self,_time,pos,label):
		'''
		Object call transfroms the input data

			Arguments:
				_time(1D np.ndarray of type np.uint32):
					time line of the event sequence consisting of elements representing
						the times that the event happened in microsecond resolution 

				pos(2D np.ndarray of type np.uint8):
					2D positions of events in 2D coordinate space [xpos, ypos, pol]
					the same indexing with the time sequence

				label(np.uint8):
					Label of the event happened. The same indexing with _time and pos

			Return:
				video, time sequence of images, created using _time and pos

					video(3D np.ndarray of type np.uint8):
						time sequence of frames consisting of pixels representing
						multiple events in one grid 

					label(np.uint8)
		'''
		t0 = utils.split_time(_time,self.incr)
		t1 = t0[1:]
		video = []
		limit = self.limit if self.limit else len(t0)
		for i,(t_0,t_1) in enumerate(zip(t0,t1)):
			if(i<limit):
				frame = utils.pos_to_frame(pos[t_0:t_1], px_frame_shape=self.px_frame_shape, px_val=self.px_val)
				video.append(frame)
			else:
				break

		return np.asarray(video),label

class scale_video():
	'''
	Scale the x and y positions in all frames by factor amount
	New frame shape will be stored in the object
	'''
	def __init__(self,factor,px_frame_shape=(128,128,3)):
		'''
		Object Initialization
			Arguments:

				factor(int):
					scaling factor

				px_frame_shape(int,int,int):
					IMPORTANT : (height,width,channel)
					frame shape in pixels. no control, choose carefully. 
					Default : (128,128,3)
		'''
		self.factor=factor
		self.px_frame_shape = (int(round(px_frame_shape[0]*factor)), int(round(px_frame_shape[1]*factor)), px_frame_shape[2])

	def __str__(self):
		'''
		Convert the object to a string
		'''
		info = "scale_video(\n"
		info += f"\tfactor = {self.factor},\n"
		info += f"\tpx_frame_shape = {self.px_frame_shape})\n"
		
		return info

	def __call__(self,video,label):
		'''
		Object call transfroms the input data
			Arguments:
				video(3D np.ndarray of type np.uint8):
					time sequence of frames consisting of pixels representing
					multiple events in one grid 

				label(np.uint8):
					Label of the event happened. The same indexing with _time and pos

				Return:
					Scaled version of the video width and height will be affected.
					width, height = width*factor, height*factor

					video(3D np.ndarray of type np.uint8)
					label(np.uint8)
		'''
		scaled = []
		for frame in video:
			dsize = tuple([int(round(dim*self.factor)) for dim in frame.shape[0:2]])
			frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_NEAREST)
			scaled.append(frame)

		return np.asarray(scaled),label

class combine():
	'''
	Combination of any number of transformations.
	The ordering is important because each transformation
	will use the preceeding transformations output
	'''
	def __init__(self, *args):
		'''
		Initialization of the combine object
			
			Arguments:
				
				*args(transformations):
					transformations to be combined
					e.g. combine(seq_to_video, scale_video)
					stored as a list
		'''
		self.transforms = args

	def __str__(self):
		'''
		Convert the object to a string
		'''
		info = "COMBINED:"
		for trans in self.transforms:
			info += '\t'.join(('\n'+trans.__str__().lstrip()).splitlines(True))
		return info

	def __call__(self,_time,pos,label):
		'''
		Apply transformations to the dvs data one by one.

			Arguments:
				_time(1D np.ndarray of type np.uint32):
					time line of the event sequence consisting of elements representing
						the times that the event happened in microsecond resolution 

				pos(2D np.ndarray of type np.uint8):
					2D positions of events in 2D coordinate space [xpos, ypos, pol]
					the same indexing with the time sequence

				label(np.uint8):
					Label of the event happened. The same indexing with _time and pos

			Return:
				Depended on the last transformation
		'''
		result = (_time,pos,label)

		for transform in self.transforms:
			result = transform(*result)
		return result

# class seq_to_volume():
# 	'''
# 	x,z in volume x,y in pixels
# 	y in volume is t
# 	# NOT COMPLETE
# 	# LOOK AT SLICING
# 	'''

# 	def __init__(self, fps=24,frame_limit=100):
# 		'''
# 		Object Initialization
# 			Arguments:
# 		'''
# 		self.fps = fps
# 		self.limit = frame_limit
# 		self.incr = int(1e6)//fps

# 	def __str__(self):
# 		'''
# 		Convert the object to a string
# 		'''
# 		info = "seq_to_volume(\n"
# 		info += f"\tfps = {self.fps},\n"
# 		info += f"\tframe_limit = {self.limit})\n"
# 		return info

# 	def __call__(self,_time,pos,label):
# 		'''
# 		Object call transfroms the input data
# 			Arguments:
# 				_time(1D np.ndarray of type np.uint32):
# 					time line of the event sequence consisting of elements representing
# 						the times that the event happened in microsecond resolution 

# 				pos(2D np.ndarray of type np.uint8):
# 					2D positions of events in 2D coordinate space [xpos, ypos, pol]
# 					the same indexing with the time sequence

# 				label(np.uint8):
# 					Label of the event happened. The same indexing with _time and pos

# 				Return:
# 					_time(1D np.ndarray of type np.uint32)
# 					pos(2D np.ndarray of type np.uint8)
# 					label(np.uint8)
# 		'''
# 		t0 = utils.split_time(_time,self.incr)
# 		t1 = t0[1:]
# 		limit = self.limit if self.limit else len(t0)
# 		time1 = []
# 		time0 = []
# 		for i,(t_0,t_1) in enumerate(zip(t0,t1)):
# 			if(i<limit):
# 				time1_length = np.asarray(pos[t_0:t_1,2],dtype=bool).sum()
# 				time0_length = np.asarray(1-pos[t_0:t_1,2],dtype=bool).sum()
# 				time1  += [i]*time1_length
# 				time0  += [i]*time0_length
# 			else:
# 				break
# 		pos = pos[:t0[self.limit]]
# 		# pol_one = pos[np.asarray(pos[:,2],dtype=bool)] # Green
# 		# pol_zero = pos[np.asarray(1-pos[:,2],dtype=bool)] # Blue

# 		# x = pol_one[:,0][:t0[self.limit]]
# 		# y = np.asarray(time_course, dtype = np.uint8)
# 		# z = pol_one[:,1][:t0[self.limit]]

# 		# x2 = pol_zero[:,0][:t0[self.limit]]
# 		# y2 = np.asarray(time_course, dtype = np.uint8)
# 		# z2 = pol_zero[:,1][:t0[self.limit]]

# 		x = pos[:,0]

# 		z = 128-pos[:,1]

# 		x1 = x[np.asarray(pos[:,2],dtype=bool)]
# 		x0 = x[np.asarray(1-pos[:,2],dtype=bool)]

# 		z1 = z[np.asarray(pos[:,2],dtype=bool)]
# 		z0 = z[np.asarray(1-pos[:,2],dtype=bool)]

# 		return x0,time0,z0,x1,time1,z1,label