'''
Utility functions

201223
author : Ugurcan Cakal
'''
import cv2
import numpy as np
import torch

from cv2 import VideoWriter, VideoWriter_fourcc

def pos_to_frame(pos, px_frame_shape=(128,128,3), px_val=255):
	'''
	Creates an image out of the input position sequence.
	Use the positions given to create pixel values.
	Positive change(pol=1) in limunosity is represented by green and
	Negative change(pol=0) in limunosity is represented by blue.

		Arguments:

			pos(np.ndarray of dtype np.uint8):
				sequence of [xpos, ypos, pol]'s to be converted into an image

			px_frame_shape(int,int,int):
				IMPORTANT : (height,width,channel)
				frame shape in pixels. no control, choose carefully. 
				default : 128,128,3

			px_val(int):
				pixel value to be set for determined positions 
				should be between 0 and 255

		Returns:
			frame(3D np.ndarray):
				BGR image of the responsible position sequence
	'''
	if not isinstance(pos, np.ndarray):
		pos = np.asarray(pos,dtype=np.int32)
	if px_val<0 or px_val>255:
		print(f"FAIL: px_val : '{px_val}', should have been between 0 and 255!")
		return

	# Frame Generation
	frame=np.zeros(px_frame_shape,dtype=np.uint8)

	# Indexes
	pol_one = pos[np.asarray(pos[:,2],dtype=bool)] # Green
	frame[pol_one[:,1],pol_one[:,0],1] = px_val

	pol_zero = pos[np.asarray(1-pos[:,2],dtype=bool)] # Blue
	frame[pol_zero[:,1],pol_zero[:,0],0] = px_val
	
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
		_time = _time.cpu().detach().numpy()

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

def text_on_frame(frame,text,scale=1,pos=(0,10),font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.28, color=(0,255,255), thickness=None):
	'''
	Print text on the frame

		Arguments:
			
			frame(np.ndarray of dtype np.uint8):
				image of interest
			
			text(str):
				text to be put on the image

			pos(int,int):
				IMPORTANT : (x,y)
				Coordinates of the bottom left corner of the text 
				Default : (0,10)
			
			font(cv2 font):
				font of the text
				Default: cv2.FONT_HERSHEY_SIMPLEX
			
			font_scale(float):
				Default : 0.28

			color(uint8,uint8,uint8):
				BGR color of the text
				Default : (0,255,255)

			thickness(int):
				font thickness 
				Default : 1

		Returns:
			frame(np.ndarray of dtype np.uint8):
				image with text on it

	'''
	if not isinstance(text, str):
		text = str(text)

	pos = tuple([dim*scale for dim in pos])
	font_scale *=scale 
	if not thickness:
		thickness = 1+(scale//2)

	cv2.putText(frame, text, pos, font, font_scale, color, thickness)
	return frame

def scale_frame(frame, scale):
	'''
	Print text on the frame

		Arguments:
			
			frame(np.ndarray of dtype np.uint8):
				image of interest

		Returns:
			frame(np.ndarray of dtype np.uint8):
				image with text on it

	'''
	dsize = tuple([int(round(dim*scale)) for dim in frame.shape[0:2]])
	frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_NEAREST)
	return frame

def save_video(frame_seq,filepath='test.avi',px_frame_shape=(128,128,3),fps=24):
	'''
	Save the given frame sequence in the form of a video

		Arguments:

			px_frame_shape(int,int,int):
				IMPORTANT : (height,width,channel)
				frame shape in pixels. no control, choose carefully. 
				default : 128,128,3
	'''
	fourcc = VideoWriter_fourcc(*'MP42')
	frame_shape = (px_frame_shape[0], px_frame_shape[1], 3)

	base_frame = np.zeros(frame_shape,dtype=np.uint8)
	video = VideoWriter(filepath, fourcc, float(fps), (base_frame.shape[1], base_frame.shape[0]))	

	for frame in frame_seq:
		base_frame[:,:,0:px_frame_shape[2]] = frame
		video.write(base_frame)
	video.release()