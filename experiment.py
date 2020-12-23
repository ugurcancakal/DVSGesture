'''
Experiment Pipeline

author: @ugurc
201217
'''
import os
import time
import torch
import logging
import datetime

import numpy as np
import pandas as pd
import torch.nn as nn
import visualisation as vis
import dvs_dataloader as load

from torch.autograd import Variable
from torch.utils.data import DataLoader

class Experiment():
	'''
	Pipeline:

		1) Setup the environment
			- Directories
			- Logging
		2) Log the experiment info
		3) Create datasets and dataloaders
		4) Construct the model
		5) Evaluate the initial state
			- Loss
			- Metric
		6) Fit the model to the training data
		7) Test the model with random samples
		8) Save the model

	Outputs:

		model.pth

		Records:
			name.log
			loss.csv
			metric.csv
			results.csv

		Figures:
			loss.png
			metric.png
			event_rec.mp4
		
	'''
	def __init__(self, name, root_dir, data_params, model, model_params, opt_fn, lr, loss_fn, metric):
		'''
			Arguments:

				root_dir(str):
					root directory to hold the experiment records

				name(str):
					name of the experiment
		'''
		
		self.name = name
		self.root_dir = root_dir
		self.init_paths(root_dir, name)

		self.init_logging()		
		self.init_data(**data_params)
		self.init_model(model, model_params, loss_fn, metric, opt_fn, lr)

	def init_paths(self, root_dir, name):
		'''
		Create subdirectories and construct paths for the output files

			Arguments:

				root_dir(str):
					root directory to hold the experiment records

				name(str):
					name of the experiment
		'''

		# Subdirectories
		exp_dir = os.path.join(root_dir,name)
		if not os.path.exists(exp_dir):
			os.makedirs(exp_dir)
		self.exp_dir = exp_dir

		record_dir = os.path.join(exp_dir,'records')
		if not os.path.exists(record_dir):
			os.makedirs(record_dir)

		figure_dir = os.path.join(exp_dir,'figures')
		if not os.path.exists(figure_dir):
			os.makedirs(figure_dir)

		# Model
		self.model_path = os.path.join(exp_dir,name+'_model.pth')

		# Records
		self.loss_csv_path = os.path.join(record_dir,name+'_loss.csv')
		self.metric_csv_path = os.path.join(record_dir,name+'_metrics.csv')
		self.log_path = os.path.join(exp_dir,name+'.log')

		# Figures
		self.loss_png_path = os.path.join(figure_dir,name+'_loss.png')
		self.metric_png_path = os.path.join(figure_dir,name+'_metrics.png')

	def init_logging(self):
		'''
		Initialize logging and print experiment info
		'''
		logging.basicConfig(format='%(asctime)s - %(message)s', 
												level=logging.INFO, 
												datefmt='%d-%b-%y %H:%M:%S', 
												handlers=[logging.FileHandler(self.log_path),
																	logging.StreamHandler()])

		_info =  f'Experiment "{self.name}" Started!\n'
		_info += f'\tStorage location : {self.exp_dir}\n'
		logging.info(_info)

	def init_data(self, data_dir, train_batch, train_transform, test_batch, test_transform):
		'''
		# Validation set may be required!
		Initilize the dataset and dataloaders

			Argument:
				data_dir(str):
					directory of the raw DVSGesture dataset
		'''
		# Train set
		self.train_set = load.DVSGesture(data_dir,
																		 is_train_set=True,
																		 transform=train_transform)

		self.train_dl = DataLoader(dataset=self.train_set,
															 batch_size=train_batch,
															 shuffle=True,
															 collate_fn=load.collate_fn)

		# Test set
		self.test_set = load.DVSGesture(data_dir,	
																		is_train_set=False,	
																		transform=test_transform)

		self.test_dl = DataLoader(dataset=self.test_set,
															batch_size=test_batch,
															shuffle=True,
															collate_fn=load.collate_fn)

		_info =  f'Dataset and dataloader objects created!\n'
		_info += f'\tTraining Set:\n'
		_info += f'\t\tLength : {len(self.train_set)}\n'
		_info += f'\t\tBatch Size : {train_batch}\n'
		_info += f'\t\tTransformations : {train_transform.__name__}\n'
		_info += f'\tTest Set:\n'
		_info += f'\t\tLength : {len(self.test_set)}\n'
		_info += f'\t\tBatch Size : {test_batch}\n'
		_info += f'\t\tTransformations : {test_transform.__name__}\n'
		logging.info(_info)

	def init_model(self, model, model_params, loss_fn, metric, opt_fn, lr):
		'''
		Initialize the model under experiment
		'''
		model = model(**model_params)
		self.model = model_parallel(model)
		self.loss_fn = loss_fn
		self.metric = metric
		self.opt = opt_fn(self.model.parameters(), lr =lr)  

		# Model description
		model_str= f'Model created!'
		model_str+= f'\n\tModel :'
		model_str+= '\t\t'.join(('\n'+model.__str__().lstrip()).splitlines(True))
		model_str+= f'\n\tOptimization Function :'
		model_str+= '\t\t'.join(('\n'+self.opt.__str__().lstrip()).splitlines(True))
		model_str+= f'\n\tLoss Funtion : {loss_fn}\n'
		model_str+= f'\tEvaluation Metric : {metric.__name__}\n'
		logging.info(model_str)

		# Initial Evaluation
		model.eval()
		self.init_train_loss, _, _ = evaluate(model, loss_fn, self.train_dl, metric = metric)
		logging.info('Initial train loss: {:.4f}'.format(self.init_train_loss))

		self.init_val_loss, _, self.init_val_metric = evaluate(model, loss_fn, self.test_dl, metric = metric)
		logging.info('Initial validation loss: {:.4f}'.format(self.init_val_loss))
		logging.info('Initial validation {}: {:.4f}'.format(metric.__name__,self.init_val_metric))

	def run_for(self, epochs, timelimit, repeat_test=0):
		'''
		'''
		timeunit = timelimit[1]
		timelimit = time_second(*timelimit)
		start = time.perf_counter()
		elapsed = 0        
		epoch_average = 0
		train_losses, val_losses, val_metrics = [self.init_train_loss], [self.init_val_loss], [self.init_val_metric]

		logging.info(f'Model will be trained for {epochs} epochs unless the timelimit {timelimit} {timeunit}s passed!')
		for epoch in range(epochs):
			if elapsed < timelimit - epoch_average:          
				# Do the operation
				tic = time.perf_counter()
				train_loss, val_loss, val_metric = fit_one_epoch(self.model, self.train_dl, self.test_dl, self.loss_fn, self.opt, self.metric)
				
				# Record the loss & metric
				train_losses.append(train_loss)
				val_losses.append(val_loss)
				val_metrics.append(val_metric)

				# Print the progress
				if self.metric:
					logging.info('Epoch: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}, {}:{:.4f}'\
								.format(epoch+1, epochs, train_loss, val_loss, \
												self.metric.__name__, val_metric))
				else:
					logging.info('Epoch: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}'\
								.format(epoch+1, epochs, train_loss, val_loss))

				# OUTRO
				toc = time.perf_counter()
				epoch_time = toc-tic
				epoch_average = epoch_time if epoch_average == 0 else (epoch_average*(epoch-1)+epoch_time)/epoch
				elapsed = time.perf_counter() - start #update the time elapsed

			else:
				logging.info(f'Time limit reached : {timelimit} {timeunit}s')
				break

		duration = time.perf_counter() - start
		self.test(repeat_test)
		self.wrap_up(self.model, train_losses, val_losses, val_metrics, duration)
		None

	def test(self, repeat):
		'''
		'''
		for i,(time,pos,label) in enumerate(self.test_dl, 1):
			
			actuals = label.cpu().detach().numpy()
			# ATTENTION
			preds = self.model(create_variable(pos.float()))
			probs = nn.Softmax(dim=1)(preds).cpu().detach().numpy()
			gesture_idx = np.argmax(probs, axis=1)

			labels_actual = [self.test_set.gesture_mapping[str(id_1)] for id_1 in actuals]
			labels_predicted = []

			for id_0 in gesture_idx:
				# ATTENTION
				labels_predicted.append(self.test_set.gesture_mapping[str(id_0+1)])

			comparison = f'Test {i}\n'
			comparison += f'\tActual : \t {labels_actual}\n'
			comparison += f'\tPredicted : \t {labels_predicted}\n'
			logging.info(comparison)
			if i >= repeat:
				break
		
	def wrap_up(self, model, train_loss, val_loss, val_metric, duration):
		# Save the state, create all the necessary items
		loss_title = "Experiment '" + self.name + "' Train & Validation Losses"
		metric_title = "Experiment '" + self.name + "' Validation " + self.metric.__name__
		
		record_losses(train_loss, val_loss, loss_title, self.loss_csv_path, self.loss_png_path)
		record_metric(self.metric, val_metric, metric_title, self.metric_csv_path, self.metric_png_path)
		
		torch.save(model.state_dict(), self.model_path)
		logging.info(f'Model State saved in {self.model_path}')
		time_passed = datetime.timedelta(seconds=round(duration))
		# test_model()

		_info= f"The experiment duration (%H:%M:%S) : {str(time_passed)}\n"
		_info+= "\nTHE END\n"
		logging.info(_info)



def fit_one_epoch(model, train_dl, test_dl, loss_fn, opt, metric):
	# Are you training or evalutating 
	model.train() 
	for time, pos, labels in train_dl:
		# ATTENTION
		time, pos, labels = [create_variable(x) for x in [time,pos.float(),labels.long()]]
		train_loss, _, _ = loss_batch(model, loss_fn, pos, labels, opt, metric=None)

	# Evaluate 
	model.eval()
	val_loss, total, val_metric = evaluate(model, loss_fn, test_dl, metric)
			
	return train_loss, val_loss, val_metric

def evaluate(model, loss_fn, valid_dl, metric=None):
	'''
	Evaluate the performance of the model on the validation dataset. 
	Call the loss_batch() function without an optimization funciton.
	In this way, model parameters will not be affected.

		Arguments:
			
			model(nn.Module object):
				The model under test

			loss_fn(function): 
				Loss function to calculate the loss

			valid_dl(pytorch.dataloader): 
				Validation dataloader feeding validation dataset

			metric(function):
				Evaluation metric. Mostly useful in classification problems.
				For regression problems, loss is enough to evaluate
				Default : None

		Returns:

			avg_loss(float):
				Average loss obtained on the validation dataset
			
			_size(float):
				Total size of the validation dataset

			avg_metric(float or list of float):
				Average of the metric on the validation dataset in the
				case that a metric or a list of metrics is provided

	'''
	avg_metric = None

	# ATTENTION
	with torch.no_grad():
		results = [loss_batch(model, loss_fn, create_variable(pos.float()), create_variable(labels.long()), metric = metric) \
							 for time, pos, labels in valid_dl]	 
		losses, _len, metrics = zip(*results)

		_size = np.sum(_len)
		avg_loss = np.sum(np.multiply(losses, _len)) / _size
		
		if metric:
			avg_metric = np.sum(np.multiply(metrics, _len)) / _size

	return avg_loss, _size, avg_metric

def loss_batch(model, loss_fn, xb, yb, opt = None, metric = None):
	'''
	Calculate the loss and metric for a batch of data.
	The flow of operation is like:
	1 - Generate predictions 
	2 - Calculate gradient
	3 - Update parameters
	4 - Reset gradients
	5 - Compute the metric
		
		Arguments:

			model(nn.Module object):
				The model under operation
			
			loss_fn(function): 
				Loss function to calculate the loss

			xb(torch.Tensor):
				batch of input data 

			yb(torch.Tensor):
				batch of target output data(labels)

			opt(function):
				Optimization function in the case that weights are going to
				be updated using gradient descent or any other method.
				Default : None 

			metric(function):
				Evaluation metric. Mostly useful in classification problems.
				For regression problems, loss is enough to evaluate	
				Default: None

		Returns:

			loss.item()(float):
				Computed loss casted from torch data to a python number

			len(xb):
				batch size of the data provided

			metric_result(float):
				Computed metric result casted from torch data to a python number
	
	'''
	metric_result = None
	preds = model(xb)
	# ATTENTION
	loss = loss_fn(preds, yb-1)
	# print(yb)
	# print(preds)
	# print(loss)

	if opt:
		loss.backward()
		opt.step()
		opt.zero_grad()
	
	if metric:
		metric_result = metric(preds, yb)

	if metric_result:
		metric_result = metric_result
	return loss.item(), len(xb), metric_result

def record_metric(metric, val_metric, title, csv_path, png_path):
	'''
	'''
	metric = dict(zip([metric.__name__],[val_metric]))

	dataframe = pd.DataFrame(metric)
	dataframe.index.name = "Epoch"
	dataframe.to_csv(csv_path)

	vis.plt_config(transparent=False)
	vis.plot_metric(dataframe, title=title, filepath=png_path)

def record_losses(train_loss, test_loss, title, csv_path, png_path):
	'''
	Record losses in .csv file and plot figure
	'''
	keys = ['train_loss', 'test_loss']
	losses = dict(zip(keys,[train_loss, test_loss]))

	dataframe = pd.DataFrame(losses)
	dataframe.index.name = "Epoch"
	dataframe.to_csv(csv_path)

	vis.plt_config(transparent=False)
	vis.plot_loss(dataframe, title=title, filepath=png_path)

def model_parallel(model):
	'''
	Helper function to parallelize the model in the case 
	there is at least one GPU(or more).

		Arguments:
			
			model(torch.model):
				network model of interest

			log(bool):
				print the model if True

		Returns:
			model(torch.model):
				parallelized model(if available)
	'''
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
		model = nn.DataParallel(model)

	if torch.cuda.is_available():
		print("Let's use GPU!")
		model.cuda()

	return model

def create_variable(tensor):
	'''
	Helper function to create a variable from a tensor 
	considering CUDA availability. 

		Arguments:
			tensor(torch.Tensor):
				pytorch tensor to be used

		Returns:
			tensor(torch.Variable):
				pytorch variable(CUDA if available)
	'''
	if torch.cuda.is_available():
		return Variable(tensor.cuda())
	else:
		return Variable(tensor)

def time_second(duration, timeunit):
	time_sec = 0

	if timeunit == 'day':
		time_sec = duration*86400
	if timeunit == 'hour':
		time_sec = duration*3600
	elif timeunit == 'minute':
		time_sec = duration*60
	elif timeunit == 'second':
		time_sec = duration

	return time_sec


