'''
Unittest for dvs_dataloader.py file

author: ugurc
201201
'''

import sys
sys.path.insert(0,'..')
import dvs_dataloader as load

from torch.utils.data import DataLoader

import unittest

class TestDVSDataloader(unittest.TestCase):
	def test_download_check(self):
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir)

	def test_len(self):
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir,is_train_set=True)
		print('Train Set length: ', dataset.__len__())
		dataset = load.DVSGesture(root_dir,is_train_set=False)
		print('Test Set length:', dataset.__len__())

	def test_getitem(self):
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir)
		print(dataset.__getitem__(0))

	def test_loader(self):
		root_dir = '/home/ugurc/drive/data/DvsGesture'
		dataset = load.DVSGesture(root_dir)

		test_loader = DataLoader(dataset=dataset,
		                         batch_size=1,
		                         shuffle=True,
		                         collate_fn=load.collate_fn)

		for epoch in range(1):
		  for i, (time, labels) in enumerate(test_loader):
		    # print(epoch, i, "\n\npos :", time, "\n\npos :", pos, "\n\nlabels :", labels)
		    print(epoch, i, "\n\ntime :", time, "\n\nlabels :", labels)
		    break

if __name__=='__main__':
	# env.clear_env()
	# env.test_dvs_dataloader()
	unittest.main()