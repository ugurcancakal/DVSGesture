'''
Test template
author: @ugurc
201224
'''

import sys
sys.path.insert(0,'..')

import unittest
import environment as env

class Template(unittest.TestCase):

	def test1(self):
		print('Test passed!\n')

if __name__=='__main__':
	env.clear_env()
	env.template()
	unittest.main()