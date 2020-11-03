#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:35:33 2020

@author: ugurc
"""

import unittest
import dvs_gesture2
import os

event_head1 = [1, 1, 8, 4, 0, 5, 5, 5]
event_head2 = [1, 1, 8, 4, 0, 4, 4, 4]
event_head3 = [1, 1, 8, 4, 0, 3, 3, 3]
event_seq1 = [[10,0,0,1],[25,10,20,0],[38,120,59,1],[48,5,9,0],[59,9,5,1]]
event_seq2 = [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1],[126562,45,59,1]]
event_seq3 = [[2068981,127,127,0],[2157619,0,15,1],[2178812,1,1,1]]

def create_test_files(aedat_file,csv_file):
  event_head = [event_head1, event_head2, event_head3]
  event_seq = [event_seq1, event_seq2, event_seq3]
  event_it = zip(event_head,event_seq)

  with open(aedat_file, 'ab') as f:
    create_file_header(f)
    for head, seq in event_it:
      create_event_header(f,head)
      for event in seq:
        create_polarity_event(f,event)

def create_file_header(_f):
  _version = b'#!AER-DAT3.1\r\n'
  _format = b'#Format: RAW\r\n'
  _source = b'#Source 1: Test\r\n'
  _start_time = b'#Start-Time: 2020-10-30 17:33:59 (TZ+0300)\r\n'
  _end_line = b'#!END-HEADER\r\n'

  _f.write(_version+_format+_source+_start_time+_end_line)

def create_event_header(_f,header):
  eventType = header[0].to_bytes(2,byteorder='little')
  eventSource = header[1].to_bytes(2,byteorder='little')
  eventSize = header[2].to_bytes(4,byteorder='little')
  eventTSOffset = header[3].to_bytes(4,byteorder='little')
  eventTSOverflow = header[4].to_bytes(4,byteorder='little')
  eventCapacity = header[5].to_bytes(4,byteorder='little') 
  eventNumber = header[6].to_bytes(4,byteorder='little')
  eventValid = header[7].to_bytes(4,byteorder='little')

  _f.write(eventType+eventSource+eventSize+\
    eventTSOffset+eventTSOverflow+eventCapacity+\
    eventNumber+eventValid)

def create_polarity_event(_f,seq):
  '''
  xpos[31-17]ypos[16-2]pol[1]valid[0] (0,0 upper left corner)
  little endian
  '''
  time = seq[0]
  data = 0

  data = data | seq[1]<<17
  data = data | seq[2]<<2
  data = data | seq[3]<<1
  _f.write(data.to_bytes(4,byteorder='little')+time.to_bytes(4,byteorder='little'))


class TestDVSGesture(unittest.TestCase):  
  data_dir = '/home/ugurc/drive/data/DvsGesture'
  trials_to_train = '/trials_to_train.txt'
  trials_to_test = '/trials_to_test.txt'

  aedat_file = 'aedat_for_testing.aedat'
  csv_file = 'aedat_for_testing_labels.csv'



  def test_filelist_train(self):
    '''
    Checks if filename_list function works properly.
    The filenames to be included in train list
    are hardcoded in a list. Test function checks if hardcoded
    lists and the lists read from the .txt files are exactly the same.
    '''
    train_aedat = ['user01_fluorescent.aedat',
                   'user01_fluorescent_led.aedat',
                   'user01_lab.aedat',
                   'user01_led.aedat',
                   'user01_natural.aedat',
                   'user02_fluorescent.aedat',
                   'user02_fluorescent_led.aedat',
                   'user02_lab.aedat',
                   'user02_led.aedat',
                   'user02_natural.aedat',
                   'user03_fluorescent.aedat',
                   'user03_fluorescent_led.aedat',
                   'user03_led.aedat',
                   'user03_natural.aedat',
                   'user04_fluorescent.aedat',
                   'user04_fluorescent_led.aedat',
                   'user04_led.aedat',
                   'user04_natural.aedat',
                   'user05_fluorescent.aedat',
                   'user05_fluorescent_led.aedat',
                   'user05_lab.aedat',
                   'user05_led.aedat',
                   'user05_natural.aedat',
                   'user06_fluorescent.aedat',
                   'user06_fluorescent_led.aedat',
                   'user06_lab.aedat',
                   'user06_led.aedat',
                   'user06_natural.aedat',
                   'user07_fluorescent.aedat',
                   'user07_fluorescent_led.aedat',
                   'user07_lab.aedat',
                   'user07_led.aedat',
                   'user08_fluorescent.aedat',
                   'user08_fluorescent_led.aedat',
                   'user08_lab.aedat',
                   'user08_led.aedat',
                   'user09_fluorescent.aedat',
                   'user09_fluorescent_led.aedat',
                   'user09_lab.aedat',
                   'user09_led.aedat',
                   'user09_natural.aedat',
                   'user10_fluorescent.aedat',
                   'user10_fluorescent_led.aedat',
                   'user10_lab.aedat',
                   'user10_led.aedat',
                   'user11_fluorescent.aedat',
                   'user11_fluorescent_led.aedat',
                   'user11_natural.aedat',
                   'user12_fluorescent_led.aedat',
                   'user12_led.aedat',
                   'user13_fluorescent.aedat',
                   'user13_fluorescent_led.aedat',
                   'user13_lab.aedat',
                   'user13_led.aedat',
                   'user13_natural.aedat',
                   'user14_fluorescent.aedat',
                   'user14_fluorescent_led.aedat',
                   'user14_led.aedat',
                   'user14_natural.aedat',
                   'user15_fluorescent.aedat',
                   'user15_fluorescent_led.aedat',
                   'user15_lab.aedat',
                   'user15_led.aedat',
                   'user15_natural.aedat',
                   'user16_fluorescent.aedat',
                   'user16_lab.aedat',
                   'user16_led.aedat',
                   'user16_natural.aedat',
                   'user17_fluorescent.aedat',
                   'user17_fluorescent_led.aedat',
                   'user17_lab.aedat',
                   'user17_led.aedat',
                   'user17_natural.aedat',
                   'user18_fluorescent.aedat',
                   'user18_fluorescent_led.aedat',
                   'user18_lab.aedat',
                   'user18_led.aedat',
                   'user19_fluorescent.aedat',
                   'user19_fluorescent_led.aedat',
                   'user19_lab.aedat',
                   'user19_led.aedat',
                   'user19_natural.aedat',
                   'user20_fluorescent.aedat',
                   'user20_fluorescent_led.aedat',
                   'user20_led.aedat',
                   'user21_fluorescent.aedat',
                   'user21_fluorescent_led.aedat',
                   'user21_lab.aedat',
                   'user21_natural.aedat',
                   'user22_fluorescent.aedat',
                   'user22_fluorescent_led.aedat',
                   'user22_lab.aedat',
                   'user22_led.aedat',
                   'user22_natural.aedat',
                   'user23_fluorescent.aedat',
                   'user23_fluorescent_led.aedat',
                   'user23_lab.aedat',
                   'user23_led.aedat']

    train_aedat = [self.data_dir+'/'+_filename for _filename in train_aedat]
    train_list = dvs_gesture2.get_filelist(self.data_dir,self.trials_to_train)
    self.assertEqual(train_list,train_aedat)

  def test_filelist_test(self):
    '''
    Checks if filename_list function works properly.
    The filenames to be included in test list
    are hardcoded in a list. Test function checks if hardcoded
    lists and the lists read from the .txt files are exactly the same.
    '''

    test_aedat = ['user24_fluorescent.aedat',
                  'user24_fluorescent_led.aedat',
                  'user24_led.aedat',
                  'user25_fluorescent.aedat',
                  'user25_led.aedat',
                  'user26_fluorescent.aedat',
                  'user26_fluorescent_led.aedat',
                  'user26_lab.aedat',
                  'user26_led.aedat',
                  'user26_natural.aedat',
                  'user27_fluorescent.aedat',
                  'user27_fluorescent_led.aedat',
                  'user27_led.aedat',
                  'user27_natural.aedat',
                  'user28_fluorescent.aedat',
                  'user28_fluorescent_led.aedat',
                  'user28_lab.aedat',
                  'user28_led.aedat',
                  'user28_natural.aedat',
                  'user29_fluorescent.aedat',
                  'user29_fluorescent_led.aedat',
                  'user29_lab.aedat',
                  'user29_led.aedat',
                  'user29_natural.aedat']

    test_aedat = [self.data_dir+'/'+_filename for _filename in test_aedat]
    test_list = dvs_gesture2.get_filelist(self.data_dir,self.trials_to_test)
    self.assertEqual(test_list,test_aedat)

  def test_get_header_line_aedat(self):
    '''
    Check if get_header_line function works properly.
    The header of the .aedat file created for test 
    is hardcoded in the form of a list. The test checks
    if header_list and the header get by reading the file
    are exactly the same.
    '''
    header_list = [b'#!AER-DAT3.1\r\n', \
      b'#Format: RAW\r\n', \
      b'#Source 1: Test\r\n', \
      b'#Start-Time: 2020-10-30 17:33:59 (TZ+0300)\r\n', \
      b'#!END-HEADER\r\n']

    os.remove(self.aedat_file)
    if not os.path.exists(self.aedat_file):
      create_test_files(self.aedat_file,self.csv_file)
      print(f'{self.aedat_file} has been created!')

    label_filepath = self.aedat_file.replace('.aedat', '_labels.csv')

    if not os.path.exists(label_filepath):
      print(f'{label_filepath} does not exist')

    try:
      with open(self.aedat_file, 'rb') as f:
        header_line = dvs_gesture2.get_header_line_aedat(f)
    except:
      print(f'{self.aedat_file} cannot be read!')

    self.assertEqual(header_list,header_line)

  def test_get_event_header_aedat(self):
    '''
    Check if get_header_line function works properly.
    The headers of the events of in the .aedat file created for test 
    is hardcoded in the form of list of dictionaries. The test checks
    if header_list of dictionaries and the header get by reading the file
    are exactly the same.
    '''
    head1 = {'eventType': 1, 'eventSource': 1, 'eventSize': 8, \
      'eventTSOffset': 4, 'eventTSOverflow': 0, \
      'eventCapacity': 5, 'eventNumber': 5, 'eventValid': 5}

    head2 = {'eventType': 1, 'eventSource': 1, 'eventSize': 8, \
      'eventTSOffset': 4, 'eventTSOverflow': 0, \
      'eventCapacity': 4, 'eventNumber': 4, 'eventValid': 4}

    head3 = {'eventType': 1, 'eventSource': 1, 'eventSize': 8, \
      'eventTSOffset': 4, 'eventTSOverflow': 0, \
      'eventCapacity': 3, 'eventNumber': 3, 'eventValid': 3}

    head_list = [head1, head2, head3]

    os.remove(self.aedat_file)
    if not os.path.exists(self.aedat_file):
      create_test_files(self.aedat_file,self.csv_file)
      print(f'{self.aedat_file} has been created!')

    label_filepath = self.aedat_file.replace('.aedat', '_labels.csv')

    if not os.path.exists(label_filepath):
      print(f'{label_filepath} does not exist')

    headers = []
    try:
      with open(self.aedat_file, 'rb') as f:
        header_line = dvs_gesture2.get_header_line_aedat(f)
        while True:
          event_header = dvs_gesture2.get_event_header_aedat(f)
          if (len(event_header)==0):
            break
          else:
            headers.append(event_header)
          if (event_header['eventType'] == 1): #polarity_event
            event = dvs_gesture2.get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
          else:
            f.read(event_header['eventNumber']*event_header['eventSize'])
    except:
      print(f'{self.aedat_file} cannot be read!')

    self.assertEqual(head_list,headers)


if __name__=='__main__':
  unittest.main()