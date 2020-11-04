#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:35:33 2020

@author: ugurc
"""

import unittest
import dvs_gesture2
import os
import numpy as np

_version = b'#!AER-DAT3.1\r\n'
_format = b'#Format: RAW\r\n'
_source = b'#Source 1: Test\r\n'
_start_time = b'#Start-Time: 2020-10-30 17:33:59 (TZ+0300)\r\n'
_end_line = b'#!END-HEADER\r\n'

header_line = _version+_format+_source+_start_time+_end_line

event_head_keys = ['eventType', 'eventSource', 'eventSize',
                   'eventTSOffset', 'eventTSOverflow',
                   'eventCapacity', 'eventNumber', 'eventValid']

event_head1 = [1, 1, 8, 4, 0, 5, 5, 5]
event_head2 = [1, 1, 8, 4, 0, 4, 4, 4]
event_head3 = [1, 1, 8, 4, 0, 3, 3, 3]

event_seq1 = [[10,0,0,1],[25,10,20,0],[38,120,59,1],[48,5,9,0],[59,9,5,1]]
event_seq2 = [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1],[126562,45,59,1]]
event_seq3 = [[2068981,127,127,0],[2157619,0,15,1],[2178812,1,1,1]]

label_header = "class,startTime_usec,endTime_usec"
event_labels = np.asarray([[1,24,50],
                           [5,102865,126562],
                           [11,2157620,2178812]])

event_seq1_clipped = [[25,10,20,0],[38,120,59,1],[48,5,9,0]]
event_seq2_clipped = [[102865,15,19,1],[102898,28,36,0],[119579,32,44,1]]
event_seq3_clipped = []

aedat_file = 'aedat_for_testing.aedat'
labels_file = 'aedat_for_testing_labels.csv'

def create_test_files(aedat_file,labels_file):
  '''
  Creates an aedat file and it's labels to test the functions 
  defined to read a real one. The function won't create files 
  if they are already exist.

  In the mock file there are 3 event records.

  event_seq1 includes 5 events and it's 
  first and last events are to be clipped. 

  event_seq2 includes 4 events and labels chosen in a way 
  to show that the interval to be considered is [startTime_usec, endTime_usec)

  event_seq3 includes 3 events and all will be clipped

  Event headers are filled to represent sequences carefully.

    Arguments:
      aedat_file(string):
        filepath to store the mock aedat file

      labels_file(string):
        filepath to store the labels.csv file
  '''
  
  # EVENTS
  if os.path.exists(aedat_file):
    print(f'{aedat_file} has already exist!')

  else:
    event_head = [event_head1, event_head2, event_head3]
    event_seq = [event_seq1, event_seq2, event_seq3]
    event_it = zip(event_head,event_seq)
    try:
      with open(aedat_file, 'ab') as f:
        create_file_header(f)
        for head, seq in event_it:
          create_event_header(f,head)
          for event in seq:
            create_polarity_event(f,event)
    except:
      print(f'Unable to create {aedat_file}!')

    else:
      print(f'{aedat_file} has been created succesfully!')

  # LABELS
  if os.path.exists(labels_file):
    print(f'{labels_file} has already exist!')
  else:
    try:
      np.savetxt(labels_file, event_labels, 
                 fmt='%u', delimiter=',', 
                 header=label_header, comments='')
    except:
      print(f'Unable to create {labels_file}!')

    else:
      print(f'{labels_file} has been created succesfully!')

def create_file_header(_f):
  '''
  Writes the header line to the given file.
  It's a slave function called by create_test_files()

    Arguments:
      _f(output file): 
        file to be written
  '''
  try:
    _f.write(header_line)
  except:
    print('Unable to write the header line!')

def create_event_header(_f,header):
  '''
  Writes the given event header to the given file.
  It's a slave function called by create_test_files()

  Converts integer values to their little endian binary representation

    Arguments:
      _f(output file): 
        file to be written
      header(int list):
        header in the form of an integer list
  '''

  eventType = header[0].to_bytes(2,byteorder='little')
  eventSource = header[1].to_bytes(2,byteorder='little')
  eventSize = header[2].to_bytes(4,byteorder='little')
  eventTSOffset = header[3].to_bytes(4,byteorder='little')
  eventTSOverflow = header[4].to_bytes(4,byteorder='little')
  eventCapacity = header[5].to_bytes(4,byteorder='little') 
  eventNumber = header[6].to_bytes(4,byteorder='little')
  eventValid = header[7].to_bytes(4,byteorder='little')

  try:
    _f.write(eventType+eventSource+eventSize+\
             eventTSOffset+eventTSOverflow+eventCapacity+\
             eventNumber+eventValid) 
  except:
    print(f'Unable to write the event header {header}')

def create_polarity_event(_f,seq):
  '''
  Writes the given event to the given file 
  It's a slave function called by create_test_files()

  Using the compression method defined in 
  https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html#aedat-31
  and 
  https://www.research.ibm.com/dvsgesture/,

  express 
  32 bits time step, 
  15 bits x position, 
  15 bits y position, 
  1 bit polarization,
  1 bit validity check
  in 64 bits  

    Arguments:
      _f(output file): 
        file to be written
      seq(int list):
        [time, xpos, ypos, pol] in the form of an integer list
  '''
  time = seq[0]
  data = 0

  data = data | seq[1]<<17
  data = data | seq[2]<<2
  data = data | seq[3]<<1

  try:
    _f.write(data.to_bytes(4,byteorder='little')+time.to_bytes(4,byteorder='little'))
  except:
    print(f'Unable to write polarity_event {seq}')


class TestDVSGesture(unittest.TestCase):  
  data_dir = '/home/ugurc/drive/data/DvsGesture'
  trials_to_train = '/trials_to_train.txt'
  trials_to_test = '/trials_to_test.txt'



  def test1_filelist_train(self):
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
    print("Train Filelist Test has been passed!")

  def test2_filelist_test(self):
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
    print("Test Filelist Test has been passed!")

  def test3_get_header_line_aedat(self):
    '''
    Check if get_header_line function works properly.
    The header of the .aedat file created for test 
    is hardcoded in the form of a list. The test checks
    if header_list and the header get by reading the file
    are exactly the same.
    '''
    header_list = [_version, _format, _source, _start_time, _end_line]

    try:
      with open(aedat_file, 'rb') as f:
        header_line = dvs_gesture2.get_header_line_aedat(f)
    except:
      print(f'{aedat_file} cannot be read!')

    self.assertEqual(header_list,header_line)
    print("Header Line Test passed!")

  def test4_get_event_header_aedat(self):
    '''
    Check if get_header_line function works properly.
    The headers of the events of in the .aedat file created for test 
    is hardcoded in the form of list of dictionaries. The test checks
    if header_list of dictionaries and the header get by reading the file
    are exactly the same.
    '''
    head1 = dict(zip(event_head_keys, event_head1))
    head2 = dict(zip(event_head_keys, event_head2))
    head3 = dict(zip(event_head_keys, event_head3))
    head_list = [head1, head2, head3]

    headers = []
    try:
      with open(aedat_file, 'rb') as f:
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
      print(f'{aedat_file} cannot be read!')

    self.assertEqual(head_list,headers)
    print("Event Header Test has been passed!")

  def test5_get_polarity_event_aedat(self):
    '''
    Check if get_polarity_event_aedat function works properly.
    The polarity events of in the .aedat file created for test 
    is hardcoded in the form of list of list of list of numpy arrays.
    The test checks if the hardcoded polarity events are the same as 
    the extracted ones. This shows us that function works as expected.
    '''

    seq1 = np.column_stack(event_seq1).astype(np.uint32)
    seq2 = np.column_stack(event_seq2).astype(np.uint32)
    seq3 = np.column_stack(event_seq3).astype(np.uint32)

    seq1 = [event for event in seq1]
    seq2 = [event for event in seq2]
    seq3 = [event for event in seq3]

    event_list = [seq1,seq2,seq3]

    try:
      events = []
      with open(aedat_file, 'rb') as f:
        header_line = dvs_gesture2.get_header_line_aedat(f)
        while True:
          event_header = dvs_gesture2.get_event_header_aedat(f)
          if (len(event_header)==0):
            break
          if (event_header['eventType'] == 1): #polarity_event
            event = dvs_gesture2.get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
            events.append(event)
          else:
            f.read(event_header['eventNumber']*event_header['eventSize'])
    except:
      print(f'{aedat_file} cannot be read!')

    for list1, list2 in zip(events, event_list):
      for event1, event2 in zip(list1, list2):
        self.assertTrue((event1 == event2).all())

    print("Polarity Event Extraction Test has been passed!")

  def test6_arrange_aedat_events(self):
    '''
    Check if arrange_aedat_events function works properly.
    It has to clip the event sequences depending on given start and 
    end times in the labels csv file. Also the function needs to
    store the labeled event sequences in the form of numpy array of 
    2D numpy arrays.
    Also the event labels should be stored in a 1D numpy array 
    '''

    _labels = np.asarray([lab[0] for lab in event_labels])

    seq1_clipped = np.array(event_seq1_clipped, dtype = np.uint32)
    seq2_clipped = np.array(event_seq2_clipped, dtype = np.uint32)
    seq3_clipped = np.array(event_seq3_clipped, dtype=np.uint32)

    _events = np.array([seq1_clipped, seq2_clipped, seq3_clipped])

    labels = np.genfromtxt(labels_file, delimiter=',', dtype=np.uint32)

    try:
      events = []
      with open(aedat_file, 'rb') as f:
        header_line = dvs_gesture2.get_header_line_aedat(f)
        while True:
          event_header = dvs_gesture2.get_event_header_aedat(f)
          if (len(event_header)==0):
            break
          if (event_header['eventType'] == 1): #polarity_event
            event = dvs_gesture2.get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
            events.append(event)
          else:
            f.read(event_header['eventNumber']*event_header['eventSize'])

      events, labels = dvs_gesture2.arrange_aedat_events(events,labels)
    except:
      print(f'{aedat_file} cannot be read by me!')

    self.assertTrue((labels == _labels).all())
    print("Label arrange test has been passed!(1/2)")

    for events1, events2 in zip(events,_events):
      for event1, event2 in zip(events1, events2):
        self.assertTrue((event1 == event2).all())

    print("Event arrange test has been passed!(2/2)")

  def test7_aedat_to_np(self):
    '''
    From test3 through test6, we have tested the subroutines of the 
    aedat_to_np() funtion. This test aims to prove that aedat to numpy array
    process works perfectly on the test file. 
    '''

    _labels = np.asarray([lab[0] for lab in event_labels])

    seq1_clipped = np.array(event_seq1_clipped, dtype = np.uint32)
    seq2_clipped = np.array(event_seq2_clipped, dtype = np.uint32)
    seq3_clipped = np.array(event_seq3_clipped, dtype=np.uint32)

    _events = np.array([seq1_clipped, seq2_clipped, seq3_clipped])

    events, labels = dvs_gesture2.aedat_to_np(aedat_file)

    self.assertTrue((labels == _labels).all())

    for events1, events2 in zip(events,_events):
      for event1, event2 in zip(events1, events2):
        self.assertTrue((event1 == event2).all())

    print('Aedat to numpy array test has been passed!')


if __name__=='__main__':
  create_test_files(aedat_file,labels_file)
  unittest.main()