'''
Unittest for process_aedat.py file
The functions are tested using a small aedat file created artificially. 
Properties of the test file can be investigated in test_env.py file.

author: ugurc
201022
'''
import sys
sys.path.insert(0,'..')

import unittest
import process_aedat as pa
import aedat_file as af
from aedat_file import AedatFile
import test_env as env
import numpy as np

data_dir = '/home/ugurc/drive/data/DvsGesture'
trials_to_train = '/trials_to_train.txt'
trials_to_test = '/trials_to_test.txt'
test_file = 'test_files/aedat1_for_testing.pickle'

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

class TestProcessAedat(unittest.TestCase):  

  def test1_filelist_train(self):
    '''
    Checks if filename_list function works properly.
    The filenames to be included in train list
    are hardcoded in a list. Test function checks if hardcoded
    lists and the lists read from the .txt files are exactly the same.
    '''
    _train_list = [data_dir+'/'+_filename for _filename in train_aedat]
    train_list = pa.get_filelist(data_dir,trials_to_train)
    self.assertEqual(train_list,_train_list)
    print("Train Filelist Test has been passed!\n")

  def test2_filelist_test(self):
    '''
    Checks if filename_list function works properly.
    The filenames to be included in test list
    are hardcoded in a list. Test function checks if hardcoded
    lists and the lists read from the .txt files are exactly the same.
    '''

    _test_list = [data_dir+'/'+_filename for _filename in test_aedat]
    test_list = pa.get_filelist(data_dir,trials_to_test)
    self.assertEqual(test_list,_test_list)
    print("Test Filelist Test has been passed!\n")

  def test3_get_header_line_aedat(self):
    '''
    Check if get_header_line function works properly.
    The header of the .aedat file created for test 
    is hardcoded in the form of a list. The test checks
    if header_list and the header get by reading the file
    are exactly the same.
    '''
    test = af.load_aedat_pickle(test_file)
    header_list = [test._version, test._format, test._source, 
                   test._start_time, test._end_line]

    aedat_file = test.filepath + '.aedat'    
    print(aedat_file)            
    try:
      with open(aedat_file, 'rb') as f:
        header_line = pa.get_header_line_aedat(f)
    except:
      print(f'{aedat_file} cannot be read!')

    self.assertEqual(header_list,header_line)
    print("Header Line Test passed!\n")

  def test4_get_event_header_aedat(self):
    '''
    Check if get_header_line function works properly.
    The headers of the events of in the .aedat file created for test 
    is hardcoded in the form of list of dictionaries. The test checks
    if header_list of dictionaries and the header get by reading the file
    are exactly the same.
    '''
    test = af.load_aedat_pickle(test_file)
    aedat_file = test.filepath + '.aedat'
    _headers = [dict(zip(test.event_head_keys,head)) for head in test.event_head]
    headers = []
    try:
      with open(aedat_file, 'rb') as f:
        header_line = pa.get_header_line_aedat(f)
        while True:
          event_header = pa.get_event_header_aedat(f)
          if (len(event_header)==0):
            break
          else:
            headers.append(event_header)
          if (event_header['eventType'] == 1): #polarity_event
            event = pa.get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
          else:
            f.read(event_header['eventNumber']*event_header['eventSize'])
    except:
      print(f'{aedat_file} cannot be read!')

    self.assertEqual(_headers,headers)
    print("Event Header Test has been passed!\n")

  def test5_get_polarity_event_aedat(self):
    '''
    Check if get_polarity_event_aedat function works properly.
    The polarity events of in the .aedat file created for test 
    is hardcoded in the form of list of list of list of numpy arrays.
    The test checks if the hardcoded polarity events are the same as 
    the extracted ones. This shows us that function works as expected.
    '''
    test = af.load_aedat_pickle(test_file)
    aedat_file = test.filepath + '.aedat'
    seqs = [np.column_stack(seq).astype(np.uint32) for seq in test.event_seq]
    event_list = [[event for event in seq] for seq in seqs]

    try:
      events = []
      with open(aedat_file, 'rb') as f:
        header_line = pa.get_header_line_aedat(f)
        while True:
          event_header = pa.get_event_header_aedat(f)
          if (len(event_header)==0):
            break
          if (event_header['eventType'] == 1): #polarity_event
            event = pa.get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
            events.append(event)
          else:
            f.read(event_header['eventNumber']*event_header['eventSize'])
    except:
      print(f'{aedat_file} cannot be read!')

    for list1, list2 in zip(events, event_list):
      for event1, event2 in zip(list1, list2):
        self.assertTrue((event1 == event2).all())

    print("Polarity Event Extraction Test has been passed!\n")

  def test6_arrange_aedat_events(self):
    '''
    Check if arrange_aedat_events function works properly.
    It has to clip the event sequences depending on given start and 
    end times in the labels csv file. Also the function needs to
    store the labeled event sequences in the form of numpy array of 
    2D numpy arrays.
    Also the event labels should be stored in a 1D numpy array 
    '''
    test = af.load_aedat_pickle(test_file)
    aedat_file = test.filepath + '.aedat'
    labels_file = test.filepath + '_labels.csv'

    _labels = np.asarray([lab[0] for lab in test.event_labels])
    _events = np.asarray([np.asarray(seq,dtype=np.uint32) for seq in test.seq_clipped])

    labels = np.genfromtxt(labels_file, delimiter=',', dtype=np.uint32)

    try:
      events = []
      with open(aedat_file, 'rb') as f:
        header_line = pa.get_header_line_aedat(f)
        while True:
          event_header = pa.get_event_header_aedat(f)
          if (len(event_header)==0):
            break
          if (event_header['eventType'] == 1): #polarity_event
            event = pa.get_polarity_event_aedat(f,event_header['eventNumber'],event_header['eventSize'])  
            events.append(event)
          else:
            f.read(event_header['eventNumber']*event_header['eventSize'])

      events, labels = pa.arrange_aedat_events(events,labels)
    except:
      print(f'{env.aedat_file} cannot be read!')

    self.assertTrue((labels == _labels).all())
    print("Label arrange test has been passed!(1/2)\n")

    for events1, events2 in zip(events,_events):
      for event1, event2 in zip(events1, events2):
        self.assertTrue((event1 == event2).all())

    print("Event arrange test has been passed!(2/2)\n")

  def test7_aedat_to_np(self):
    '''
    From test3 through test6, we have tested the subroutines of the 
    aedat_to_np() funtion. This test aims to prove that aedat to numpy array
    process works perfectly on the test file. 
    '''

    test = af.load_aedat_pickle(test_file)
    aedat_file = test.filepath + '.aedat'
    labels_file = test.filepath + '_labels.csv'

    _labels = np.asarray([lab[0] for lab in test.event_labels])
    _events = np.asarray([np.asarray(seq,dtype=np.uint32) for seq in test.seq_clipped])

    events, labels = pa.aedat_to_np(aedat_file)

    self.assertTrue((labels == _labels).all())

    for events1, events2 in zip(events,_events):
      for event1, event2 in zip(events1, events2):
        self.assertTrue((event1 == event2).all())

    print('Aedat to numpy array test has been passed!\n')

if __name__=='__main__':
  env.clear_env()
  env.process_aedat_env()
  unittest.main()