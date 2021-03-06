# -*- encoding: utf-8 -*-

import cv2
import time
import os
import alpr
import ConfigParser
import sys
import Queue, threading
import gc

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

def fail(message, code=None):
  print 'fail'
  print >> sys.stderr, message
  if code is not None:
    try:
      code=int(code)
    except:
      code=1
    finally:
      sys.exit(code)

if getattr(sys, 'frozen', False):
  #frozen
  exec_dir = os.path.dirname(sys.executable)
else:
  #unfrozen
  exec_dir = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv)>1:
  config=sys.argv[1]
  if not os.path.isabs(config) and not os.path.isfile(config):
    config=os.path.join(exec_dir, config)
else:
  config=os.path.join(exec_dir, 'autonum.cfg')

if not os.path.isfile(config):
  fail("Configuration file %s does not exists" % config, -1)

try:
  Config = ConfigParser.ConfigParser()
  Config.read(config)
except:
  fail('Could not parse config file', -1)

if 'Capture' not in Config.sections():
  fail('Capture session missing in config', -1)

for key in ['url']:
  if key not in Config.options('Capture'):
    fail('Key "%s" is missing in Capture session of config' % key, -1)

if 'frame_rate' in Config.options('Capture'):
  try:
    frame_rate=float(Config.get('Capture', 'frame_rate'))
  except Exception:
    fail('Key "frame_rate" is used in Capture session of config but not float', -1)
else:
  frame_rate=None

if 'loop' in Config.options('Capture'):
  try:
    capture_loop=int(Config.get('Capture', 'loop'))
  except Exception:
    fail('Key "loop" is used in Capture session of config but not integer', -1)
else:
  capture_loop=0

if frame_rate is not None and 'force_frame_rate' in Config.options('Capture'):
  try:
    force_frame_rate=int(Config.get('Capture', 'force_frame_rate'))
  except Exception:
    fail('Key "force_frame_rate" is used in Capture session of config but not integer', -1)
else:
  force_frame_rate=None

if 'Transform' not in Config.sections():
  fail('Transform session missing in config', -1)

for key in ['crop_x', 'crop_y', 'crop_width', 'crop_height']:
  if key not in Config.options('Transform'):
    fail('Key "%s" is missing in Transform session of config' % key, -1)

try:
  crop_x=int(Config.get('Transform', 'crop_x'))
except Exception:
  fail('Key "crop_x" is used in Transform session of config or not integer', -1)

try:
  crop_y=int(Config.get('Transform', 'crop_y'))
except Exception:
  fail('Key "crop_x" is used in Transform session of config but not integer', -1)

try:
  crop_width=int(Config.get('Transform', 'crop_width'))
except Exception:
  fail('Key "crop_width" is used in Transform session of config but not integer', -1)

try:
  crop_height=int(Config.get('Transform', 'crop_height'))
except Exception:
  fail('Key "crop_width" is used in Transform session of config but not integer', -1)

crop=((crop_y, crop_x), (crop_height, crop_width))

if 'Demo' not in Config.sections():
  fail('Demo session missing in config', -1)

for key in ['show']:
  if key not in Config.options('Demo'):
    fail('Key "%s" is missing in Demo session of config' % key, -1)

try:
  demo_show=int(Config.get('Demo', 'show'))
except Exception:
  fail('Key "show" is used in Demo session of config but not integer', -1)

if demo_show:
  for key in ['scale', 'text_x', 'text_y', 'caption', 'show_timeout']:
    if key not in Config.options('Demo'):
      fail('Key "%s" is missing in Demo session of config' % key, -1)

try:
  demo_show_timeout=int(Config.get('Demo', 'show_timeout'))
except Exception:
  fail('Key "show_timeout" is used in Demo session of config but not integer', -1)

try:
  demo_scale=float(Config.get('Demo', 'scale'))
except Exception:
  fail('Key "scale" is used in Demo session of config but not float', -1)

demo_caption=Config.get('Demo', 'caption')

try:
  demo_text_x=int(Config.get('Demo', 'text_x'))
except Exception:
  fail('Key "text_x" is used in Demo session of config but not integer', -1)

try:
  demo_text_y=int(Config.get('Demo', 'text_y'))
except Exception:
  fail('Key "text_y" is used in Demo session of config but not integer', -1)

try:
  cam = Config.get('Capture', 'url')
  cap = cv2.VideoCapture(cam)
  if not cap.isOpened():
    fail('Could not open VideoCapture', -1)
  capture_frames=cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
  if frame_rate is not None:
    cap.set(cv2.cv.CV_CAP_PROP_FPS, frame_rate)
  if frame_rate is None or not force_frame_rate:
    frame_rate=cap.get(cv2.cv.CV_CAP_PROP_FPS)
except Exception:
  fail('Failed VideoCapture', -1)

datadir=os.path.join(exec_dir, 'data')

if not os.path.isdir(datadir):
  fail("Data dir %s does not exists" % datadir, -1)

for datafile in ['haarcascade_russian_plate_number.xml', 'svm_left_border.xml', 'svm_right_border.xml',
                 'svm_plate.xml', 'svm_vertical_stage1.xml'] + \
                ['svm_letter_%s_stage_1.xml' % l for l in ['0O', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                                           'A', 'B', 'C', 'E', 'H', 'K', 'M', 'P', 'T', 'X', 'Y',
                                                            '8dot', 'dotB', 'dotC', 'dotE', 'dotH', 'dotM', 'dotO', 'dotP']]:
  if not os.path.isfile(os.path.join(datadir, datafile)):
    fail("Data file %s does not exists" % datafile, -1)

try:
  engine=alpr.Engine(datadir, crop, transform=None)
except Exception:
  fail('Coult not init ALPR engine', -1)

def add_input(input_queue):
  line=sys.stdin.readline()
  while line:
    input_queue.put(line.rstrip('\n\r'))
    line=sys.stdin.readline()
  input_queue.put('quit')

input_queue = Queue.Queue()
input_thread = threading.Thread(target=add_input, args=(input_queue,))
input_thread.daemon = True
input_thread.start()

def add_alpr_work(input_queue, output_queue):
  while True:
    try:
      frame=input_queue.get()
      plates=engine.detect(frame, '')
      detected=(len(plates)>0)

      if detected:
        output_queue.put((frame, detected, "%s,%d,%d,%d,%d" % ((plates[0][0],)+plates[0][1])))
      else:
        output_queue.put((frame, detected, '?'))

      #force garbage collection after engine.detect to avoid excessive memory usage
      gc.collect()
    except:
      fail('General ALPR engine failure')

alrp_input_queue = Queue.Queue()
alrp_output_queue = Queue.Queue()
alpr_work_thread = threading.Thread(target=add_alpr_work, args=(alrp_input_queue, alrp_output_queue,))
alpr_work_thread.daemon = True
alpr_work_thread.start()

waiting=False
waiting_time=time.time()
recognizing=False
show_frame=None

print 'ready'

if demo_show:
  cv2.namedWindow(demo_caption, cv2.cv.CV_WINDOW_AUTOSIZE)

#arbitrary limit
is_loop_possible = (capture_loop and (capture_frames>0) and (capture_frames<1e6))

frame_start_time=time.time()
frame_counter=0
while True:
  #rewind
  frame_counter+=1
  if is_loop_possible and frame_counter>=capture_frames:
    frame_counter = 0
    try:
      cam = Config.get('Capture', 'url')
      cap = cv2.VideoCapture(cam)
      capture_frames=cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
      if frame_rate is not None:
        cap.set(cv2.cv.CV_CAP_PROP_FPS, frame_rate)
      frame_start_time=time.time()
    except Exception:
      fail('Failed VideoCapture', -1)

  # Capture frame-by-frame
  ret, current_frame = cap.read()
  if type(current_frame) == type(None):
    fail("Couldn't read frame!", -1)
  frame_time=time.time()

  command=None
  if not input_queue.empty():
    command=input_queue.get()
    cmd=command.split(" ", 1)
    command=cmd[0]
    params=None
    if len(cmd)>1:
      params=cmd[1]

    if command=='quit':
      break
    elif command=='test':
      print 'ok'
    elif command=='get':
      recognizing=True
      show_frame=current_frame
      if params is not None and len(params)>0:
        try:
          if not cv2.imwrite(params, current_frame):
            fail("Could not save frame to file %s" % params)
        except:
          fail("Could not save frame to file %s" % params)
      alrp_input_queue.put(current_frame)
    elif command=='save':
      if params is not None and len(params)>0:
        try:
          if not cv2.imwrite(params, current_frame):
            fail("Could not save frame to file %s" % params)
        except:
          fail("Could not save frame to file %s" % params)

  detected=False
  plate=None
  if not alrp_output_queue.empty():
    show_frame, detected, plate=alrp_output_queue.get()
    recognizing=False
    detected=True
    if detected:
      waiting=True
      waiting_time=frame_time

    print "number="+str(plate)

  if demo_show:
    if show_frame is None or not (waiting or recognizing):
      show_frame=current_frame

    if waiting and frame_time-waiting_time>demo_show_timeout:
      waiting=False

    cv2.rectangle(show_frame, (crop_x, crop_y), (crop_x+crop_width, crop_y+crop_height), (255,0,0), 2)

    if detected:
      waiting=True
      waiting_time=frame_time

      text=plate

      font_face=cv2.FONT_HERSHEY_SIMPLEX
      font_scale=4
      font_thickness=7

      size=cv2.getTextSize(text, font_face, font_scale, font_thickness)
      cv2.putText(show_frame, text, (demo_text_x, demo_text_y+size[0][1]), font_face, font_scale, (0, 0, 255), font_thickness, cv2.CV_AA)

    # Display the resulting frame
    frame=cv2.resize(show_frame, (0,0), fx=demo_scale, fy=demo_scale)
    cv2.imshow(demo_caption, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  new_frame_start_time=time.time()
  sleep=1.0/frame_rate-new_frame_start_time+frame_start_time
  if sleep>0:
    time.sleep(sleep)
  frame_start_time=new_frame_start_time+sleep

# release the capture
cap.release()
cv2.destroyAllWindows()
