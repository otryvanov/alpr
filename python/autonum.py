# -*- encoding: utf-8 -*-

import cv2
import time
import os
import alpr
import ConfigParser
import sys
import Queue, threading#, Thread, Lock

if len(sys.argv)>1:
  config=sys.argv[1]
else:
  config=os.path.dirname(os.path.abspath(__file__))
  config=os.path.join(config, 'autonum.cfg')

if not os.path.isfile(config):
  raise IOError("Configuration file %s does not exists" % config)

Config = ConfigParser.ConfigParser()
Config.read(config)

if 'Capture' not in Config.sections():
  raise LookupError('Capture session missing in config')

for key in ['url']:
  if key not in Config.options('Capture'):
    raise LookupError('Key "%s" is missing in Capture session of config' % key)

if 'Transform' not in Config.sections():
  raise LookupError('Transform session missing in config')

for key in ['crop_x', 'crop_y', 'crop_width', 'crop_height']:
  if key not in Config.options('Transform'):
    raise LookupError('Key "%s" is missing in Transform session of config' % key)

if 'Demo' not in Config.sections():
  raise LookupError('Demo session missing in config')

for key in ['show', 'scale', 'text_x', 'text_y', 'caption']:
  if key not in Config.options('Demo'):
    raise LookupError('Key "%s" is missing in Demo session of config' % key)

cam = Config.get('Capture', 'url')

cap = cv2.VideoCapture(cam)
if not cap:
  raise ValueError("Failed VideoCapture")

if 'frame_rate' in Config.options('Capture'):
  frame_rate=int(Config.get('Capture', 'url'))
  cap.set(cv2.cv.CV_CAP_PROP_FPS, frame_rate)

crop_x=int(Config.get('Transform', 'crop_x'))
crop_y=int(Config.get('Transform', 'crop_y'))
crop_width=int(Config.get('Transform', 'crop_width'))
crop_height=int(Config.get('Transform', 'crop_height'))
crop=((crop_y, crop_x), (crop_height, crop_width))

demo_show=int(Config.get('Demo', 'show'))
demo_show_timeout=int(Config.get('Demo', 'show_timeout'))
demo_scale=float(Config.get('Demo', 'scale'))
demo_caption=Config.get('Demo', 'caption')
demo_text_x=int(Config.get('Demo', 'text_x'))
demo_text_y=int(Config.get('Demo', 'text_y'))

datadir=os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(datadir, 'data')

engine=alpr.Engine(datadir, crop, transform=None)

def add_input(input_queue):
  while True:
    input_queue.put(sys.stdin.readline().rstrip('\n\r'))

input_queue = Queue.Queue()
input_thread = threading.Thread(target=add_input, args=(input_queue,))
input_thread.daemon = True
input_thread.start()

waiting=False
waiting_time=time.time()
show_frame=None

print 'ready'

if demo_show:
  cv2.namedWindow(demo_caption, cv2.cv.CV_WINDOW_AUTOSIZE)

#mutex=Lock()

while True:
  # Capture frame-by-frame
  ret, current_frame = cap.read()
  if type(current_frame) == type(None):
    raise ValueError("Couldn't read frame!")
    break
  frame_time=time.time()
  detected=False
  plates=[]

  command=None
  if not input_queue.empty():
    command=input_queue.get()

    if command=='quit':
      break
    elif command=='test':
      print 'ok'
    elif command=='get':
      plates=engine.detect(current_frame, '')
      detected=(len(plates)>0)
      print "number="+str(plates[0])

  if demo_show:
    if show_frame==None or not waiting or detected:
      show_frame=current_frame

    if waiting and frame_time-waiting_time>demo_show_timeout:
      waiting=False

    cv2.rectangle(show_frame, (crop_x, crop_y), (crop_x+crop_width, crop_y+crop_height), (255,0,0), 2)

    if detected:
      waiting=True
      waiting_time=frame_time

      text=plates[0]
      cv2.putText(show_frame, text, (demo_text_x, demo_text_y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 7, cv2.CV_AA);

    # Display the resulting frame
    frame=cv2.resize(show_frame, (0,0), fx=demo_scale, fy=demo_scale)
    cv2.imshow(demo_caption, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# release the capture
cap.release()
cv2.destroyAllWindows()
