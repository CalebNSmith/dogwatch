import tornado.ioloop
import tornado.web
import tornado.websocket
import random
import socket
import asyncio
import threading
import time
import cv2
import queue
import math
import struct
import datetime
import os
import signal
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import config
import tensorflow as tf
from tornado.ioloop import IOLoop
from tornado.options import define, options, parse_command_line
define('port', default=8888, type=int)


#Various global values used in the separate threads

android_client = None
android_rtsp_uri = None

exit_flag = False

io_loop = None

is_model_loaded = False

close = False

webcam_client = None
webcam_thread = None
webcam_stream = False

cam = None


'''
    Websocket that interacts with the various clients, such as
    an android device or a webcam on a separate computer. This manages
    which VideoCapture object is in use for the TensorFlowThread to be
    reading from.
'''
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    
    def open(self, *args):
        print("new connection " + str(self.request.remote_ip))
        self.write_message(self.request.remote_ip)

    def on_message(self, message):
        global android_client
        global android_rtsp_uri
        global webcam_client
        global webcam_thread
        global webcam_stream
        global cam
        global close

        if "android" in message:
            android_client = self
            print("android client connected")
            if "rtsp" in message:
                rtsp_uri = message.split(" ")[1]
                android_rtsp_uri = rtsp_uri
                if cam is not None:
                    cam.close_connection()
                    cam = None
                cam = VideoCapture(android_rtsp_uri, 10, False)

            elif "start stream" in message:
                cam = VideoCapture(0, 10, True)
                if webcam_client is not None:
                    webcam_client.write_message("start")
                    cam = VideoCapture(0, 10, True)
                    #cam = VideoCapture("udp://192.168.1.42:8888", 10, True)
                else:
                    android_client.write_message("webcam not connected")
                
            elif "end stream" in message:
                if webcam_client is not None:
                    webcam_client.write_message("end")
                else:
                    android_client.write_message("webcam not connected")

                if cam is not None:
                    cam.close_connection()
                    cam = None

            elif "close" in message:
                close = True
                if cam is not None:
                    cam.close_connection()
                    cam = None

            elif "restart" in message:
                os.kill(os.getpgrp(), signal.SIGKILL)

        elif "camera" in message:
            webcam_client = self
        
        if android_client is not None:
            android_client.write_message("model loaded: " + str(is_model_loaded))

    def on_close(self):
        global android_rtsp_uri
        global android_client
        global cam
        global close

        if self is android_client:
            android_rtsp_uri = None
            android_client = None
            if cam is not None:
                cam.close_connection()
                cam = None

            if webcam_client is not None:
                webcam_client.write_message("end")
            print("android connection closed")
        
        close = False
        print("connection closed")

'''
    Handles connection to the input stream (usb webcam or stream) and maintains
    the most recently read frame, so the model is running inference on the most
    up to date frame.
'''
class VideoCapture:
    def __init__(self, name, attempts=10, forward_stream=False):
        self.forward_stream = forward_stream
        self.cap = cv2.VideoCapture()
       
       #do multiple reconnect attemps in case stream takes a while to start
        while attempts > 0:
            try:
                self.cap.open(name)
                if self.cap is not None and self.is_opened():
                    print("connection successful")
                    break
                else:
                    print("attempting reconnect, attempts remaining: " + str(attempts))
                    self.cap.release()
                    attempts -= 1
                    time.sleep(1)
            except Exception as e:
                print(e)
                print("connection error")

        if self.is_opened():
            self.once_connected()

    '''
        start threads once connected to input device or stream
    '''
    def once_connected(self):
        self.q = queue.Queue()
        self.end_connection = False
        read_thread = threading.Thread(target=self._frame_read)
        read_thread.daemon = True
        read_thread.start()

        if forward_stream:
            forward_thread = threading.Thread(target=self._forward_stream)
            forward_thread.daemon = True
            forward_thread.start()

    ''' Maintains the queue to contain the most recent frame.
        
        Typically ran as a separate thread.
    '''
    def _frame_read(self):
        global close
        while not self.end_connection and not close:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
    '''
        Takes the most recent frame and sends it via socket
        to an endpoint (e.g. android device) for viewing.
        This is to allow a stream of frames to be intercepted
        for inference, while also being able to view the frames.
        
        Typically ran as a separate thread.
    '''
    def _forward_stream(self):
        global close
        global android_client

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        MAX_DGRAM = 2**16
        MAX_IMAGE_DGRAM = MAX_DGRAM - 64
        UDP_IP = android_client.request.remote_ip #"192.168.1.39" #android ip
        UDP_PORT = 12345
        
        

        while not self.end_connection and not close:
            img = self.read()

            compress_img = cv2.imencode('.jpg', img)[1]
            dat = compress_img.tostring()
            num_of_segments = math.ceil(size/(MAX_IMAGE_DGRAM))
            array_pos_start = 0
            try:
                while num_of_segments:
                    array_pos_end = min(size, array_pos_start + MAX_IMAGE_DGRAM)
                    s.sendto(struct.pack("B", num_of_segments) +
                        dat[array_pos_start:array_pos_end],
                        (UDP_IP, UDP_PORT)
                        )
                    array_pos_start = array_pos_end
                    num_of_segments -= 1
            except Exception as e:
                print(e)
                print("socket send failed")

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        return self.q.get()

    def close_connection(self):
        self.end_connection = True
        self.cap.release()


'''
    Separate thread that loads the tensorflow model and waits for the VideoCapture object to be
    instantiated, then runs inference on those frames and sends the results to the android
    client via websocket. The model is maintained in memory between VideoCapture objects swaps, so
    it only has to load once.
'''
class TensorFlowThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def send_values(self, values):
        global android_client
        
        try:
            if android_client is not None:
                android_client.write_message(values)
        except Exception as e:
            print(e)
            print("send values error")

    def send_model_loaded(self):
        global android_client
        if android_client is not None:
            android_client.write_message("model loaded: " + str(is_model_loaded))
    
    #to call a function on the websocket thread
    def call(self, cbfn, *args):
        global io_loop
        
        ioloop = io_loop
        if len(args) > 0:
            ioloop.add_callback(cbfn, args[0])
        else:
            ioloop.add_callback(cbfn)


    def run(self):
        global is_model_loaded
        global cam
        global close

        physical_devices = config.list_physical_devices('GPU')
        model = load_model('../saved_models/reduce_on_plateau-fine_tuned')
        
        is_model_loaded = True
        self.call(self.send_model_loaded)

        while not exit_flag:
            while cam is None or not cam.is_opened():
                time.sleep(1)
                print("waiting for camera connection")

            while cam is not None and not close:
                try:
                    #get and resize the most recent frame
                    frame = cam.read()
                    frame = cv2.resize(frame, (224, 224))

                    #convert the image and run inference
                    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
                    classes = model.predict(input_tensor, verbose=1)

                    #round the output and send it to the client via websocket
                    classes = np.round(classes, decimals=2)
                    self.call(self.send_values, "prediction:" + str(classes[0][0]) + ',' + str(classes[0][1]) + ',' + str(classes[0][2]))
                except Exception as e:
                    print(e)
                    print("CAMERA READ ERROR")
                    cam.close_connection()
                    cam = None
                    break


if __name__ == '__main__':
    app = tornado.web.Application([
        (r'/ws/', WebSocketHandler)
    ])
    serv = tornado.httpserver.HTTPServer(app)
    serv.listen(8888)

    io_loop = IOLoop.current()
    
    print("starting tensorflow thread")
    tf_thread = TensorFlowThread()
    tf_thread.start()

    print("starting websocket")
    io_loop.start()
    
