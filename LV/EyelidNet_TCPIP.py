# Real time eyeblink detection
#  Vitaly Lerner 2022
# Pipeline:
#   Images are acquired using Labview at >300fps
#   Images are stored on HD
#   Portion of the images are queued for pupillometry
#   The images are sent through TCP protocolto python(this module), at controlled framerate,~30fps
#   This script receives images 
#   Each image is processed with MEYE (pupillomety.it)
#     Github of MEYE: github.com/fabiocarrara/meye/
#     Publication of MEYE: 10.5281/zenodo.4488164
#   MEYE model estimates probability of eyeblink
#   The estimation is sent back to Labview
#   Labview generates a eyelink signal
#   Based on the signal, a trigger to start an experimental procedure is generated

import socket
import struct
from numpy import *
import logging

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import cv2 as cv
from skimage.measure import label, regionprops

# Import the NN model
from tensorflow.keras.models import load_model
MODELPATH = r"MEYE\meye-segmentation_i128_s4_c1_f16_g1_a-relu.hdf5"
model = load_model(MODELPATH)
requiredFrameSize = model.input.shape[1:3]

    
logging.basicConfig(filename='PyEyeblink.log',
                     filemode='w',
                     level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d  %(module)s - %(funcName)s: %(message)s',
    datefmt='%H:%M:%S',)
log_img=False

logging.info("Initialized")
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 2055  # Port to listen on (non-privileged ports are > 1023)
s= socket.socket()#socket.AF_INET, socket.SOCK_STREAM) 

def check_socket(msg):
    #check if received correctly
    if msg==b'':
        raise RuntimeError("socket connection broken") 

def number_to_byte(num):
    #convert 8bit (byte) integer to array of 8 bits (bools)
    return [bool((num >> i) & 1) for i in range(8)]    


try:
    s.connect((HOST,PORT))

    logging.info ('connected')
    
    bConn=True
except:
    bConn=False
    logging.error('No connection')
#Constants and enumerators
MSG_HANDSHAKE=0
MSG_IMG=1
MSG_TERM=2
TCP_RESPONSE=struct.pack('b',101)






#*******************************************************
#*********Communication protocol with Labview part******
#*******************************************************

try:
    bCont=bConn
    while bCont:
        msgtype_s=s.recv(1)
        check_socket(msgtype_s)
        msgtype = int.from_bytes(msgtype_s,'big',signed=False)
        #print (msgtype)

        #*****Handshake message******
        if msgtype==MSG_HANDSHAKE:
            s.send(TCP_RESPONSE)
        if msgtype==MSG_TERM:
            bCont=False

        #******Image message******
        elif msgtype==MSG_IMG:
#image message:
# # bytes content
# 1 1     type:1 (MSG_IMG in python, SendImage in LV)
# 2 16    timestamp (long unsigned: yymmddHHMMSSsss)
# 3 4     image size L 
# 4 L^2   U8 image


#image response
# # bytes content
# 1 16     timestamp, as received
# 2 1      probability of eye
# 3 1      probability of eyeblink 

#Communication:receive
            #receive timestamp
            if log_img:
                logging.info('img receieve started')
            ts_s=s.recv(8)
            check_socket(ts_s)
            ts=struct.unpack('q',ts_s)
            if log_img:
                logging.info('    timestamp received:{}'.format(ts))
                
            #receive size of the image N 
            imgsize_s=s.recv(4)
            check_socket(imgsize_s)
            
            imgsize = int.from_bytes(imgsize_s,'big',signed=False)
            if log_img:
                logging.info('    imgsize received: {}'.format(imgsize))
                
            #Calculate number of bytes to receive
            npixels=imgsize**2
            nbytes=npixels#int(ceil(npixels/8))
            
            #receieve image
            imgbytes_s=s.recv(nbytes)
            check_socket(imgbytes_s)
            if log_img:
                logging.info('    image received')
            img=array([int(b) for b in imgbytes_s]).reshape((imgsize,imgsize)).astype(uint8)

            if log_img:
                logging.info('    imgage size: {}'.format(shape(img)))
            savez('Debug.npz',Img=img)
            
            img2=cv.resize(img,(128,128))
            if log_img:
                logging.info('    new imgage size: {}'.format(shape(img2)))
            networkInput = img2.astype(np.float32) / 255.0  # convert from uint8 [0, 255] to float32 [0, 1]

            networkInput = networkInput[None, :, :, None]  # add batch and channel dimensions
            
            mask, info = model(networkInput)
            
            eyeProbability = info[0,0]
            
            blinkProbability = info[0,1]
            blinkprob=int(floor(blinkProbability*100))
            eyeprob=int(floor(eyeProbability*100))
            #prob=9
            if log_img:
                logging.info('    probability calculated: {:.3f},{:.3f}'.format(eyeProbability,blinkProbability))
#Communication: send response
            blinkprob_s=struct.pack('b',blinkprob)
            eyeprob_s=struct.pack('b',eyeprob)
            s.send(ts_s)
            s.send(eyeprob_s) 
            s.send(blinkprob_s) 
            if log_img:
                logging.info('    probability sent to lv')
except:
    logging.info('End of the program2')
    s.close()
finally:
    try:
        s.close()
    except:
        pass
    logging.info('End of the program')