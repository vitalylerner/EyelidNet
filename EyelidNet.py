#***************************************#
#    EyelidNet                          #
# Vitay Lerner 2022                     #
# Main Class of the EyelidNet           #
# this file does not run by itself      #
#***************************************#


# if there's a problem with some dual library 
# $ conda install nomkl



import cv2 
from scipy import signal
import pandas as pd
from matplotlib.pyplot import *
import os
from numpy import *
from EyelidNet_Common import *
from tensorflow.keras.models import load_model


os.environ['KMP_DUPLICATE_LIB_OK']='True'

	
	
	
class EyelidNet:
    features=[]
    features_scaled=[]
    SCALE_BY_MEAN_STD=101
    SCALE_BY_MIN_MAX=102
    SCALE_BY_NONE=103
    SCALE_PM1=201
    SCALE_01=202
    images=[]
    
    feat_stats=None

    path_tr_set = 'SETS/'
    path_nn     = 'NETS/'

    def __init__(self,*,tr_set_ver:str,net_ver:str):
        self.tr_set_ver=tr_set_ver
        self.net_ver=net_ver

        fName_NN=self.path_nn+'EyelidsNet_v'+self.net_ver
        fName_scaling=self.path_tr_set+'EyelidNet_TrainingSet_v'+tr_set_ver+ '_scaling.csv'
        fName_scaling_meta=self.path_tr_set+'EyelidNet_TrainingSet_v' + tr_set_ver+ '_scaling_meta.npz'
        EyelidNet_updatestatus('Init',\
                  'Loading pre-trained network and scaling\n'+\
                  fName_NN+'\n'+fName_scaling+'\n')
        
        if not (os.path.exists(self.path_nn)):
            raise EN_NN_NotFoundErr
        
        if not (os.path.exists(fName_scaling)):
            raise EN_TrSetScaNotFoundErr
        self.nn_load()

        with load(fName_scaling_meta) as D:
            self.scale_by=D['scale_by']
            self.scale_range=D['scale_range']
            self.feat_stats=pd.read_csv(fName_scaling)
            print (self.feat_stats.head())
            
    
    #***************************************************#
    # Scaling of the output parameters                  #
    #***************************************************#   
    def feat_unscale(self,a_scaled):
        a=a_scaled*1.
        stats_max=array(list(self.feat_stats["max"]))
        stats_min=array(list(self.feat_stats["min"]))
        stats_mean=array(list(self.feat_stats["mean"]))
        stats_std=array(list(self.feat_stats["std"]))
        if self.scale_by==self.SCALE_BY_NONE:
            pass
        elif self.scale_by==self.SCALE_BY_MIN_MAX:
            if self.scale_range==self.SCALE_PM1:
                a+=1.
                a/=2.
            a*=(stats_max-stats_min)
            a+=stats_min
        elif self.scale_by==self.SCALE_BY_MEAN_STD:
            a*=stats_std
            a+=stats_mean
        return a
        

    #***************************************************#
    #        Eye pixel count (EPC)                      #
    #***************************************************#

    def get_eyepixelscount(self,a):
        x0,x1,x2,x3=a[0],a[1],a[2],a[3]
        y0,y1,y2,y3=a[4],a[5],a[6],a[7]
        
        d1=sqrt((x2-x0)**2+(y2-y0)**2)
        d2=sqrt((x3-x1)**2+(y3-y1)**2)
        return 0.95*d2 +0.05*d1
            
    def get_eyelids(self,a):
        if False: 
            x=array(a[:8])
            y=array(a[8:])
            xfitup=x[:5]
            yfitup=y[:5]
            xfitdown=hstack([x[4:],x[0]])
            yfitdown=hstack([y[4:],y[0]])
            pUp=polyfit(xfitup,yfitup,3)
            pDown=polyfit(xfitdown,yfitdown,3)
            pUp1d=poly1d(pUp)
            pDown1d=poly1d(pDown)
            
            fUp=poly1d(a[0:4])
            fDown=poly1d(a[4:8])
            x0=x[0]
            x1=x[4]
            x=arange(x0,x1+1)
            yUp=pUp1d(x)
            yDown=pDown1d(x)
            return x,yUp,yDown
        else:
            x0,x1,x2,x3=a[0],a[1],a[2],a[3]
            y0,y1,y2,y3=a[4],a[5],a[6],a[7]
            
            xline1=[x0,x2]
            yline1=[y0,y2]
            xline2=[x1,x3]
            yline2=[y1,y3]
            
        return xline1,yline1,xline2,yline2
    #***************************************************#
    #        Graphics                                   #
    #***************************************************#    
        
    def show_input_output(self,I,a,newfigure:bool=False,color='k'):
        if color==[]:
            color=self.clr_true
        if newfigure:
            figure()
        imshow(I)
        xUp,yUp,xDown,yDown=self.get_eyelids(a)
        xp=a[:4]
        yp=a[4:]
        
        plot(xp,yp,'o',color=color)
        plot(xUp,yUp,color=color)
        plot(xDown,yDown,color=color)    
        
        epc=self.get_eyepixelscount(a)
        text(xp[0],yp[0],'{:.1f}'.format(epc),color=color)

    #***************************************************#
    #        NN                                         #
    #***************************************************#  

    def nn_load (self):    
        #loads pre-trained model
        fName_Model=self.path_nn+'EyelidsNet_v'+self.net_ver
        EyelidNet_updatestatus('NN','Loading '+fName_Model)
        if os.path.exists(fName_Model):
            self.model = load_model(fName_Model)
            self.model.summary()
        else:
            raise EN_NN_NotFoundErr
            
    def nn_predict(self,img):
        if shape(img)[0]==40:
            img2=expand_dims(img,0)
        elif shape(img)[0]==1:
            img2=img*1
            
        a=squeeze(self.model.predict(img2,verbose=0))
        a=self.feat_unscale(a)
        return a
        

