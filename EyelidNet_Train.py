# conda install nomkl
import cv2 
from scipy import signal
import pandas as pd
from matplotlib.pyplot import *
import os
from numpy import *
from EyelidNet_Error import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout
from tensorflow.keras.models import Sequential,load_model
#from tensorflow.keras.model
fNameT='../EyesTrain/test01_img{:06d}.png'

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def getimg(imgn:int):
    fName=fNameT.format(imgn)
    img = cv2.imread(fName)
    return img[y1:y2,x1:x2,1]
	
	
	
class EyelidNet_Train:
#columns:
#1: image number
#2-5 : p3,p2,p1,p0 of the upper eyelid polynom  (y=p3 x^3 +p2 x^2 +p1 x +p0)
#6-9: same for the lower eyelid polynom
#10,11: x-axis bounds of the eyelids
       

    
    features=[]
    features_scaled=[]
    
    images=[]
    
    feat_stats=None
    
    SCALE_BY_MEAN_STD=101
    SCALE_BY_MIN_MAX=102
    SCALE_BY_NONE=103
    SCALE_PM1=201
    SCALE_01=202
    
    scale_by=103
    scale_range=201
    

    mode=0
    path_tr_set = 'SETS/'
    path_nn     = 'NETS/'
    
    clr_true=[[0.1,0.1,0.1,0.5],[0.8,0.8,0.8,0.8],[0.8,0.8,0.8,0.8]]
    clr_pred=[[0.8,0.1,0.2,0.3],[0.8,0.1,0.2,0.8],[0.8,0.1,0.2,0.8]]
    
    def __init__(self,*,tr_set_ver:str,net_ver:str,mode:int=0):
        self.tr_set_ver=tr_set_ver
        self.net_ver=net_ver
        self.mode=mode
        if mode==0:
            if not (os.path.exists(self.path_nn)):
                print ('creating directory')
                os.mkdir(self.path_nn)
            if not (os.path.exists(self.path_tr_set)):
                raise EN_TrSetNotFoundErr
                
            trset_fname=self.path_tr_set+'EyelidNet_TrainingSet_v'+tr_set_ver+'.npz'
            if not (os.path.exists(trset_fname)):
                raise EN_TrSetVerNotFoundErr
                
            with load(trset_fname) as D:
                self.images=D['img']
                poly_features=D['params']
            
            #Transform from polynoms to poins#
            nimages=shape(poly_features)[0]
            self.features=zeros([nimages,EN_NFeatures],dtype=float)

            for ia,a in enumerate(self.features):
                #x,yUp,yDown=self.get_eyelids(a) 
                self.features[ia,:]=equipoints(poly_features[ia,:])
            
            
            #End of transform#   
            
        elif mode==1:
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
            
#self.feat_stats=pd.DataFrame(data={'mean':stat_mean,'std':stat_std,'max':stat_max,'min':stat_min})
#self.feat_stats.to_csv(trset_scale_fname)
#savez(trset_scale_meta_fname,scale_by=self.scale_by,scale_range=self.scale_range)

            with load(fName_scaling_meta) as D:
                self.scale_by=D['scale_by']
                self.scale_range=D['scale_range']
            self.feat_stats=pd.read_csv(fName_scaling)
            print (self.feat_stats.head())
        else:
            raise EN_WrongMode
        
    
#***************************************************#
# Scaling (normalizing) polynom coefficients        #
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
        
    def scale(self,sc_by:int=103,sc_range:int=202):
        nfeat=shape(self.features)[1]
        
        STATS=zeros(nfeat)
        stat_mean=zeros(nfeat)
        stat_std=zeros(nfeat)
        stat_min=zeros(nfeat)
        stat_max=zeros(nfeat)
        self.scale_by=sc_by
        self.scale_range=sc_range
        
        self.features_scaled=self.features*1.
        
        for iFeat in range(nfeat):
            a=self.features_scaled[:,iFeat]
            a_mean=mean(a)
            a_std=std(a)
            a_max=max(a)
            a_min=min(a)
            stat_mean[iFeat]=a_mean
            stat_std[iFeat]=a_std
            stat_min[iFeat]=a_min
            stat_max[iFeat]=a_max
            if self.scale_by==self.SCALE_BY_MIN_MAX:
                a-=a_min
                a/=(a_max-a_min)
                if sc_range==self.SCALE_PM1:
                    a*=2
                    a-=1
            elif self.scale_by==self.SCALE_BY_MEAN_STD:
                a-=a_mean
                a/=a_std
            elif self.scale_by==self.SCALE_BY_NONE:
                pass
            self.features_scaled[:,iFeat]=a 
        trset_scale_fname=self.path_tr_set+'EyelidNet_TrainingSet_v'+self.tr_set_ver+'_scaling.csv'
        trset_scale_meta_fname=self.path_tr_set+'EyelidNet_TrainingSet_v'+self.tr_set_ver+'_scaling_meta.npz'
        
        self.feat_stats=pd.DataFrame(data={'mean':stat_mean,'std':stat_std,'max':stat_max,'min':stat_min})
        self.feat_stats.to_csv(trset_scale_fname,index=False)
        savez(trset_scale_meta_fname,scale_by=self.scale_by,scale_range=self.scale_range)
        #elf.feat_stats.to_csv('EyelidsNet_v1_scaling_meta.csv')



#***************************************************#
#        Eye pixel count (EPC)                      #
#***************************************************#
    def get_eyepixelscount(self,a):
        x,yUp,yDown=self.get_eyelids(a)
        return int(sum(abs(yUp-yDown)))
    def get_tr_set_epc(self):
        nSamples=shape(self.features)[0]
        return [self.get_eyepixelscount(self.features[k,:]) for k in range(nSamples)]  
    def sortbyepc(self):
        epc=self.get_tr_set_epc()
        ii=argsort(epc)
        self.features=self.features[ii]
        self.features_scaled=self.features_scaled[ii]
        
    def get_eyelids(self,a):
    
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
        
#***************************************************#
#        Graphics                                   #
#***************************************************#    
    def nn_peek(self):
        img=self.images
        for imgn in [46,44,97,98,195,196]:
        
            I=img[imgn,:,:]
            I2=img[imgn:imgn+1,:,:]
            a_gt=squeeze(self.features[imgn,:])
            a_pred=self.nn_predict(I2)
            
            self.show_input_output(I,a_gt,True)
            self.show_input_output(I,a_pred,False)
            savefig('IMG/{}'.format(imgn))
            close(gcf())
        #show()
        
    def show_input_output(self,I,a,newfigure:bool=False,color:list=[]):
        if color==[]:
            color=self.clr_true
        if newfigure:
            figure()
        imshow(I)
        x,yUp,yDown=self.get_eyelids(a)
        xp=a[:8]
        yp=a[8:]
        plot(xp,yp,'o',color=color[0])
        plot(x,yUp,color=color[1])
        plot(x,yDown,color=color[2])    
        
        #epc=self.get_eyepixelscount(a)


        #title(epc)
        
#***************************************************#
#        NN                                         #
#***************************************************#  


    def nn_init(self):
        
        EyelidNet_updatestatus('NN','Initialization ver ')
        
        
        num_filters = 128 #32 for ver1, 64 for ver2, 32 for ver3
            
        filter_size = 2
        pool_size = 2
        
        model = Sequential([
          Conv2D(num_filters, filter_size, input_shape=(40, 40, 1)),
          MaxPooling2D(pool_size=pool_size),
          Dropout(0.5), 
          Flatten(),
          #Dense(32,activation='softmax'),
          Dense(EN_NFeatures*2,activation='sigmoid'),
          Dense(EN_NFeatures, activation='sigmoid'),
            ])

        model.compile(
          'adam',
          steps_per_execution=5,
          loss='mean_squared_error',
          metrics=['accuracy'],
        )
        self.model=model
        EyelidNet_updatestatus('NN','Initialization Complete')
        
    def nn_load (self):    
        #loads pre-trained model
        fName_Model=self.path_nn+'EyelidsNet_v'+self.net_ver
        EyelidNet_updatestatus('NN','Loading '+fName_Model)
        if os.path.exists(fName_Model):
            self.model = load_model(fName_Model)
            self.model.summary()
        else:
            raise EN_NN_NotFoundErr
            
    def nn_train(self):
        #trains the model and saves the network architecture and weights
        EyelidNet_updatestatus('NN','Training begin')
        fName_Model=self.path_nn+'EyelidsNet_v'+self.net_ver
        IM=(self.images/255)-0.5

        feat=self.features_scaled
        IM1=expand_dims(IM,axis=3)
        SetL=shape(IM)[0]

        val_split=0.4
        ValL=int(val_split*SetL)
        range_whole=random.permutation(arange(SetL))
        range_val=range_whole[:ValL]
        range_train=range_whole[ValL:]

        train_images=IM1[range_train,:,:,:]
        train_feat=feat[range_train,:]

        val_images=IM1[range_val,:,:,:]
        val_feat=feat[range_val,:]
        for iEpoch in range(50):
            EyelidNet_updatestatus('NN','Meta-Epoch: {}/{}'.format(iEpoch,100))
            self.model.fit(
              train_images,
              train_feat,
              epochs=1,
              validation_data=(val_images, val_feat),
              verbose=1
            )
            
            self.model.summary()
            self.nn_peek()
            self.model.save(fName_Model)
            #self.model.save_weights(self.path_nn+'EyelidsNet_v'+self.net_ver+'.{}.hd5'.format(iEpoch))
        EyelidNet_updatestatus('NN','Training end')
        
    def nn_predict(self,img):
        shape(img)[0]
        shape(img)[1]
        
        model=self.model
        a=squeeze(model.predict([img]))
        #print (a)
        a=self.feat_unscale(a)
        #print (a)
        return a
        


   
Network_V='1.3'
TrainingSet_V='1'
mode=1



NN=EyelidNet_Train(tr_set_ver=TrainingSet_V,net_ver=Network_V,mode=mode)
if mode==0:
    NN.scale(EyelidNet_Train.SCALE_BY_MIN_MAX,EyelidNet_Train.SCALE_01)
    
    #img=NN.images[10,:,:]
    ##a=NN.features[10,:]
    #NN.show_input_output(img,a)
    #NN.sortbyepc()
    #epc=array(NN.get_tr_set_epc(),dtype=int)
    #F=NN.features_scaled
    
    NN.nn_init()
    NN.nn_train()
    
    
elif mode==1:
    #fName=
    fSetName='SETS/EyelidNet_TrainingSet_v1.npz'
    with load(fSetName) as D:
        img=D['img']
        params=D['params']
    for imgn in  range(0,6000,1):
        #imgn=12
        I=img[imgn,:,:]
        I2=img[imgn:imgn+1,:,:]
        #a_gt=
        a_gt=equipoints(squeeze(params[imgn,:]))
        a_pred=NN.nn_predict(I2)
        NN.show_input_output(I,a_gt,True,color=NN.clr_true)
        NN.show_input_output(I,a_pred,False,color=NN.clr_pred)    
        savefig('IMG/{}.png'.format(imgn))
        close(gcf())

        
        
        
    #pass
    #NN.nn_load()
show()









