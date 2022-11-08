#*************************************#
#    EyelidNet
# Vitay Lerner 2022
# Creates and trains the network
#*************************************#


# if there's a problem with some dual library 
# $ conda install nomkl

#*******************************************#
#      Parsing arguments                    #
#*******************************************#
import argparse
parser = argparse.ArgumentParser(description='Trains the Eyelid Network')
parser.add_argument('--mode', help='preview,train,validate',type=str,required=True)
parser.add_argument('--net_ver', help='1.1, 1.2,...',type=str,required=False)
parser.add_argument('--tr_set_vet', help='1, 2',type=str,required=False)
args = parser.parse_args()
Network_V='1.4'
TrainingSet_V='1'
if not args.mode==None:
    program_mode=args.mode
else:
    parser.print_help()
    exit()
try:
    if not args.net_ver==None:
        Network_V=args.net_ver
except:
    pass
    
try:    
    if not args.tr_set_ver==None:
        TrainingSet_V=args.tr_set_ver
except:
    pass

print ('net_ver=',Network_V,'tr_set_ver=',TrainingSet_V)
    
if program_mode in ['preview','train','validate']:
    import cv2 
    import pandas as pd
    from matplotlib.pyplot import *
    import os
    from numpy import *
    from EyelidNet_Common import *
    from EyelidNet import *
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
else:
    parser.print_help()
    exit()
    
if program_mode =='train':
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout
    from tensorflow.keras.models import Sequential,load_model
elif program_mode=='validate':
    from tensorflow.keras.models import load_model
#*******************************************#
#      End Parsing arguments                #
#*******************************************#


	
	
class EyelidNet_Train(EyelidNet):
    #*******************************************#
    #      inherits and trains EylidNet         #
    #*******************************************#
    mode=0
   
    def __init__(self,*,tr_set_ver:str,net_ver:str,mode:int=0):
        self.tr_set_ver=tr_set_ver
        self.net_ver=net_ver
        self.mode=mode
        if mode==0: #new model
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
            
        elif mode==1: #trained model
            super().__init__(tr_set_ver=tr_set_ver,net_ver=net_ver)
        else:
            raise EN_WrongMode
        
    
    #***************************************************#
    # Scaling (normalizing) polynom coefficients        #
    #***************************************************#   

        
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

    def get_tr_set_epc(self):
        nSamples=shape(self.features)[0]
        #print ('AAAA')
        #print (nSamples)
        return [self.get_eyepixelscount(self.features[k,:]) for k in range(nSamples)] 
        
    def sortbyepc(self):
        epc=self.get_tr_set_epc()
        
        ii=argsort(epc)
        self.features=self.features[ii,:]
        self.features_scaled=self.features_scaled[ii,:]
        self.images=self.images[ii,:,:]

    #***************************************************#
    #        Graphics                                   #
    #***************************************************#    
    def nn_peek(self,imglst:list=[]):
        img=self.images
        if len(imglst)==0:
            imglst=list(range(10,25020,5000))
        for imgn in imglst:
        
            I=img[imgn,:,:]
            I2=img[imgn:imgn+1,:,:]
            a_gt=squeeze(self.features[imgn,:])
            a_pred=self.nn_predict(I2)
            
            self.show_input_output(I,a_gt,newfigure=True,color='k')
            self.show_input_output(I,a_pred,newfigure=False,color='r')
            savefig('IMG/{}'.format(imgn))
            close(gcf())
        
        
        
    #***************************************************#
    #        NN                                         #
    #***************************************************#  


    def nn_init(self):
        
        EyelidNet_updatestatus('NN','Initialization ver ')
        
        
        num_filters1 = 64 
        filter_size1 = 2
        pool_size1 = 2
        
        num_filters2 = 32 
        filter_size2 = 2
        pool_size2 = 2
        
        model = Sequential([
          Conv2D(num_filters1, filter_size1, input_shape=(40, 40, 1)),
          MaxPooling2D(pool_size=pool_size1),
          Conv2D(num_filters2, filter_size2, input_shape=(20, 20, 1)),
          MaxPooling2D(pool_size=pool_size2),
          #Dropout(0.5), 
          Flatten(),
          #Dense(32,activation='softmax'),
          Dense(128,activation='softmax'),
          Dense(64,activation='softmax'),
          Dense(32,activation='sigmoid'),
          Dense(EN_NFeatures, activation='sigmoid'),
            ])

        model.compile(
          'adam',
          steps_per_execution=5,
          loss='mean_squared_error',
          metrics=['accuracy'],
        )
        self.model=model
        self.model.summary()
        EyelidNet_updatestatus('NN','Initialization Complete')
    
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

            self.nn_peek()
            self.model.save(fName_Model)
            #self.model.save_weights(self.path_nn+'EyelidsNet_v'+self.net_ver+'.{}.hd5'.format(iEpoch))
        EyelidNet_updatestatus('NN','Training end')

        


   






if program_mode=='preview': #preview , but don't train
    NN=EyelidNet_Train(tr_set_ver=TrainingSet_V,net_ver=Network_V,mode=0)
    NN.scale(EyelidNet_Train.SCALE_BY_MIN_MAX,EyelidNet_Train.SCALE_01)
    NN.sortbyepc()
    EPC=NN.get_tr_set_epc()
    #figure()
    hist(EPC,100)
    imglst=list(range(100,25000,2000))
    img=NN.images
    feat=NN.features
    for imgn in imglst:
            I=img[imgn,:,:]
            a_gt=squeeze(feat[imgn,:])
            NN.show_input_output(I,a_gt,newfigure=True,color='k')
            #savefig('IMG/{}'.format(imgn))
            #close(gcf())
    show()
elif program_mode=='train': #train
    NN=EyelidNet_Train(tr_set_ver=TrainingSet_V,net_ver=Network_V,mode=0)
    NN.scale(EyelidNet_Train.SCALE_BY_MIN_MAX,EyelidNet_Train.SCALE_01)
    NN.sortbyepc()
    
    NN.nn_init()
    NN.nn_train() 
    show()
elif program_mode=='validate':# view results of training
    NN=EyelidNet_Train(tr_set_ver=TrainingSet_V,net_ver=Network_V,mode=1)
    #NN.sortbyepc()
    fSetName='SETS/EyelidNet_TrainingSet_v1.npz'
    with load(fSetName) as D:
        img=D['img']
        params=D['params']
    EPC_GT=[]
    EPC_PRED=[]
    for imgn in  range(0,26000,100):
        #imgn=12
        I=img[imgn,:,:]
        I2=img[imgn:imgn+1,:,:]
        #a_gt=
        a_gt=equipoints(squeeze(params[imgn,:]))
        a_pred=NN.nn_predict(I2)
        epc_gt=NN.get_eyepixelscount(a_gt)
        epc_pred=NN.get_eyepixelscount(a_pred)
        EPC_GT.append(epc_gt)
        EPC_PRED.append(epc_pred)
        NN.show_input_output(I,a_gt,newfigure=True,color='k')
        NN.show_input_output(I,a_pred,newfigure=False,color='w')    
        savefig('IMG/{}.png'.format(imgn))
        close(gcf())
        if imgn%1000==0:
            figure()
            plot(EPC_GT,EPC_PRED,'.')
            savefig('IMG/0_EPC.png')
            close(gcf())
        show()
else:
    parser.print_help()
    #print ('ERROR: use \n$ python EyelidNet_Train.py --mode=preview\n$ python EyelidNet_Train.py --mode=train\n$ python EyelidNet_Train.py --mode=validante\n*******')










