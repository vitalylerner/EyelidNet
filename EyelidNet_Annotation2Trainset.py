#*************************************#
#    EyelidNet
# Vitay Lerner 2022
# This file generates sets of images 
# and polynoms from raw images 
# each image is slighlty rotated and 
# translated
#*************************************#

from numpy import *
from matplotlib.pyplot import *
import cv2
import os
import pandas as pd
from scipy import ndimage

from EyelidNet_Common import *

class EyelidNet_Annotation2Dataset():



    D={}
    tmpImg={}
    
    figCount=0
    L=120
    N=40
    
    def get_active_folders(self):
        return list(self.D.keys())
    

    
    def img_read(self,*, fld:str, imgn:int):
        L=int(self.L/2)
        if fld in self.D.keys():
            fname=fld+'RawImages/img{:03d}.png'.format(imgn)
            if (os.path.exists(fname)):
                img=cv2.imread(fname)
                img=squeeze(img[:,:,0])
                d=self.D[fld]
                d=d[d['imgn']==imgn]
                #if several rows, use only the last one
                if d.shape[0]>1:
                    d=d.iloc[-1:]
                    xy=array(d)
                else:
                    xy=array(d)
                    
                
                if len(xy)>0:
                    xy=xy[0]
                    x=array(xy[:8])
                    y=array(xy[8:])
                    xc=mean(x)
                    yc=mean(y)
                    if xc-L<0:
                        L=xc-1
                    if xc+L>shape(img)[0]:
                        L=shape(img)[0]-xc
                    if yc-L<0:
                        L=yc-1
                    if yc+L>shape(img)[0]:
                        L=shape(img)[1]-yc    
                    rect=[xc-L,xc+L,yc-L,yc+L]
                    self.tmpImg={'img':img,'points':xy[1:],'rect':rect,'imgn':imgn,'fld':fld,'L':L*2}
                    
                else:
                    self.tmpImg={}
                    raise EN_RecordError
            else:
                self.tmpImg={}
                raise EN_ImgNOutOfRangeErr
           
        else:
            self.tmpImg={}  
            raise EN_NoManualPointsErr
    def img_reset(self):
        self.img_read(fld=self.tmpImg['fld'],imgn=self.tmpImg['imgn'])
        
    def img_show_crop(self):
        crop=self.img_prepare()
        img=crop['img']
        imgg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        xy=crop['points']
        params=crop['params']
                
        
        x=array(xy[:8])
        y=array(xy[8:])
        
        pUp1d=poly1d(params[:4])       
        pDown1d=poly1d(params[4:8])
        fit_x0=int(params[8])
        fit_x1=int(params[9])
        fit_x=arange(fit_x0,fit_x1+1,1)
        fit_yup=pUp1d(fit_x)
        fit_ydown=pDown1d(fit_x)
        
                
        
        P1=vstack([x,y]).T.astype(int)
        
        U=vstack([fit_x,fit_yup]).T.astype(int)
        D=vstack([fit_x,fit_ydown]).T.astype(int)
        #cv2.drawContours(imgg, [P1], 0, (180,0,0), 1)
        cv2.drawContours(imgg, [U], 0, (200,200,50), 1,lineType=cv2.LINE_AA)
        cv2.drawContours(imgg, [D], 0, (50,200,200), 1,lineType=cv2.LINE_AA)
        imgg=cv2.resize(imgg,(200,200))
        cv2.imshow('{}'.format(self.figCount),imgg)
        self.figCount+=1
            
    def img_show(self,mode:str='Full'):
        #mode='Full','Crop'
        
        if not self.tmpImg=={}:
            if mode=='Full':
                img=self.tmpImg['img']*1
                p=self.tmpImg['points']
                
                rect=self.tmpImg['rect']
                xrect=[rect[0],rect[1],rect[1],rect[0]]
                yrect=[rect[2],rect[2],rect[3],rect[3]]
                C=vstack([xrect,yrect]).astype(int).T
                P=array([(p[i],p[i+8]) for i in range(8)])

                cv2.drawContours(img, [P], 0, (180,0,0), 1)
                cv2.drawContours(img, [C], 0, (250,0,0), 1)
                
                cv2.imshow('{}'.format(self.figCount),img)
                self.figCount+=1
            elif mode=='Crop':
                self.img_show_crop()
        #cv2.waitKey(1)
        
    
    def img_rotate(self, deg:float=0.):
        img=self.tmpImg['img']
        xy=self.tmpImg['points']
        x=array(xy[:8])
        y=array(xy[8:])
        if deg<0:
            deg=360+deg
            
        img2 = ndimage.rotate(img, deg,reshape=False)
        N=shape(img)[0]
        M=shape(img)[1]
        
        x0=N*1./2
        y0=M*1./2
        th=-deg*pi/180
        
        dx=x-x0
        dy=y-y0
        R=array([[cos(th),-sin(th)],[sin(th),cos(th)]]).astype(float)
        DXY=vstack([dx,dy])
        DXY1=matmul(R,DXY)
        dx=DXY1[0,:].squeeze()
        dy=DXY1[1,:].squeeze()
        x=trunc(x0+dx)
        y=trunc(y0+dy)

        xy=array(list(x)+list(y)).astype(int)

        self.tmpImg['img']=img2
        self.tmpImg['points']=xy
        
    def img_translate(self, dx:int=0,dy:int=0):
        rect=array(self.tmpImg['rect'])
        rect=rect+[dx,dx,dy,dy]
        self.tmpImg['rect']=list(rect.astype(int))

    
    def img_addnoise(self,amp:int=5):
        img=self.tmpImg['img'].astype(int)
        X=(random.rand(shape(img)[0],shape(img)[1])*amp*2-amp).astype(int)
        img=img+X
        img[img>255]=255
        img[img<0]=0
        self.tmpImg['img']=img.astype(uint8)
    

    def img_prepare(self):
        
        img=self.tmpImg['img'].astype(uint8)
        rect=array(self.tmpImg['rect']).astype(int)
        
        L=self.tmpImg['L']
        xy=self.tmpImg['points']
        x=array(xy[:8])
        y=array(xy[8:])

        #crop
        bBorders=False
        if rect[0]<0:
            rect[0]=0
            bBorders=True
        if rect[1]>shape(img)[1]:
            rect[1]=shape(img)[1]
            bBorders=True
        if rect[2]<0:
            rect[2]=0
            bBorders=True
        if rect[3]>shape(img)[0]:
            rect[3]=shape(img)[0]
            bBorders=True
        if bBorders:
            L=min(rect[3]-rect[2],rect[1]-rect[0])
        img=img[rect[2]:rect[3],rect[0]:rect[1]]

        x=x-rect[0]
        y=y-rect[2]

        #fit the points to polynom and scale
        scaling_f=L*1./self.N
        
        #print (shape(img),L,self.N,scaling_f)

            
        
        xfitup=x[:5]
        yfitup=y[:5]
        xfitdown=hstack([x[4:],x[0]])
        yfitdown=hstack([y[4:],y[0]])
        POLY_D=3
        pUp=polyfit(xfitup,yfitup,POLY_D)
        pDown=polyfit(xfitdown,yfitdown,POLY_D)
        for iPD in range(POLY_D):
            pUp[iPD]  *= (scaling_f**(POLY_D-iPD))
            pDown[iPD]*= (scaling_f**(POLY_D-iPD))
        pUp= pUp/scaling_f
        pDown=pDown/scaling_f
        params=list(pUp)+list(pDown)+[x[0]*1./scaling_f]+[x[4]*1./scaling_f]
        #params=[x[0]*1./scaling_f,x[4]*1./scaling_f,y[0]*1./scaling_f,y[4]*1./scaling_f]
        #scale image to NxN
        img=cv2.resize(img,(self.N,self.N))
        scaling_f=self.L*1./self.N
        x=(x*1./scaling_f).astype(int)
        y=(y*1./scaling_f).astype(int)

        xy=array(list(x)+list(y)).astype(int)
            

        return {'img':img,'points':xy,'params':params}
        
    def img_fit_eyelids(self):
        #convert points to polynom of 3d order
        
        #'Up_p3','Up_p2','Up_p1','Up_p0','Down_p3','Down_p2','Down_p1','Down_p0','xstart','xend'
        xy=self.tmpImg['points']
        x=array(xy[:8])
        y=array(xy[8:])
        xfitup=x[:5]
        yfitup=y[:5]
        xfitdown=hstack([x[4:],x[0]])
        yfitdown=hstack([y[4:],y[0]])
        pUp=polyfit(xfitup,yfitup,3)
        pDown=polyfit(xfitdown,yfitdown,3)
        pUp1d=poly1d(pUp)
        pDown1d=poly1d(pDown)
        params=list(pUp)+list(pDown)+[x[0]]+[x[4]]
        
        #params=[x[0],x[4],y[0],y[4]]
        return params
    
    def img_generate_set(self,*,fld:str,imgn:int,ranges:dict):
        self.img_read(fld=fld,imgn=imgn)
        DEG     = ranges['deg']
        XSHIFT  = ranges['dx']
        YSHIFT  = ranges['dy']
        
        Nimages=len(DEG)*len(XSHIFT)*len(YSHIFT)
        N=self.N
        noise   = ranges['noise']
        IMG=zeros( [Nimages,N,N],dtype=uint8)
        PARAMS=zeros((Nimages,EN_NFeatures),dtype=float)
        k=0
        for ideg,deg in enumerate(DEG):
            for ix,dx in enumerate(XSHIFT):
                for iy,dy in enumerate(YSHIFT):
                    self.img_reset()
                    self.img_rotate(deg)
                    self.img_addnoise(noise)
                    self.img_translate(dx,dy)                 
                    
                    crop=self.img_prepare()
                    PARAMS[k,:]=crop['params']
                    IMG[k,:,:]=crop['img']
                    #cv2.imshow('100{}'.format(k),crop['img'])
                    
                    k+=1
        return IMG, PARAMS
                    
    def generate_set(self,*,fld:str,ranges:dict):
        print ('**************')
        print ('EyelidNet')
        print ('Generation of training set begins')

        print ('**************')
        FLD=self.get_active_folders()
        nvariations=len(ranges['deg'])*len(ranges['dx'])*len(ranges['dy'])

        colnames= ['imgn','varn','Up_p3','Up_p2','Up_p1','Up_p0','Down_p3','Down_p2','Down_p1','Down_p0','xstart','xend']
        IMG_AGG_ALL=[]
        PARAMS_AGG_ALL=[]
        print ('')
        for fld in FLD:
            IMGN=self.folder_imglist(fld=fld)
            IMG_AGG=zeros( (len(IMGN)*nvariations,self.N,self.N),dtype=uint8)
            PARAMS_AGG=zeros( (len(IMGN)*nvariations,10))
            for imgn in IMGN:
                
                print (fld,imgn,'          ',end='\r')
                IMG, PARAMS=self.img_generate_set(fld=fld,imgn=imgn,ranges=ranges)
                #print (PARAMS)
                IMG_AGG[imgn*nvariations:(imgn+1)*nvariations,:,:]=IMG
                
                PARAMS_AGG[imgn*nvariations:(imgn+1)*nvariations,:]=PARAMS
            IMG_AGG_ALL.append(IMG_AGG)
            PARAMS_AGG_ALL.append(PARAMS_AGG)
        IMG_AGG_ALL=vstack(IMG_AGG_ALL)
        PARAMS_AGG_ALL=vstack(PARAMS_AGG_ALL)
        return IMG_AGG_ALL,PARAMS_AGG_ALL
        
        
    def __init__(self,*,N:int=50):
        lstFld=[f for f in os.listdir('RawImages/') if os.path.isdir(f) ]
        
        lstFld=[f for f in lstFld if f[0]=='P']
        fRaw='RawPoints/'
        if len(lstFld)>0:
            for f in lstFld:
                fRawName=fRaw+f+'.csv'
                if os.path.exists(fRawName):
                    self.D[f]=pd.read_csv(fRawName)
        print ('**************')
        print ('EyelidNet')
        print ('Active Folders')
        print (self.get_active_folders())
        print ('**************')
    def folder_imglist(self,*,fld:str):
        if fld in self.get_active_folders():
            d=self.D[fld]
            return list(set(list(d['imgn'])))
        else:
            return []
            
    def folder_validate(self,*,fld:str):
        if fld in self.get_active_folders():
            d=self.D[fld]
            for imgn in self.folder_imglist(fld=fld):
               self.img_read(fld=fld,imgn=imgn) 
               self.img_show('Crop')
        else:
            raise EN_RecordError
            

            
EN_M2D=EyelidNet_Annotation2Dataset(N=40)

#EN_M2D.folder_validate(fld='P3')
 

rng={'deg': [-5,-2,0,2,5],'dx':[-7,0,7],'dy':[-7,0,7],'noise':8}
img,params=EN_M2D.generate_set(ranges=rng,fld='P1')
savez_compressed('SETS/EyelidNet_TrainingSet_v1.npz',img=img,params=params)
#EN_M2D.img_generate_set(fld='P3',imgn=15,ranges=rng)

cv2.waitKey(0)

