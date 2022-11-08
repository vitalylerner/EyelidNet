#***************************************#
#    EyelidNet                          #
# Vitay Lerner 2022                     #
# Common structures and functions       #
# this file does not run by itself      #
#***************************************#

from numpy import *
from scipy.interpolate import interp1d
class EyelidNet_Error(RuntimeError):
     def __init__(self,message):
        super().__init__("\n***********\n EyelidNet: \n"+message+"\n***********")

def EyelidNet_updatestatus(source:str,message:str):
    print ('*** Eyelid Net  ***')
    print (source)
    print(message)
    print ('********************')

def equipoints(a):
    #converts
    #p13 p12 p11 p10 p23 p22 p21 p20 x0 x1 
    #to 8
    #equally distanced 8 points
    #on the eyelids
    
    xst=a[8]
    xend=a[9]
    
    p1=poly1d(a[:4])
    p2=poly1d(a[4:8])
    
    x1=linspace(xst,xend,40)
    x2=linspace(xst,xend,40)
    y1=p1(x1)
    y2=p2(x2)
    
    alpha = linspace(0, 1, 3)
    
    distance = cumsum(sqrt( ediff1d(x1, to_begin=0)**2 + ediff1d(y1, to_begin=0)**2 ))
    distance = distance/distance[-1]
    fx1, fy1 = interp1d( distance, x1 ), interp1d( distance, y1 )
    x_reg1, y_reg1 = fx1(alpha), fy1(alpha)
    
    distance = cumsum(sqrt( ediff1d(x2, to_begin=0)**2 + ediff1d(y2, to_begin=0)**2 ))
    distance = distance/distance[-1]
    fx2, fy2 = interp1d( distance, x2 ), interp1d( distance, y2 )
    x_reg2, y_reg2 = fx2(alpha), fy2(alpha)
    
    #x_reg2,y_reg2=x_reg2[3:0:-1],y_reg2[3:0:-1]
    x_reg=array([x_reg1[0],x_reg1[1],x_reg1[2],x_reg2[1]],dtype=float)
    y_reg=array([y_reg1[0],y_reg1[1],y_reg1[2],y_reg2[1]],dtype=float)

    return hstack([x_reg,y_reg])
    
EN_NFeatures=8
#**********Labeling errors***************#
EN_NoManualPointsErr    = EyelidNet_Error("Curently there is no result created with manual clicking")
EN_ImgNOutOfRangeErr    = EyelidNet_Error('Image Number out of range')
EN_RecordError          = EyelidNet_Error('Points of image not found')   

#**********Training errors***************#    

EN_TrSetNotFoundErr    = EyelidNet_Error("Training Set Folder  Not Found")
EN_TrSetVerNotFoundErr = EyelidNet_Error("Training Set Version Not Found")
EN_TrSetScaNotFoundErr = EyelidNet_Error("Training Set Scaling Not Found")

EN_WrongMode      = EyelidNet_Error("Wrong Mode: 0 for Train, 1 for Use")
EN_NN_NotFoundErr      = EyelidNet_Error("Neural Network file Not Found")

EN_IMG_NotSquare    = EyelidNet_Error("Image is not square")