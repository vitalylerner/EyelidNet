from EyelidNet import *
import scipy.io as sio

mode='Movie'

 
#Network_V='1.3'
#TrainingSet_V='1'
#ELN=EyelidNet(tr_set_ver=TrainingSet_V,net_ver=Network_V)

mov_dir='D:/Inbar_Eyeblink/EyesTrain/OriginalMat/'
fig_cnt=0
MOV_RECT={1:[100,200,100,200],7:[50,170,20,140],8:[20,120,10,110]}
imov=8

def mov_convert(imov:int):
    mov_fname_mat=mov_dir+'vid{}.mat'.format(imov)
    mov_fname_npz=mov_dir+'vid{}.npz'.format(imov)
    D=sio.loadmat(mov_fname_mat)
    M=squeeze(D['mov'])
    savez_compressed(mov_fname_npz,M=M)


def mov_loadnpz():
    global imov
    mov_fname_npz=mov_dir+'vid{}.npz'.format(imov)
    with load(mov_fname_npz) as D:
        M=D['M']
    return M

def mov_loadcroppednpz():
    global imov
    mov_fname_npz=mov_dir+'vid{}_cropped.npz'.format(imov)
    with load(mov_fname_npz) as D:
        M=D['IMG']
    return M
    
def mov_preview():
    global fig_cnt,IMG,imov
    img0=IMG[:,:,0]
    if imov in MOV_RECT.keys():
        rct=MOV_RECT[imov]
        img0=img0[rct[0]:rct[1],rct[2]:rct[3]]
        img0=cv2.resize(img0,(40,40))
    cv2.imshow('{}'.format(fig_cnt),img0)
    fig_cnt+=1

def mov2imgseq():
    global IMG,IMG_CROP,MOV_RECT,fig_cnt,imov
    rct=MOV_RECT[imov]

    
    nimages=shape(IMG)[2]
    IMG_CROP=zeros((nimages,40,40))
    for imgn in range(nimages):
        img=squeeze(IMG[rct[0]:rct[1],rct[2]:rct[3],imgn])
        IMG_CROP[imgn,:,:]=cv2.resize(img,(40,40))
    
def nn_load():
    global ELN
    Network_V='1.3'
    TrainingSet_V='1'
    ELN=EyelidNet(tr_set_ver=TrainingSet_V,net_ver=Network_V)
start_from_npz=False
start_from_cropped=True
start_nn=True
show_analysis=False
if start_from_npz:
    IMG_CROP=[]    
    IMG=mov_loadnpz()
    crop_npz=mov_dir+'vid{}_cropped.npz'.format(imov)
    mov2imgseq()
    savez_compressed(crop_npz,IMG=IMG_CROP)
if start_from_cropped:
    IMG_CROP=mov_loadcroppednpz()
if start_nn:   
    nn_load()

    nimages=shape(IMG_CROP)[0]

    EPC=zeros(nimages)
    A=zeros((nimages,EN_NFeatures))
    print ('')
    result_npz=mov_dir+'movie{}_analysis.npz'.format(imov)


    for imgn in range(nimages):
        #print (nimages,imgn)
        img=IMG_CROP[imgn:imgn+1,:,:]
        a=ELN.nn_predict(img)
        epc=ELN.get_eyepixelscount(a)
        EPC[imgn]=epc
        A[imgn,:]=a
        if imgn%20==0:
            print('{}/{}     '.format(imgn,nimages),end='\r')
            savez(result_npz,EPC=EPC,A=A)
    #mov_preview()
    savez(result_npz,EPC=EPC,A=A)    
    #cv2.waitKey(0)    
if show_analysis:
    result_npz=mov_dir+'movie{}_analysis.npz'.format(imov)
    with load(result_npz) as D:
        EPC=D['EPC']
        A=D['A']
    plot(EPC)
    show()
   
   
"""    
    
if mode=='Generate':   
    Network_V='1.3'
    TrainingSet_V='1'
    ELN=EyelidNet(tr_set_ver=TrainingSet_V,net_ver=Network_V)

        
    fSetName='SETS/EyelidNet_TrainingSet_v1.npz'
    with load(fSetName) as D:
        img=D['img']
        params=D['params']

    EPC_PRED=[]
    EPC_GT=[]
    print ('')
    with load('tmp.npz') as D:
        EPC_PRED=D['PRED']
        EPC_GT=D['GT']
    for imgn in  range(4000,6000,1):
        if imgn%10==0:
            print('{}/{}          '.format(imgn,2000),end='\r')
            savez('tmp.npz',GT=EPC_GT,PRED=EPC_PRED)
        I=img[imgn,:,:]
        I2=img[imgn:imgn+1,:,:]

        a_gt=equipoints(squeeze(params[imgn,:]))
        a_pred=ELN.nn_predict(I2)
        
        epc_gt=ELN.get_eyepixelscount(a_gt)
        epc_pred=epc=ELN.get_eyepixelscount(a_pred)
        EPC_GT.append(epc_gt)
        EPC_PRED.append(epc_pred)
        #f=figure()
        #ELN.show_input_output(I,a_gt,False,['k']*3) 
        #ELN.show_input_output(I,a_pred,False)    
        #title('{},{}'.format(epc_gt,epc_pred))
        #savefig('IMG/{}.png'.format(imgn))
        #close(f)
      
elif mode=='Movie':
    
   
elif mode=='Analyze1':
    with load('tmp.npz') as D:
        EPC_PRED=D['PRED']
        EPC_GT=D['GT']
    plot(EPC_GT,EPC_PRED,'o',color=[0.5,0.5,0.5,0.5])
    plot([0,300],[0,300],'--')
    
show()


"""






