from numpy import *
from matplotlib.pyplot import *
from scipy.signal import savgol_filter

mov_dir='D:/Inbar_Eyeblink/EyesTrain/OriginalMat/'

for imov in [1,7,8]:
    figure(figsize=(10,5))
    for NetV in ['1.6','1.7']:
        result_npz=mov_dir+'movie{}_analysis_v'.format(imov)+NetV+'.npz'
        try:
            with load(result_npz) as D:
                EPC=D['EPC']
                A=D['A']

            EPC=EPC[EPC>0]

            EPC2 = savgol_filter(EPC, 3, 2) # window size 51, polynomial order 3
            dt=0.04#s
            t=arange(len(EPC),dtype=float)*dt
            
            #plot(t,EPC,'.')
            plot(t,EPC2,'-',linewidth=2,label=NetV)
            
        except FileNotFoundError:
            print (result_npz)
    legend()
    title('{}'.format(imov))
show()