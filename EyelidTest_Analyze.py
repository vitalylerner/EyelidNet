from numpy import *
from matplotlib.pyplot import *
from scipy.signal import savgol_filter

mov_dir='D:/Inbar_Eyeblink/EyesTrain/OriginalMat/'

for imov in [7,8]:
    result_npz=mov_dir+'movie{}_analysis.npz'.format(imov)
    with load(result_npz) as D:
        EPC=D['EPC']
        A=D['A']

    EPC=EPC[EPC>0]

    EPC2 = savgol_filter(EPC, 5, 3) # window size 51, polynomial order 3
    dt=0.04#s
    t=arange(len(EPC),dtype=float)*dt
    figure()
    plot(t,EPC,'.',color=[0.1,0.1,0.1,0.1])
    plot(t,EPC2,'-',color='k',linewidth=2)
    title('{}'.format(imov))
show()