from numpy import *
from matplotlib.pyplot import *
mov_dir='D:/Inbar_Eyeblink/EyesTrain/OriginalMat/'

imov=8
result_npz=mov_dir+'movie{}_analysis.npz'.format(imov)
with load(result_npz) as D:
    EPC=D['EPC']
    A=D['A']
plot(EPC)
show()