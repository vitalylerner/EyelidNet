from EyelidNet import *
from matplotlib.pyplot import *
import cv2
import time
fname='test01.png'
img=cv2.imread(fname)
img=img[:,:,1]
img=cv2.resize(img,(40,40))

#cv2.imshow(fname,img)
#cv2.waitKey(0)

#graphical view
print(shape(img))
ELN=EyelidNet(net_ver='1.7',tr_set_ver='1')
a=ELN.nn_predict(img)
ELN.show_input_output(img,a,newfigure=True,color='w')


#speed test
t0=time.time()
npoints=500
T=zeros(npoints)
print ('')
for i in range(npoints):
    a=ELN.nn_predict(img)
    epc=ELN.get_eyepixelscount(a)
    if i%10==0:
        print('{}/{}      '.format(i,npoints),end='\r')
    T[i] = time.time() - t0

figure()
plot(T,'.',color=[.5,.5,.5,.5])
it=range(npoints)
f=polyfit(it,T,1)
p=poly1d(f)
plot(it,p(it),'-k')
xlabel('#iteration')
ylabel('runtime(s)')
title('{} ms/iteration'.format(int(f[0]*1000)))
#print ((T[-1]-T[0])/npoints)

show()