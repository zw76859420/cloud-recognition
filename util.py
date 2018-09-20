import numpy as np
from scipy.misc import imread,imresize
import progressbar

pr = progressbar.ProgressBar().start()
dir='./train/data/'
label=[]
train=[]
end=4580
size=96
n=0

for out in open('train\\train.csv','r'):
	n+=1
	if n==1:
		continue
	out=out.strip('\n').split(',')
	im=imread(dir+out[0])
	im=imresize(im,size=(size,size))
	im=np.asarray(im,dtype='float64')/255
	nn=None
	if len(im.shape)<3:
		nn=0
		continue
	else:
		nn=im.shape[2]
	if nn<3:
		im=np.concatenate((im,np.zeros((size,size,3-nn))))
	if nn>3:
		im=im[:,:,:3]
	train.append(im)
	label.append([int(out[1])])
	pr.update(int((n/end)*100))
	#if n==1000:
	#	break

pr.finish()
train_np=np.asarray(train)
label_np=np.asarray(label)
print(train_np.shape,label_np.shape,train_np.dtype)
np.save('train',train_np)
np.save('label',label_np)