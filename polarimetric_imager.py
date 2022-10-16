import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from IPython.display import Image
import glob
import math
from pystackreg import StackReg
def isolateblue(image):
    r,g,b=cv2.split(image)
    return b
def rgb_preview(imagelist):
	return cv2.merge((imagelist[2],imagelist[1],imagelist[0])).astype('uint8')
def hsv_processing(imagelist):
	i0 =imagelist[0]/1
	i45 =imagelist[1]/1
	i90 =imagelist[1]/1
	stokesI = i0 + i90
	stokesQ = i0 - i90
	stokesU = (2.0 * i45)- stokesI
	polint = np.sqrt(stokesQ*stokesQ+stokesU*stokesU)
	poldolp = polint/(stokesI+((np.ones(stokesI.shape)+0.001)))
	polaop = 0.5 * np.arctan(stokesU, stokesQ)
	h=(polaop+(np.ones(polaop.shape)*(np.pi/2.0)))/np.pi
	s=poldolp*200
	s[s<0]=0
	s[s>255]=255
	v=polint
	hsvpolar=cv2.merge((h,s,v))
	rgbimg = cv2.cvtColor(hsvpolar.astype('uint8'),cv2.COLOR_HSV2RGB)*2
	rgbimg[rgbimg<0]=0
	rgbimg[rgbimg>255]=255
	return rgbimg
if __name__ == '__main__':
	imagefiles=glob.glob(r"#whatever your filepath is")
	imagefiles.sort()
	images=[]
	for filename in imagefiles:
	  img=cv2.imread(filename)
	  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	  img=np.int16(img)
	  images.append(img)
	polchannels=[]
	for image in images:
	    polchannels.append(isolateblue(image))
	num_images=len(polchannels)
	sr=StackReg(StackReg.AFFINE)
	polchannels=sr.register_transform_stack(np.stack((polchannels[0],polchannels[1],polchannels[2])), reference='first')
	cv2.imwrite("rgb preview.jpg",rgb_preview(polchannels))
	cv2.imwrite("polarimetric image.jpg",hsv_processing(polchannels))
	print("done")
