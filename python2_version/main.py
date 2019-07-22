from pit import *
import tensorflow as tf
# packages for reading the image
import scipy.ndimage as spimg
from scipy.misc import imsave
import glob
import os
import imageio
import time

def compressed_sensing(Xorig, p=0.1):
	# Xorig = Xorig[:,:,0]
	nx,ny,nchan = Xorig.shape
	print Xorig.shape
	# pywt can change the size -> easy fix by transforming back and forth
	# x = pywt.wavedec2(Xorig, wavelet='db1', level=1)
	# Xorig = pywt.waverec2(x, wavelet='db1')

	# picture size
	shape = [nx,ny]
	n = np.prod(shape)

	# # choice of WT parameters
	# # choose frame
	wlet = 'db12'
	# # wlet = 'coif12'
	# # wlet = 'sym12'
	# # print(pywt.wavelist(kind='discrete'))
	L = 3 
	# amplification of lower level wavelet coefficients
	amp = np.linspace(1,.2,L)
	amp = np.kron(amp, np.ones(3) )
	amp = np.insert(amp,0, 20 ) #prepend -> for approximation coeffcient


	# set relative number of pixels
	# p = 0.1

	# total number of pixels
	m = int(p*n)
	print("Total number of pixels: {}k".format(int(n/1000.)))
	print("Number of remaining pixels: {}k".format( int(m/1000) ))
	print("Sample rate: "+str(p))


	### schedule for DCT and WT iterations
	steps_wt = 14
	steps_dct = 10 #parameter for nr. of steps in fixed basis
	N = 7 #nr. of switching bases: wt and dct

	# threshold values
	# th_dct = 3
	# th_wt = 4
	th_wt  = np.append( np.linspace(25,4,N-1), 4 )
	th_dct = np.append( np.linspace(15,3,N-1), 3 )


	# # generate subsampled picture
	mask = getRandMask(n, m)
	Xsub = np.zeros(Xorig.shape)
	Z = np.zeros(Xorig.shape)
	for i in range(nchan):
		X = Xorig[:,:,i]
		Xsub[:,:,i].flat[mask] = X.flat[mask]
	# print Xsub[:,:,0]
	# print Xorig[:,:,0]


		# set initial guess for the reconstruction
		if 'Xrec' not in locals():
		    Xrec = Xsub[:,:,i]
		    
		Xrec = Xsub[:,:,i]

		# set transformations
		dct = DCT(shape)
		wt = WT(shape, wavelet=wlet,level=L, amplify=amp)

		#with myprofile.Profiler(fname='profile.dat'):
		for j in range(N):
		    thOp = softTO(th_dct[j])
		    Xrec=FISTA(dct, thOp, mask, Xsub[:,:,i], stepsize = .75, n_steps=steps_dct, Xorig=X, X0=Xrec)
		    thOp = softTO(th_wt[j])
		    Xrec=FISTA(wt, thOp, mask, Xsub[:,:,i], stepsize = .75, n_steps=steps_wt, Xorig=X, X0=Xrec)

		Z[:,:,i] = Xrec
	Z = Z.astype('uint8')
	Xsub = Xsub.astype('uint8')
	return [Xsub, Z]


def main():
	start = time.time()
	s = 0.1
	pic_file = "./pics/Escher_Waterfall.jpg"
	Xorig = imageio.imread(pic_file)
	Xorig = np.array(Xorig)

	print "Image shape", Xorig.shape

	IMG_orig = tf.placeholder(tf.uint8)
	sample_rate = tf.placeholder(tf.float32)
	y = tf.py_func(compressed_sensing, [IMG_orig, sample_rate], (tf.uint8, tf.uint8))

	with tf.Session() as session:
		[Xsub, Z] = session.run(y, feed_dict={IMG_orig:Xorig, sample_rate:s})


	plt.figure(figsize=(12,8))
	plt.subplot(1,3,1)
	plt.imshow(Xorig)
	plt.subplot(1,3,2)
	plt.imshow(Xsub)
	plt.subplot(1,3,3)
	plt.imshow(Z)
	plt.savefig('output.jpg')
	end = time.time()

	print "Running Time: {}s".format(end-start)

main()
