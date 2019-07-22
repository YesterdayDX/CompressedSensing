import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import imageio as imio
import cvxpy as cvx
import matplotlib.image as mpimg
from pylbfgs import owlqn

def dct2(x):
	return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
	return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, step):
	"""An in-memory evaluation callback."""

	# we want to return two things: 
	# (1) the norm squared of the residuals, sum((Ax-b).^2), and
	# (2) the gradient 2*A'(Ax-b)

	# expand x columns-first
	x2 = x.reshape((_image_dims[0], _image_dims[1])).T

	# Ax is just the inverse 2D dct of x2
	Ax2 = idct2(x2)

	# stack columns and extract samples
	Ax = Ax2.T.flat[_ri_vector].reshape(_b_vector.shape)

	# calculate the residual Ax-b and its 2-norm squared
	Axb = Ax - _b_vector
	fx = np.sum(np.power(Axb, 2))

	# project residual vector (k x 1) onto blank image (ny x nx)
	Axb2 = np.zeros(x2.shape)
	Axb2.T.flat[_ri_vector] = Axb # fill columns-first

	# A'(Ax-b) is just the 2D dct of Axb2
	AtAxb2 = 2 * dct2(Axb2)
	AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

	# copy over the gradient vector
	np.copyto(g, AtAxb)

	return fx

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
	# Can Display the optimization process here
	return 0

def sample(img, sp_rate):
	# Randomly sample an image
	# img: 3-dimension matrix -- nx*nx*chan
	# sp_rate: sample rate
	nx, ny, nchan = img.shape

	mask = np.zeros(img.shape, dtype='uint8')
	B = [0 for i in range(nchan)]

	k = int(round(nx * ny * sp_rate))
	ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

	for j in range(nchan):
	    # extract channel
	    X = img[:,:,j].squeeze()

	    # create images of mask (for visualization)
	    Xm = 255 * np.ones(X.shape)

	    Xm.T.flat[ri] = X.T.flat[ri]
	    mask[:,:,j] = Xm

	    # take random samples of image, store them in a vector b
	    B[j] = X.T.flat[ri].astype(float)
	return ri, mask, B

def recover(mask, B, ri):
	# Recover the original image from mask
	# mask: the sampled image -- nx*ny*nchan
	# B: a nchan*1 vector which contains the sampled image in each channel
	# ri: a vector that indicates the sampled locations

	nx, ny, nchan = mask.shape
	Z = np.zeros(mask.shape)

	for i in range(nchan):
		print("Recovering color channel "+str(i))
		Z[:,:,i] = recover_1_channel(mask[:,:,i], B[i], ri)

	Z = Z.astype('uint8')
	return Z

def recover_1_channel(Xm, b, ri):
	# Recover the image on a single channel
	# Xm: nx*ny matrix, sampled matrix in a single color channel
	# b: a vector that contains that sampled part of X
	# ri: a vector that indicates the sampled locations

	global _b_vector, _ri_vector
	_b_vector = b
	_ri_vector = ri

	ny,nx = Xm.shape
	Z = np.zeros(Xm.shape, dtype='uint8') # The recovered image

	# perform the L1 minimization in memory
	coef = 8 # coefficient before the L1 norm
	Xat2 = owlqn(nx*ny, evaluate, progress, coef)

	# transform the output back into the spatial domain
	Xat = Xat2.reshape(nx, ny).T # stack columns
	Xa = idct2(Xat)
	Z = Xa.astype('uint8')
	return Z

def compressed_sensing(Xorig, sample_rate):
	# Xorig: original image
	mask = np.zeros(Xorig.shape, dtype='uint8')
	ri, mask, B = sample(Xorig, sample_rate)
	Z = recover(mask, B, ri)
	return Z

def main():
	global _image_dims
	# read original image
	Xorig = imio.imread('../1.jpg')
	Xorig = np.array(Xorig)

	print("Image shape", Xorig.shape)

	s = 0.1 # sample rate
	_image_dims = [Xorig.shape[1], Xorig.shape[0]]

	# compressed_sensing(Xorig, s)
	# print(_ri_vector)
	# return


	IMG_orig = tf.placeholder(tf.uint8)
	sample_rate = tf.placeholder(tf.float32)
	y = tf.py_func(compressed_sensing, [IMG_orig, sample_rate], tf.uint8)

	with tf.Session() as session:
		Z = session.run(y, feed_dict={IMG_orig:Xorig, sample_rate:s})

		plt.subplot(1,2,1)
		plt.imshow(Xorig)
		plt.subplot(1,2,2)
		plt.imshow(Z)
		plt.savefig('output_large'+str(s)+'.jpg')


main()



# def my_func(x, sampled_rate):
#     # x will be a numpy array with the contents of the placeholder below
#     print(type(np.sum(x)))
#     print(sampled_rate)
#     return np.sum(x)


