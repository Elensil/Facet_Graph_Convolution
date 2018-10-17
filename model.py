from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
from utils import *
#import h5py

random_seed=0
std_dev=0.05

# Levi-Civita tensor of dimension 3
LC_tensor = tf.constant([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]],dtype=tf.float32)

Id_tensor = tf.constant([0,0,1,0,1,0,1,0,0],shape=[3,3], dtype=tf.float32)

ref_axis = tf.constant([0,0,1],dtype=tf.float32)

def broadcast(tensor, shape):
	return tensor + tf.zeros(shape, dtype=tensor.dtype)


def weight_variable(shape):
		initial = tf.random_normal(shape, stddev=std_dev)
		#initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
		return tf.Variable(initial, name="weight")

def bias_variable(shape):
		initial = tf.random_normal(shape, stddev=std_dev)
		#initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
		return tf.Variable(initial, name="bias")

def assignment_variable(shape):
		initial = tf.random_normal(shape, stddev=std_dev)
		#initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
		return tf.Variable(initial, name="assignment")

def reusable_weight_variable(shape, name="weight"):
		initial = tf.random_normal_initializer(stddev=std_dev)
		#initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
		return tf.get_variable(name, shape=shape, initializer=initial)

def reusable_bias_variable(shape, name="bias"):
		initial = tf.random_normal_initializer(stddev=std_dev)
		#initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
		return tf.get_variable(name, shape=shape, initializer=initial)

def reusable_assignment_variable(shape, name="assignment"):
		initial = tf.random_normal_initializer(stddev=std_dev)
		#initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
		return tf.get_variable(name, shape=shape, initializer=initial)


def tile_repeat(n, repTime):
		'''
		create something like 111..122..2333..33 ..... n..nn 
		one particular number appears repTime consecutively.
		This is for flattening the indices.
		'''
		idx = tf.range(n)
		idx = tf.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
		idx = tf.tile(idx, [1, repTime])  # Create multiple columns, each column has one number repeats repTime 
		y = tf.reshape(idx, [-1])
		return y

def get_weight_assigments(x, adj, u, v, c):
		batch_size, in_channels, num_points = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()
		# [batch_size, M, N]
		ux = tf.map_fn(lambda x: tf.matmul(u, x), x)
		vx = tf.map_fn(lambda x: tf.matmul(v, x), x)
		# [batch_size, N, M]
		vx = tf.transpose(vx, [0, 2, 1])
		# [batch_size, N, K, M]
		patches = get_patches(vx, adj)
		# [K, batch_size, M, N]
		patches = tf.transpose(patches, [2, 0, 3, 1])
		# [K, batch_size, M, N]
		patches = tf.add(ux, patches)
		# [K, batch_size, N, M]
		patches = tf.transpose(patches, [0, 1, 3, 2])
		patches = tf.add(patches, c)
		# [batch_size, N, K, M]
		patches = tf.transpose(patches, [1, 2, 0, 3])
		patches = tf.nn.softmax(patches)
		return patches

def get_weight_assigments_translation_invariance(x, adj, u, c):
		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()
		# [batch, N, K, ch]
		patches = get_patches(x, adj)
		# [batch, N, ch, 1]
		x = tf.reshape(x, [batch_size, -1, in_channels, 1])
		# [batch, N, ch, K]
		patches = tf.transpose(patches, [0, 1, 3, 2])
		# [batch, N, ch, K]
		patches = tf.subtract(x, patches)
		# [batch, ch, N, K]
		patches = tf.transpose(patches, [0, 2, 1, 3])
		# [batch, ch, N*K]
		x_patches = tf.reshape(patches, [batch_size, in_channels, -1])
		# batch, M, N*K
		patches = tf.map_fn(lambda x: tf.matmul(u, x) , x_patches)
		# batch, M, N, K
		patches = tf.reshape(patches, [batch_size, M, -1, K])
		# [batch, K, N, M]
		patches = tf.transpose(patches, [0, 3, 2, 1])
		# [batch, K, N, M]
		patches = tf.add(patches, c)
		# batch, N, K, M
		patches = tf.transpose(patches, [0, 2, 1, 3])
		patches = tf.nn.softmax(patches)
		return patches

# def get_weight_assignments_partly_translation_invariance(x, adj, u, v, c):
		# 		batch_size, in_channels, num_points = x.get_shape().as_list()
		# 		batch_size, num_points, K = adj.get_shape().as_list()
		# 		M, in_channels = u.get_shape().as_list()
		# 		# [batch_size, M, N]
		# 		ux = tf.map_fn(lambda x: tf.matmul(u, x), x)
		# 		vx = tf.map_fn(lambda x: tf.matmul(v, x), x)
		# 		# [batch_size, N, M]
		# 		vx = tf.transpose(vx, [0, 2, 1])
		# 		# [batch_size, N, K, M]
		# 		patches = get_patches(vx, adj)
		# 		# [K, batch_size, M, N]
		# 		patches = tf.transpose(patches, [2, 0, 3, 1])
		# 		# [K, batch_size, M, N]
		# 		patches = tf.add(ux, patches)
		# 		# [K, batch_size, N, M]
		# 		patches = tf.transpose(patches, [0, 1, 3, 2])
		# 		patches = tf.add(patches, c)
		# 		# [batch_size, N, K, M]
		# 		patches = tf.transpose(patches, [1, 2, 0, 3])
		# 		patches = tf.nn.softmax(patches)
		# 		return patches

def getRotationToAxis(x):

		batch_size, num_points, in_channels = x.get_shape().as_list()
		xn = tf.slice(x,[0,0,0],[-1,-1,3])

		ref_axis_t = tf.reshape(ref_axis,[1,1,3])
		# [batch, N, 3]
		#ref_axes = tf.tile(ref_axes,[batch_size,num_points,1])
		#ref_axes = broadcast(ref_axes,x.shape)

		ref_axes = tf.zeros_like(xn)
		ref_axes = ref_axes + ref_axis_t

		# [batch, N, 3]
		ref_cross = tf.cross(xn,ref_axes)
		# [batch, N, 1]
		ref_sin = tf.norm(ref_cross)
		# [batch, N, 1]
		ref_cos = tensorDotProduct(ref_axis,xn)

		# [batch, N, 3, 1]
		ref_cross = tf.expand_dims(ref_cross,-1)
		# [batch, N, 3, 3, 1]
		ref_cross = tf.tile(tf.expand_dims(ref_cross,2),[1,1,3,1,1])
		# [1, 1, 3, 3, 3]
		LC = tf.reshape(LC_tensor, [1,1,3,3,3])
		# [batch, N, 3, 3, 1]
		temp_zero = tf.zeros_like(ref_cross)
		# [batch, N, 3, 3, 3]
		temp_zero = tf.tile(temp_zero,[1,1,1,1,3])


		# [batch, N, 3, 3, 3]
		LC = LC + temp_zero
		#LC = tf.tile(LC,[batch_size,num_points,1,1,1])

		# [batch, N, 3, 3, 1]
		ssm = tf.matmul(LC,ref_cross)
		# [batch, N, 3, 3]
		ssm = tf.squeeze(ssm)

		# [batch, N, 1]
		rot_coef = tf.divide(tf.subtract(1.0,ref_cos), tf.multiply(ref_sin,ref_sin))
		# [batch, N, 3, 3]
		rot_coef = tf.tile(tf.reshape(rot_coef,[batch_size,-1,1,1]),[1,1,3,3])
		# [1, 1, 3, 3]
		Idmat = tf.reshape(Id_tensor,[1,1,3,3])
		# [batch, N, 3, 3]
		Idmat = Idmat + tf.zeros_like(rot_coef)
		#Idmat = tf.tile(Idmat,[batch_size,num_points,1,1])


		# [batch, N, 3, 3]
		rot = Idmat + ssm + tf.multiply(tf.matmul(ssm,ssm), rot_coef)

		return rot
		# rot gives a (3,3) rotation matrix for every face

def get_weight_assigments_rotation_invariance(x, adj, u, c):

	# The trick here is to compute a rotation from any face normal to a fixed axis (say, (0,0,1))
	# We follow the procedure described here: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d#476311
	# Using the Levi-Civita tensor, as explained here https://math.stackexchange.com/questions/258775/from-a-vector-to-a-skew-symmetric-matrix

		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()

					# ref_axis_t = tf.reshape(ref_axis,[1,1,3])
					# print("ref_axis_t shape: "+str(ref_axis_t.shape))
					# # [batch, N, 3]
					# print("batch_size = "+str(batch_size))
					# print("num_points = "+str(num_points))
					# #ref_axes = tf.tile(ref_axes,[batch_size,num_points,1])
					# #ref_axes = broadcast(ref_axes,x.shape)

					# ref_axes = tf.zeros_like(x)
					# ref_axes = ref_axes + ref_axis_t

					# print("ref_axes shape: "+str(ref_axes.shape))
					# # [batch, N, 3]
					# ref_cross = tf.cross(x,ref_axes)
					# # [batch, N, 1]
					# ref_sin = tf.norm(ref_cross)
					# # [batch, N, 1]
					# ref_cos = tensorDotProduct(ref_axis,x)

					# # [batch, N, 3, 1]
					# ref_cross = tf.expand_dims(ref_cross,-1)
					# # [batch, N, 3, 3, 1]
					# ref_cross = tf.tile(tf.expand_dims(ref_cross,2),[1,1,3,1,1])
					# # [1, 1, 3, 3, 3]
					# LC = tf.reshape(LC_tensor, [1,1,3,3,3])
					# # [batch, N, 3, 3, 1]
					# temp_zero = tf.zeros_like(ref_cross)
					# # [batch, N, 3, 3, 3]
					# temp_zero = tf.tile(temp_zero,[1,1,1,1,3])


					# # [batch, N, 3, 3, 3]
					# LC = LC + temp_zero
					# #LC = tf.tile(LC,[batch_size,num_points,1,1,1])

					# # [batch, N, 3, 3, 1]
					# ssm = tf.matmul(LC,ref_cross)
					# # [batch, N, 3, 3]
					# ssm = tf.squeeze(ssm)

					# # [batch, N, 1]
					# rot_coef = tf.divide(tf.subtract(1.0,ref_cos), tf.multiply(ref_sin,ref_sin))
					# # [batch, N, 3, 3]
					# rot_coef = tf.tile(tf.reshape(rot_coef,[batch_size,-1,1,1]),[1,1,3,3])
					# # [1, 1, 3, 3]
					# Idmat = tf.reshape(Id_tensor,[1,1,3,3])
					# # [batch, N, 3, 3]
					# Idmat = Idmat + tf.zeros_like(rot_coef)
					# #Idmat = tf.tile(Idmat,[batch_size,num_points,1,1])


					# # [batch, N, 3, 3]
					# rot = Idmat + ssm + tf.multiply(tf.matmul(ssm,ssm), rot_coef)
					# # [batch, N, K, 3, 3]

		rot = getRotationToAxis(x)

		rot = tf.tile(tf.expand_dims(rot,axis=2),[1,1,K,1,1])
		# rot gives a (3,3) rotation matrix for every face



		# [batch, N, K, ch]
	
		patches = get_patches(x, adj)
		# [batch, N, K, ch, 1]
		patches = tf.expand_dims(patches, -1)
		# [batch, N, K, ch, 1]
		patches = tf.matmul(rot,patches)
		# [batch, N, K, ch]
		patches = tf.reshape(patches,[batch_size,-1,K,3])
		# [batch, ch, N, K]
		patches = tf.transpose(patches, [0, 3, 1, 2])
							# # [batch, N, ch, 1]
							# x = tf.reshape(x, [-1, num_points, in_channels, 1])
							# # [batch, N, ch, K]
							# patches = tf.transpose(patches, [0, 1, 3, 2])
							# # [batch, N, ch, K]
							# patches = tf.subtract(x, patches)
							# # [batch, ch, N, K]
							# patches = tf.transpose(patches, [0, 2, 1, 3])
		# [batch, ch, N*K]
		x_patches = tf.reshape(patches, [batch_size, in_channels, -1])
		# batch, M, N*K
		patches = tf.map_fn(lambda x: tf.matmul(u, x) , x_patches)
		# batch, M, N, K
		patches = tf.reshape(patches, [batch_size, M, -1, K])
		# [batch, K, N, M]
		patches = tf.transpose(patches, [0, 3, 2, 1])
		# [batch, K, N, M]
		patches = tf.add(patches, c)
		# batch, N, K, M
		patches = tf.transpose(patches, [0, 2, 1, 3])
		patches = tf.nn.softmax(patches)
		return patches



def get_weight_assigments_rotation_invariance_with_area(x, adj, u, c):

	# The trick here is to compute a rotation from any face normal to a fixed axis (say, (0,0,1))
	# We follow the procedure described here: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d#476311
	# Using the Levi-Civita tensor, as explained here https://math.stackexchange.com/questions/258775/from-a-vector-to-a-skew-symmetric-matrix

		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()

		xa = tf.slice(x,[0,0,3],[-1,-1,-1])
		
		xn = tf.slice(x,[0,0,0],[-1,-1,3])



		ref_axis_t = tf.reshape(ref_axis,[1,1,3])
		print("ref_axis_t shape: "+str(ref_axis_t.shape))
		# [batch, N, 3]
		print("batch_size = "+str(batch_size))
		print("num_points = "+str(num_points))
		#ref_axes = tf.tile(ref_axes,[batch_size,num_points,1])
		#ref_axes = broadcast(ref_axes,x.shape)

		ref_axes = tf.zeros_like(xn)
		ref_axes = ref_axes + ref_axis_t

		print("ref_axes shape: "+str(ref_axes.shape))
		# [batch, N, 3]
		ref_cross = tf.cross(xn,ref_axes)
		# [batch, N, 1]
		ref_sin = tf.norm(ref_cross)
		# [batch, N, 1]
		ref_cos = tensorDotProduct(ref_axis,xn)

		# [batch, N, 3, 1]
		ref_cross = tf.expand_dims(ref_cross,-1)
		# [batch, N, 3, 3, 1]
		ref_cross = tf.tile(tf.expand_dims(ref_cross,2),[1,1,3,1,1])
		# [1, 1, 3, 3, 3]
		LC = tf.reshape(LC_tensor, [1,1,3,3,3])
		# [batch, N, 3, 3, 1]
		temp_zero = tf.zeros_like(ref_cross)
		# [batch, N, 3, 3, 3]
		temp_zero = tf.tile(temp_zero,[1,1,1,1,3])


		# [batch, N, 3, 3, 3]
		LC = LC + temp_zero
		#LC = tf.tile(LC,[batch_size,num_points,1,1,1])

		# [batch, N, 3, 3, 1]
		ssm = tf.matmul(LC,ref_cross)
		# [batch, N, 3, 3]
		ssm = tf.squeeze(ssm)

		# [batch, N, 1]
		rot_coef = tf.divide(tf.subtract(1.0,ref_cos), tf.multiply(ref_sin,ref_sin))
		# [batch, N, 3, 3]
		rot_coef = tf.tile(tf.reshape(rot_coef,[batch_size,-1,1,1]),[1,1,3,3])
		# [1, 1, 3, 3]
		Idmat = tf.reshape(Id_tensor,[1,1,3,3])
		# [batch, N, 3, 3]
		Idmat = Idmat + tf.zeros_like(rot_coef)
		#Idmat = tf.tile(Idmat,[batch_size,num_points,1,1])


		# [batch, N, 3, 3]
		rot = Idmat + ssm + tf.multiply(tf.matmul(ssm,ssm), rot_coef)
		# [batch, N, K, 3, 3]
		rot = tf.tile(tf.expand_dims(rot,axis=2),[1,1,K,1,1])
		# rot gives a (3,3) rotation matrix for every face



		# [batch, N, K, ch]
		print("x shape: "+str(x.shape))
		print("adj shape: "+str(adj.shape))
		patches = get_patches(x, adj)
		print("patches shape: "+str(patches.shape))

		# Normals part of patches
		npatches = tf.slice(patches,[0,0,0,0],[-1,-1,-1,3])

		# [batch, N, K, ch, 1]
		npatches = tf.expand_dims(npatches, -1)
		print("npatches shape: "+str(npatches.shape))
		# [batch, N, K, ch, 1]
		npatches = tf.matmul(rot,npatches)
		print("npatches shape: "+str(npatches.shape))
		# [batch, N, K, ch]
		npatches = tf.reshape(npatches,[batch_size,-1,K,3])
		


		# Area part of patches
		# [batch, N, K, 1]
		apatches = tf.slice(patches,[0,0,0,3],[-1,-1,-1,-1])

		apatches = tf.divide(apatches,tf.expand_dims(xa,axis=2))

		patches = tf.concat([npatches,apatches],axis=-1)

		print("patches shape: "+str(patches.shape))
		# [batch, ch, N, K]
		patches = tf.transpose(patches, [0, 3, 1, 2])

		# [batch, ch, N*K]
		x_patches = tf.reshape(patches, [batch_size, in_channels, -1])
		# batch, M, N*K
		patches = tf.map_fn(lambda x: tf.matmul(u, x) , x_patches)
		# batch, M, N, K
		patches = tf.reshape(patches, [batch_size, M, -1, K])
		# [batch, K, N, M]
		patches = tf.transpose(patches, [0, 3, 2, 1])
		# [batch, K, N, M]
		patches = tf.add(patches, c)
		# batch, N, K, M
		patches = tf.transpose(patches, [0, 2, 1, 3])
		patches = tf.nn.softmax(patches)
		return patches


def get_weight_assigments_rotation_invariance_with_position(x, adj, u, c):

	# The trick here is to compute a rotation from any face normal to a fixed axis (say, (0,0,1))
	# We follow the procedure described here: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d#476311
	# Using the Levi-Civita tensor, as explained here https://math.stackexchange.com/questions/258775/from-a-vector-to-a-skew-symmetric-matrix

		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()

		# Position
		xp = tf.slice(x,[0,0,3],[-1,-1,-1])
		# Normal
		xn = tf.slice(x,[0,0,0],[-1,-1,3])

		rot = getRotationToAxis(xn)

							# ref_axis_t = tf.reshape(ref_axis,[1,1,3])
							# print("ref_axis_t shape: "+str(ref_axis_t.shape))
							# # [batch, N, 3]
							# print("batch_size = "+str(batch_size))
							# print("num_points = "+str(num_points))
							# #ref_axes = tf.tile(ref_axes,[batch_size,num_points,1])
							# #ref_axes = broadcast(ref_axes,x.shape)

							# ref_axes = tf.zeros_like(xn)
							# ref_axes = ref_axes + ref_axis_t

							# print("ref_axes shape: "+str(ref_axes.shape))
							# # [batch, N, 3]
							# ref_cross = tf.cross(xn,ref_axes)
							# # [batch, N, 1]
							# ref_sin = tf.norm(ref_cross)
							# # [batch, N, 1]
							# ref_cos = tensorDotProduct(ref_axis,xn)

							# # [batch, N, 3, 1]
							# ref_cross = tf.expand_dims(ref_cross,-1)
							# # [batch, N, 3, 3, 1]
							# ref_cross = tf.tile(tf.expand_dims(ref_cross,2),[1,1,3,1,1])
							# # [1, 1, 3, 3, 3]
							# LC = tf.reshape(LC_tensor, [1,1,3,3,3])
							# # [batch, N, 3, 3, 1]
							# temp_zero = tf.zeros_like(ref_cross)
							# # [batch, N, 3, 3, 3]
							# temp_zero = tf.tile(temp_zero,[1,1,1,1,3])


							# # [batch, N, 3, 3, 3]
							# LC = LC + temp_zero
							# #LC = tf.tile(LC,[batch_size,num_points,1,1,1])

							# # [batch, N, 3, 3, 1]
							# ssm = tf.matmul(LC,ref_cross)
							# # [batch, N, 3, 3]
							# ssm = tf.squeeze(ssm)

							# # [batch, N, 1]
							# rot_coef = tf.divide(tf.subtract(1.0,ref_cos), tf.multiply(ref_sin,ref_sin))
							# # [batch, N, 3, 3]
							# rot_coef = tf.tile(tf.reshape(rot_coef,[batch_size,-1,1,1]),[1,1,3,3])
							# # [1, 1, 3, 3]
							# Idmat = tf.reshape(Id_tensor,[1,1,3,3])
							# # [batch, N, 3, 3]
							# Idmat = Idmat + tf.zeros_like(rot_coef)
							# #Idmat = tf.tile(Idmat,[batch_size,num_points,1,1])


							# # [batch, N, 3, 3]
							# rot = Idmat + ssm + tf.multiply(tf.matmul(ssm,ssm), rot_coef)
		# [batch, N, K, 3, 3]
		rot = tf.tile(tf.expand_dims(rot,axis=2),[1,1,K,1,1])
		# rot gives a (3,3) rotation matrix for every face


		# [batch, N, K, ch]
		patches = get_patches(x, adj)

		# Normals part of patches
		npatches = tf.slice(patches,[0,0,0,0],[-1,-1,-1,3])

		# [batch, N, K, ch, 1]
		npatches = tf.expand_dims(npatches, -1)
		# [batch, N, K, ch, 1]
		npatches = tf.matmul(rot,npatches)
		# [batch, N, K, ch]
		npatches = tf.reshape(npatches,[batch_size,-1,K,3])
		


		# Position part of patches
		# [batch, N, K, 3]
		ppatches = tf.slice(patches,[0,0,0,3],[-1,-1,-1,-1])

		# Compute displacement to current face
		ppatches = tf.subtract(ppatches,tf.expand_dims(xp,axis=2))

		# Rotate, just like the normals
		ppatches = tf.expand_dims(ppatches, -1)
		ppatches = tf.matmul(rot,ppatches)
		ppatches = tf.reshape(ppatches,[batch_size,-1,K,3])

		patches = tf.concat([npatches,ppatches],axis=-1)

		# [batch, ch, N, K]
		patches = tf.transpose(patches, [0, 3, 1, 2])

		# [batch, ch, N*K]
		x_patches = tf.reshape(patches, [batch_size, in_channels, -1])
		# batch, M, N*K
		patches = tf.map_fn(lambda x: tf.matmul(u, x) , x_patches)
		# batch, M, N, K
		patches = tf.reshape(patches, [batch_size, M, -1, K])
		# [batch, K, N, M]
		patches = tf.transpose(patches, [0, 3, 2, 1])
		# [batch, K, N, M]
		patches = tf.add(patches, c)
		# batch, N, K, M
		patches = tf.transpose(patches, [0, 2, 1, 3])
		patches = tf.nn.softmax(patches)
		return patches


def get_slices(x, adj):		#adj is one-indexed
		batch_size, num_points, in_channels = x.get_shape().as_list()
		#num_points = x.shape[1]
		batch_size, input_size, K = adj.get_shape().as_list()
		zeros = tf.zeros([batch_size, 1, in_channels], dtype=tf.float32)
		x = tf.concat([zeros, x], 1)
		#x = tf.reshape(x, [batch_size*(num_points+1), in_channels])
		x = tf.reshape(x, [-1, in_channels])
		#adj = tf.reshape(adj, [batch_size*num_points*K])
		#adj = tf.reshape(adj, [-1])
		# adj_flat = tile_repeat(batch_size, num_points*K)
		# adj_flat = adj_flat*in_channels
		# adj_flat = adj_flat + adj
		#adj_flat = tf.reshape(adj_flat, [batch_size*num_points, K])
		adj_flat = tf.reshape(adj, [-1, K])
		slices = tf.gather(x, adj_flat)
		#slices = tf.reshape(slices, [batch_size, num_points, K, in_channels])
		slices = tf.reshape(slices, [batch_size, -1, K, in_channels])
		return slices

def get_patches(x, adj):
		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		patches = get_slices(x, adj)
		return patches

def custom_conv2d(x, adj, out_channels, M, translation_invariance=False, rotation_invariance=False):
		
		batch_size, input_size, in_channels = x.get_shape().as_list()
		W0 = weight_variable([M, out_channels, in_channels])
		b = bias_variable([out_channels])
		u = assignment_variable([M, in_channels])
		c = assignment_variable([M])
		batch_size, input_size, K = adj.get_shape().as_list()
		# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
		adj_size = tf.count_nonzero(adj, 2)
		#deal with unconnected points: replace NaN with 0
		non_zeros = tf.not_equal(adj_size, 0)
		adj_size = tf.cast(adj_size, tf.float32)
		adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
		# [batch_size, input_size, 1, 1]
		#adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
		adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])
		

		if (translation_invariance == False) and (rotation_invariance == False):
			v = assignment_variable([M, in_channels])
		elif translation_invariance == True:
			print("Translation-invariant\n")
			# [batch_size, input_size, K, M]
			q = get_weight_assigments_translation_invariance(x, adj, u, c)
		elif rotation_invariance == True:
			print("Rotation-invariant\n")
			# [batch_size, input_size, K, M]
			if in_channels==3:
				q = get_weight_assigments_rotation_invariance(x, adj, u, c)
			elif in_channels==4:
				q = get_weight_assigments_rotation_invariance_with_area(x, adj, u, c)
			elif in_channels==6:
				q = get_weight_assigments_rotation_invariance_with_position(x, adj, u, c)

		# [batch_size, in_channels, input_size]
		x = tf.transpose(x, [0, 2, 1])
		W = tf.reshape(W0, [M*out_channels, in_channels])
		# Multiple w and x -> [batch_size, M*out_channels, input_size]
		wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
		# Reshape and transpose wx into [batch_size, input_size, M*out_channels]
		wx = tf.transpose(wx, [0, 2, 1])
		# Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
		patches = get_patches(wx, adj)
		# [batch_size, input_size, K, M]

		if (translation_invariance == False) and (rotation_invariance == False):
			q = get_weight_assigments(x, adj, u, v, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
		else:
			#q = get_weight_assigments_translation_invariance(x, adj, u, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
			pass

		#patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
		patches = tf.reshape(patches, [batch_size, -1, K, M, out_channels])
		# [out, batch_size, input_size, K, M]
		patches = tf.transpose(patches, [4, 0, 1, 2, 3])
		patches = tf.multiply(q, patches)
		patches = tf.transpose(patches, [1, 2, 3, 4, 0])
		# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
		patches = tf.reduce_sum(patches, axis=2)
		patches = tf.multiply(adj_size, patches)
		# Add add elements for all m
		patches = tf.reduce_sum(patches, axis=2)
		# [batch_size, input_size, out]
		patches = patches + b
		return patches, W0


def getRotInvPatches(x, adj):

	# The trick here is to compute a rotation from any face normal to a fixed axis (say, (0,0,1))
	# We follow the procedure described here: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d#476311
	# Using the Levi-Civita tensor, as explained here https://math.stackexchange.com/questions/258775/from-a-vector-to-a-skew-symmetric-matrix

		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()

		# Position
		xp = tf.slice(x,[0,0,3],[-1,-1,-1])
		# Normal
		xn = tf.slice(x,[0,0,0],[-1,-1,3])

		ref_axis_t = tf.reshape(ref_axis,[1,1,3])
		# [batch, N, 3]

		#ref_axes = tf.tile(ref_axes,[batch_size,num_points,1])
		#ref_axes = broadcast(ref_axes,x.shape)

		ref_axes = tf.zeros_like(xn)
		ref_axes = ref_axes + ref_axis_t

		# [batch, N, 3]
		ref_cross = tf.cross(xn,ref_axes)
		# [batch, N, 1]
		ref_sin = tf.norm(ref_cross)
		# [batch, N, 1]
		ref_cos = tensorDotProduct(ref_axis,xn)

		# [batch, N, 3, 1]
		ref_cross = tf.expand_dims(ref_cross,-1)
		# [batch, N, 3, 3, 1]
		ref_cross = tf.tile(tf.expand_dims(ref_cross,2),[1,1,3,1,1])
		# [1, 1, 3, 3, 3]
		LC = tf.reshape(LC_tensor, [1,1,3,3,3])
		# [batch, N, 3, 3, 1]
		temp_zero = tf.zeros_like(ref_cross)
		# [batch, N, 3, 3, 3]
		temp_zero = tf.tile(temp_zero,[1,1,1,1,3])

		# [batch, N, 3, 3, 3]
		LC = LC + temp_zero
		#LC = tf.tile(LC,[batch_size,num_points,1,1,1])

		# [batch, N, 3, 3, 1]
		ssm = tf.matmul(LC,ref_cross)
		# [batch, N, 3, 3]
		ssm = tf.squeeze(ssm)

		# [batch, N, 1]
		rot_coef = tf.divide(tf.subtract(1.0,ref_cos), tf.multiply(ref_sin,ref_sin))
		# [batch, N, 3, 3]
		rot_coef = tf.tile(tf.reshape(rot_coef,[batch_size,-1,1,1]),[1,1,3,3])
		# [1, 1, 3, 3]
		Idmat = tf.reshape(Id_tensor,[1,1,3,3])
		# [batch, N, 3, 3]
		Idmat = Idmat + tf.zeros_like(rot_coef)
		#Idmat = tf.tile(Idmat,[batch_size,num_points,1,1])


		# [batch, N, 3, 3]
		rot = Idmat + ssm + tf.multiply(tf.matmul(ssm,ssm), rot_coef)
		# [batch, N, K, 3, 3]
		rot = tf.tile(tf.expand_dims(rot,axis=2),[1,1,K,1,1])
		# rot gives a (3,3) rotation matrix for every face



		# [batch, N, K, ch]
		patches = get_patches(x, adj)

		# Normals part of patches
		npatches = tf.slice(patches,[0,0,0,0],[-1,-1,-1,3])

		# [batch, N, K, ch, 1]
		npatches = tf.expand_dims(npatches, -1)
		# [batch, N, K, ch, 1]
		npatches = tf.matmul(rot,npatches)
		# [batch, N, K, ch]
		npatches = tf.reshape(npatches,[batch_size,-1,K,3])
		


		# Position part of patches
		# [batch, N, K, 3]
		ppatches = tf.slice(patches,[0,0,0,3],[-1,-1,-1,-1])

		# Compute displacement to current face
		ppatches = tf.subtract(ppatches,tf.expand_dims(xp,axis=2))

		# Rotate, just like the normals
		ppatches = tf.expand_dims(ppatches, -1)
		ppatches = tf.matmul(rot,ppatches)
		ppatches = tf.reshape(ppatches,[batch_size,-1,K,3])

		# [batch, N, K, ch (6)]
		patches = tf.concat([npatches,ppatches],axis=-1)

		return patches


def custom_conv2d_norm_pos(x, adj, out_channels, M, translation_invariance=False, rotation_invariance=False):
		
		batch_size, input_size, in_channels = x.get_shape().as_list()
		W0 = weight_variable([M, out_channels, in_channels])
		b = bias_variable([out_channels])
		u = assignment_variable([M, in_channels])
		c = assignment_variable([M])
		batch_size, input_size, K = adj.get_shape().as_list()
		# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
		adj_size = tf.count_nonzero(adj, 2)
		#deal with unconnected points: replace NaN with 0
		non_zeros = tf.not_equal(adj_size, 0)
		adj_size = tf.cast(adj_size, tf.float32)
		adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
		# [batch_size, input_size, 1, 1]
		#adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
		adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])
		
		# [batch, N, K, in_ch]
		rotInvPatches = getRotInvPatches(x, adj)

		if (translation_invariance == False) and (rotation_invariance == False):
			v = assignment_variable([M, in_channels])
		elif translation_invariance == True:
			print("Translation-invariant\n")
			# [batch_size, input_size, K, M]
			q = get_weight_assigments_translation_invariance(x, adj, u, c)
		elif rotation_invariance == True:
			print("Rotation-invariant\n")
			# [batch_size, input_size, K, M]
			if in_channels==3:
				q = get_weight_assigments_rotation_invariance(x, adj, u, c)
			elif in_channels==4:
				q = get_weight_assigments_rotation_invariance_with_area(x, adj, u, c)
			elif in_channels==6:
				q = get_weight_assigments_rotation_invariance_with_position(x, adj, u, c)


		# [batch, in_ch, N*K]
		rotInvPatches = tf.reshape(rotInvPatches, [batch_size,in_channels, -1])

		W = tf.reshape(W0, [M*out_channels, in_channels])
		# Multiple w and rotInvPatches -> [batch_size, M*out_channels, N*K]
		wx = tf.map_fn(lambda x: tf.matmul(W, x), rotInvPatches)
		# Reshape and transpose wx into [batch_size, N*K, M*out_channels]
		wx = tf.transpose(wx, [0, 2, 1])

		patches = tf.reshape(wx,[batch_size,-1,K,M*out_channels])
		

		if (translation_invariance == False) and (rotation_invariance == False):
			q = get_weight_assigments(x, adj, u, v, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
		else:
			#q = get_weight_assigments_translation_invariance(x, adj, u, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
			pass

		#patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
		patches = tf.reshape(patches, [batch_size, -1, K, M, out_channels])
		# [out, batch_size, input_size, K, M]
		patches = tf.transpose(patches, [4, 0, 1, 2, 3])
		patches = tf.multiply(q, patches)
		patches = tf.transpose(patches, [1, 2, 3, 4, 0])
		# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
		patches = tf.reduce_sum(patches, axis=2)
		patches = tf.multiply(adj_size, patches)
		# Add add elements for all m
		patches = tf.reduce_sum(patches, axis=2)
		# [batch_size, input_size, out]
		print("Your patches shape is "+str(patches.shape))
		patches = patches + b
		return patches, W0


def custom_conv2d_pos_for_assignment(x, adj, out_channels, M, translation_invariance=False, rotation_invariance=False):
		
		batch_size, input_size, in_channels_ass = x.get_shape().as_list()
                in_channels_weights = x.get_shape().as_list()[2] - 3
                in_channels_ass = x.get_shape().as_list()[2]
		xn = tf.slice(x,[0,0,0],[-1,-1,in_channels_weights])	# take normals only
		W0 = weight_variable([M, out_channels, in_channels_weights])
		b = bias_variable([out_channels])
		u = assignment_variable([M, in_channels_ass])
		c = assignment_variable([M])
		batch_size, input_size, K = adj.get_shape().as_list()
		# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
		adj_size = tf.count_nonzero(adj, 2)
		#deal with unconnected points: replace NaN with 0
		non_zeros = tf.not_equal(adj_size, 0)
		adj_size = tf.cast(adj_size, tf.float32)
		adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
		# [batch_size, input_size, 1, 1]
		#adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
		adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])
		

				# if (translation_invariance == False) and (rotation_invariance == False):
				# 	v = assignment_variable([M, in_channels_ass])
				# elif translation_invariance == True:
				# 	print("Translation-invariant\n")
				# 	# [batch_size, input_size, K, M]
				# 	q = get_weight_assigments_translation_invariance(x, adj, u, c)
				# elif rotation_invariance == True:
				# 	print("Rotation-invariant\n")
				# 	# [batch_size, input_size, K, M]
				# 	if in_channels==3:
				# 		q = get_weight_assigments_rotation_invariance(x, adj, u, c)
				# 	elif in_channels==4:
				# 		q = get_weight_assigments_rotation_invariance_with_area(x, adj, u, c)
				# 	elif in_channels==6:
				# 		q = get_weight_assigments_rotation_invariance_with_position(x, adj, u, c)


		# Make new assignement, that is translation invariant wrt position, but not normals
		vn = assignment_variable([M, in_channels_ass-in_channels_weights])
		up = tf.slice(u,[0,3],[-1,-1])
		vp = -up
		v = tf.concat([vn,vp],axis=-1)


		# [batch_size, in_channels, input_size]
		x = tf.transpose(x, [0, 2, 1])
		xn = tf.transpose(xn, [0, 2, 1])
		W = tf.reshape(W0, [M*out_channels, in_channels_weights])
		# Multiple w and x -> [batch_size, M*out_channels, input_size]
		wx = tf.map_fn(lambda x: tf.matmul(W, x), xn)
		# Reshape and transpose wx into [batch_size, input_size, M*out_channels]
		wx = tf.transpose(wx, [0, 2, 1])
		# Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
		patches = get_patches(wx, adj)
		# [batch_size, input_size, K, M]

		q = get_weight_assigments(x, adj, u, v, c)
		# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]

		#patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
		patches = tf.reshape(patches, [batch_size, -1, K, M, out_channels])
		# [out, batch_size, input_size, K, M]
		patches = tf.transpose(patches, [4, 0, 1, 2, 3])
		patches = tf.multiply(q, patches)
		patches = tf.transpose(patches, [1, 2, 3, 4, 0])
		# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
		patches = tf.reduce_sum(patches, axis=2)
		patches = tf.multiply(adj_size, patches)
		# Add add elements for all m
		patches = tf.reduce_sum(patches, axis=2)
		# [batch_size, input_size, out]
		patches = patches + b
		return patches, W0


def decoding_layer(x, adj, W, translation_invariance=False):

		batch_size, input_size, in_channels = x.get_shape().as_list()
		M, out_channels, in_channels = W.get_shape().as_list()

		#W = weight_variable([M, out_channels, in_channels])
		b = bias_variable([out_channels])
		u = assignment_variable([M, in_channels])
		c = assignment_variable([M])
		batch_size, input_size, K = adj.get_shape().as_list()
		# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
		adj_size = tf.count_nonzero(adj, 2)
		#deal with unconnected points: replace NaN with 0
		non_zeros = tf.not_equal(adj_size, 0)
		adj_size = tf.cast(adj_size, tf.float32)
		adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
		# [batch_size, input_size, 1, 1]
		#adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
		adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])
		

		if translation_invariance == False:
			v = assignment_variable([M, in_channels])
		else:
			print("Translation-invariant\n")
			# [batch_size, input_size, K, M]
			q = get_weight_assigments_translation_invariance(x, adj, u, c)

		# [batch_size, in_channels, input_size]
		x = tf.transpose(x, [0, 2, 1])
		W = tf.reshape(W, [M*out_channels, in_channels])
		# Multiple w and x -> [batch_size, M*out_channels, input_size]
		wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
		# Reshape and transpose wx into [batch_size, input_size, M*out_channels]
		wx = tf.transpose(wx, [0, 2, 1])
		# Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
		patches = get_patches(wx, adj)
		# [batch_size, input_size, K, M]

		if translation_invariance == False:
			q = get_weight_assigments(x, adj, u, v, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
		else:
			#q = get_weight_assigments_translation_invariance(x, adj, u, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
			pass

		#patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
		patches = tf.reshape(patches, [batch_size, -1, K, M, out_channels])
		# [out, batch_size, input_size, K, M]
		patches = tf.transpose(patches, [4, 0, 1, 2, 3])
		patches = tf.multiply(q, patches)
		patches = tf.transpose(patches, [1, 2, 3, 4, 0])
		# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
		patches = tf.reduce_sum(patches, axis=2)
		patches = tf.multiply(adj_size, patches)
		# Add add elements for all m
		patches = tf.reduce_sum(patches, axis=2)
		# [batch_size, input_size, out]
		print("Your patches shape is "+str(patches.shape))
		patches = patches + b
		return patches
		

def reusable_custom_conv2d(x, adj, out_channels, M, translation_invariance=False, name="conv2d"):
		
		batch_size, input_size, in_channels = x.get_shape().as_list()
		W = reusable_weight_variable([M, out_channels, in_channels], name=(name+"_weight"))
		b = reusable_bias_variable([out_channels], name=(name+"_bias"))
		u = reusable_assignment_variable([M, in_channels], name=(name+"_u"))
		c = reusable_assignment_variable([M], name=(name+"_c"))
		batch_size, input_size, K = adj.get_shape().as_list()

		# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
		adj_size = tf.count_nonzero(adj, 2)
		#deal with unconnected points: replace NaN with 0
		non_zeros = tf.not_equal(adj_size, 0)
		adj_size = tf.cast(adj_size, tf.float32)
		adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))

		# [batch_size, input_size, 1, 1]
		#adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
		adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])

		if translation_invariance == False:
			v = reusable_assignment_variable([M, in_channels], name=(name+"_v"))
		else:
			print("Translation-invariant\n")
			# [batch_size, input_size, K, M]
			q = get_weight_assigments_translation_invariance(x, adj, u, c)

		# [batch_size, in_channels, input_size]
		x = tf.transpose(x, [0, 2, 1])
		W = tf.reshape(W, [M*out_channels, in_channels])
		# Multiple w and x -> [batch_size, M*out_channels, input_size]
		wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
		# Reshape and transpose wx into [batch_size, input_size, M*out_channels]
		wx = tf.transpose(wx, [0, 2, 1])
		# Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
		patches = get_patches(wx, adj)
		# [batch_size, input_size, K, M]

		if translation_invariance == False:
			q = get_weight_assigments(x, adj, u, v, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
		else:
			#q = get_weight_assigments_translation_invariance(x, adj, u, c)
			# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
			pass
		
		#patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
		patches = tf.reshape(patches, [batch_size, -1, K, M, out_channels])
		# [out, batch_size, input_size, K, M]
		patches = tf.transpose(patches, [4, 0, 1, 2, 3])
		patches = tf.multiply(q, patches)
		patches = tf.transpose(patches, [1, 2, 3, 4, 0])

		# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
		patches = tf.reduce_sum(patches, axis=2)
		patches = tf.multiply(adj_size, patches)
		# Add add elements for all m
		patches = tf.reduce_sum(patches, axis=2)
		# [batch_size, input_size, out]
		print("Your patches shape is "+str(patches.shape))
		patches = patches + b
		return patches


def custom_lin(input, out_channels):
		batch_size, input_size, in_channels = input.get_shape().as_list()

		W = weight_variable([in_channels, out_channels])
		b = bias_variable([out_channels])
		return tf.map_fn(lambda x: tf.matmul(x, W), input) + b

def reusable_custom_lin(input, out_channels,name="lin"):
		batch_size, input_size, in_channels = input.get_shape().as_list()

		W = reusable_weight_variable([in_channels, out_channels], name=(name+"_weight"))
		b = reusable_bias_variable([out_channels], name=(name+"_bias"))
		return tf.map_fn(lambda x: tf.matmul(x, W), input) + b


def custom_max_pool(input, kernel_size, stride=[2, 2], padding='VALID'):
		kernel_h, kernel_w = kernel_size
		stride_h, stride_w = stride
		outputs = tf.nn.max_pool(input, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1], padding=padding)
		return outputs

def custom_binary_tree_pooling(x):

	batch_size, input_size, channels = x.shape
	# Pairs of nodes should already be grouped together
	x = tf.reshape(x,[batch_size,-1,2,channels])
	outputs = tf.reduce_max(x,axis=2)

	return outputs

def custom_upsampling(x):
	batch_size, input_size, channels = x.shape

	x = tf.expand_dims(x,axis=2)
	outputs = tf.tile(x,[1,1,2,1])
	outputs = tf.reshape(outputs,[batch_size,-1,channels])

	return outputs

def get_model(x, adj, num_classes, architecture):
		""" 
		0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
		"""
		if architecture == 0:
				out_channels_fc0 = 16
				h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
				# Conv1
				M_conv1 = 9
				out_channels_conv1 = 32
				h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
				# Conv2
				M_conv2 = 9
				out_channels_conv2 = 64
				h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
				# Conv3
				# M_conv3 = 9
				# out_channels_conv3 = 128
				# h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
				# Lin(1024)
				out_channels_fc1 = 1024
				h_fc1 = tf.nn.relu(custom_lin(h_conv2, out_channels_fc1))
				# Lin(num_classes)
				y_conv = custom_lin(h_fc1, num_classes)
				return y_conv



def get_model_reg(x, adj, architecture, keep_prob):
		""" 
		0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
		"""
		bTransInvariant = False
		bRotInvariant = False
		if architecture == 0:		# Original Nitika's architecture (3 conv layers)
				
			out_channels_fc0 = 16
			h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
			
			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 32
			h_conv1, _ = custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 64
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 128
			h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv3_act = tf.nn.relu(h_conv3)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 1:		# Copy of original architecture, with relus replaced by sigmoids
				
			out_channels_fc0 = 16
			h_fc0 = tf.nn.sigmoid(custom_lin(x, out_channels_fc0))
			
			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 32
			h_conv1 = tf.nn.sigmoid(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1,translation_invariance=bTransInvariant))
			
			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 64
			h_conv2 = tf.nn.sigmoid(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant))
			
			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 128
			h_conv3 = tf.nn.sigmoid(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant))
			
			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.sigmoid(custom_lin(h_conv3, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 2:		# one big linear hidden layer

			out_channels_fc0 = 60000
			h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0),name="h_fc0")
			nan0 = tf.is_nan(h_fc0,name="nan0")

			out_channels_reg = 1
			y_conv = custom_lin(h_fc0, out_channels_reg)
			return y_conv

		if architecture == 3:		# Two smaller linear layers

			out_channels_fc0 = 1800
			h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))

			out_channels_fc1 = 1800
			h_fc1 = tf.nn.relu(custom_lin(h_fc0, out_channels_fc1))

			out_channels_reg = 1
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 4:		#Six small linear layers (+ dropout)

			out_channels_fc0 = 200
			h_fc0 = tf.nn.relu6(custom_lin(x, out_channels_fc0),name="h_fc0")
			nan0 = tf.is_nan(h_fc0,name="nan0")
			# apply DropOut to hidden layer
			drop_out0 = tf.nn.dropout(h_fc0, keep_prob)  # DROP-OUT here

			out_channels_fc1 = 200
			h_fc1 = tf.nn.relu6(custom_lin(drop_out0, out_channels_fc1),name="h_fc1")
			nan1 = tf.is_nan(h_fc1,name="nan1")
			# apply DropOut to hidden layer
			drop_out1 = tf.nn.dropout(h_fc1, keep_prob)  # DROP-OUT here

			out_channels_fc2 = 200
			h_fc2 = tf.nn.relu6(custom_lin(drop_out1, out_channels_fc2),name="h_fc2")
			nan2 = tf.is_nan(h_fc2,name="nan2")
			# apply DropOut to hidden layer
			drop_out2 = tf.nn.dropout(h_fc2, keep_prob)  # DROP-OUT here

			out_channels_fc3 = 400
			h_fc3 = tf.nn.relu6(custom_lin(drop_out2, out_channels_fc3),name="h_fc3")
			nan3 = tf.is_nan(h_fc3,name="nan3")
			# apply DropOut to hidden layer
			drop_out3 = tf.nn.dropout(h_fc3, keep_prob)  # DROP-OUT here

			out_channels_fc4 = 400
			h_fc4 = tf.nn.relu6(custom_lin(drop_out3, out_channels_fc4),name="h_fc4")
			nan4 = tf.is_nan(h_fc4,name="nan4")
			# apply DropOut to hidden layer
			drop_out4 = tf.nn.dropout(h_fc4, keep_prob)  # DROP-OUT here

			out_channels_fc5 = 200
			h_fc5 = tf.nn.relu6(custom_lin(drop_out4, out_channels_fc5),name="h_fc5")
			nan5 = tf.is_nan(h_fc5,name="nan5")
			# apply DropOut to hidden layer
			drop_out5 = tf.nn.dropout(h_fc5, keep_prob)  # DROP-OUT here

			out_channels_fc6 = 200
			h_fc6 = tf.nn.relu6(custom_lin(drop_out5, out_channels_fc6),name="h_fc6")
			nan6 = tf.is_nan(h_fc6,name="nan6")
			# apply DropOut to hidden layer
			drop_out6 = tf.nn.dropout(h_fc6, keep_prob)  # DROP-OUT here

			out_channels_reg = 3
			y_conv = custom_lin(drop_out6, out_channels_reg)
			return y_conv

		if architecture == 5:		# Reusable archi 4 (w/o dropout)

			with tf.variable_scope('fc0'):
				out_channels_fc0 = 200
				h_fc0 = tf.nn.relu6(reusable_custom_lin(x, out_channels_fc0),name="h_fc0")
				nan0 = tf.is_nan(h_fc0,name="nan0")

			with tf.variable_scope('fc1'):
				out_channels_fc1 = 200
				h_fc1 = tf.nn.relu6(reusable_custom_lin(h_fc0, out_channels_fc1),name="h_fc1")
				nan1 = tf.is_nan(h_fc1,name="nan1")

			with tf.variable_scope('fc2'):
				out_channels_fc2 = 200
				h_fc2 = tf.nn.relu6(reusable_custom_lin(h_fc1, out_channels_fc2),name="h_fc2")
				nan2 = tf.is_nan(h_fc2,name="nan2")

			with tf.variable_scope('fc3'):
				out_channels_fc3 = 400
				h_fc3 = tf.nn.relu6(reusable_custom_lin(h_fc2, out_channels_fc3),name="h_fc3")
				nan3 = tf.is_nan(h_fc3,name="nan3")

			with tf.variable_scope('fc4'):
				out_channels_fc4 = 400
				h_fc4 = tf.nn.relu6(reusable_custom_lin(h_fc3, out_channels_fc4),name="h_fc4")
				nan4 = tf.is_nan(h_fc4,name="nan4")

			with tf.variable_scope('fc5'):
				out_channels_fc5 = 200
				h_fc5 = tf.nn.relu6(reusable_custom_lin(h_fc4, out_channels_fc5),name="h_fc5")
				nan5 = tf.is_nan(h_fc5,name="nan5")

			with tf.variable_scope('fc6'):
				out_channels_fc6 = 200
				h_fc6 = tf.nn.relu6(reusable_custom_lin(h_fc5, out_channels_fc6),name="h_fc6")
				nan6 = tf.is_nan(h_fc6,name="nan6")

			with tf.variable_scope('fcfinal'):
				out_channels_reg = 1
				y_conv = reusable_custom_lin(h_fc6, out_channels_reg)
			return y_conv

		if architecture == 6:		# One conv layer, concatenated w/ output of previous layer

			out_channels_fc0 = 16
			h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
			
			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 32
			h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1,translation_invariance=bTransInvariant))
			
			test_layer = tf.concat([h_conv1, h_fc0], axis=2)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(test_layer, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 7:		# Kind of like 6, with one extra conv layer

			out_channels_fc0 = 16
			h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
			
			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 32
			h_conv1, _ = custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1,translation_invariance=bTransInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 40 	#64
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant)
			h_conv2_act = tf.nn.relu(h_conv2)

			test_layer = tf.concat([h_conv2_act, h_conv1_act, h_fc0], axis=2)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(test_layer, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 8:		# Reusable archi 0 for iterative network
			
			with tf.variable_scope('fc0'):
				out_channels_fc0 = 16
				h_fc0 = tf.nn.relu(reusable_custom_lin(x, out_channels_fc0))
			
			# Conv1
			with tf.variable_scope('conv1'):
				M_conv1 = 9
				out_channels_conv1 = 32
				h_conv1 = tf.nn.relu(reusable_custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1,translation_invariance=bTransInvariant))
			
			# Conv2
			with tf.variable_scope('conv2'):
				M_conv2 = 9
				out_channels_conv2 = 64
				h_conv2 = tf.nn.relu(reusable_custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant))
			
			# Conv3
			with tf.variable_scope('conv3'):
				M_conv3 = 9
				out_channels_conv3 = 128
				h_conv3 = tf.nn.relu(reusable_custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant))
			
			# Lin(1024)
			with tf.variable_scope('fc1'):
				out_channels_fc1 = 1024
				h_fc1 = tf.nn.relu(reusable_custom_lin(h_conv3, out_channels_fc1))
			
			# Lin(num_classes)
			with tf.variable_scope('fcfinal'):
				out_channels_reg = 3
				y_conv = reusable_custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 9:		# Auto-encoder test
				
			out_channels_fc0 = 16
			h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
			
			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 32
			h_conv1, conv1_W = custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1,translation_invariance=bTransInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 64
			h_conv2, conv2_W = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 128
			h_conv3, conv3_W = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant)
			h_conv3_act = tf.nn.relu(h_conv3)

			# End encoding
			# ---
			# Start decoding
			conv3_Wp = tf.transpose(conv3_W,[0,2,1])
			d_conv2 = decoding_layer(h_conv3_act, adj, conv3_Wp, translation_invariance=bTransInvariant)
			d_conv2_act = tf.nn.relu(d_conv2)	
			# Same as custom_conv2d layer, expect that the weight matrix is provided
			# Thus, no need to provide the number of output channels and number of filters, they can be deduced from W
			# Assignments and bias are independant (initialized within the function)

			conv2_Wp = tf.transpose(conv2_W,[0,2,1])
			d_conv1 = decoding_layer(d_conv2_act, adj, conv2_Wp, translation_invariance=bTransInvariant)
			d_conv1_act = tf.nn.relu(d_conv1)

			conv1_Wp = tf.transpose(conv1_W,[0,2,1])
			d_conv0 = decoding_layer(d_conv1_act, adj, conv1_Wp, translation_invariance=bTransInvariant)
			d_conv0_act = tf.nn.relu(d_conv0)

			# Keep linear layers ??

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(d_conv0_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 10:		# 3 conv layers, first one is rotation invariant

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 16
			#h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
			h_conv1, _ = custom_conv2d(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 32
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 36
			h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 11:		# 4 conv layers, first one is rotation invariant

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 16
			h_conv1, _ = custom_conv2d(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 32
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 64
			h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			# Conv4
			M_conv4 = 9
			out_channels_conv4 = 128
			h_conv4, _ = custom_conv2d(h_conv3_act, adj, out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv4_act = tf.nn.relu(h_conv4)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(h_conv4_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv


		if architecture == 12:		# 4 conv layers, first one is rotation invariant

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 8
			h_conv1, _ = custom_conv2d(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 16
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 32
			h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			# Conv4
			M_conv4 = 9
			out_channels_conv4 = 48
			h_conv4, _ = custom_conv2d(h_conv3_act, adj, out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv4_act = tf.nn.relu(h_conv4)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(h_conv4_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 13:		# Like 10, with MORE WEIGHTS!

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 16
			h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
			#h_conv1, _ = custom_conv2d(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 32
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 64
			h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 14:		# 3 conv layers, first one is translation invariant for position only. Position is only used for assignment

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 16
			h_conv1, _ = custom_conv2d_pos_for_assignment(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 32
			h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 64
			h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv

		if architecture == 15:		# Same as 14, with concatenation at every layer

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 16
			h_conv1, _ = custom_conv2d_pos_for_assignment(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)

			concat1 = tf.concat([h_conv1_act,x],axis=2)

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 32
			h_conv2, _ = custom_conv2d(concat1, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			concat2 = tf.concat([h_conv2_act,concat1],axis=2)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 64
			h_conv3, _ = custom_conv2d(concat2, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			concat3 = tf.concat([h_conv3_act,concat2],axis=2)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(concat3, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv


# For this function, we give a pyramid of adjacency matrix, from detailed to coarse
# (This is used for the pooling layers)
# Edge weights????

def get_model_reg_multi_scale(x, adjs, architecture, keep_prob):
		""" 
		0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
		"""
		bTransInvariant = False
		bRotInvariant = False
		
		if architecture == 0:		# Multi-scale, like in FeaStNet paper (figure 3)

			# Conv1
			M_conv1 = 9
			out_channels_conv1 = 16
			#h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
			h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
			h_conv1_act = tf.nn.relu(h_conv1)
			# shape [batch, N, out_ch]

			# Pooling 1
			pool1 = tf.custom_binary_tree_pooling(h_conv1_act)	# TODO: deal with fake nodes??

			# Conv2
			M_conv2 = 9
			out_channels_conv2 = 32
			h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv2_act = tf.nn.relu(h_conv2)

			# Pooling 2
			pool2 = custom_binary_tree_pooling(h_conv2_act)

			# Conv3
			M_conv3 = 9
			out_channels_conv3 = 36
			h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			h_conv3_act = tf.nn.relu(h_conv3)

			# --- Central features ---

			#DeConv3
			dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
			dconv3_act = tf.nn.relu(dconv3)

			#Upsampling2
			upsamp2 = custom_upsampling(dconv3_act)

			#DeConv2
			concat2 = tf.concat([upsamp2, h_conv2_act], axis=-1)
			dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
			dconv2_act = tf.nn.relu(dconv2)

			#Upsampling1
			upsamp1 = custom_upsampling(dconv2_act)
			concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

			# Lin(1024)
			out_channels_fc1 = 1024
			h_fc1 = tf.nn.relu(custom_lin(concat1, out_channels_fc1))
			
			# Lin(num_classes)
			out_channels_reg = 3
			y_conv = custom_lin(h_fc1, out_channels_reg)
			return y_conv



def custom_pooling(x,pooltype):

	batch_size, input_size,

def encode_images(images):

    dim1 = images.get_shape().as_list()[1]
    dim2 = images.get_shape().as_list()[2]
    dim3 = images.get_shape().as_list()[3]
    dim4 = images.get_shape().as_list()[4]
    images = tf.reshape(images,[dim1,dim2,dim3,dim4])

    # Conv 1
    out_channels1=16
    wc1_w = 4
    wc1_h = 4
    wc1 = weight_variable([wc1_w,wc1_h,dim4,out_channels1])
    padding='SAME'
    #stride=[]
    conv1 = tf.nn.convolution(images,wc1,padding)
    conv1 = tf.nn.relu(conv1)
    """
    # Conv 2
    out_channels2=32
    wc2_w = 4
    wc2_h = 4
    wc2 = weight_variable([wc2_w,wc2_h,out_channels1,out_channels2])
    padding='SAME'
    #stride=[]
    conv2 = tf.nn.convolution(conv1,wc2,padding)
    conv2 = tf.nn.relu(conv2)

    # Conv 3
    out_channels3=64
    wc3_w = 4
    wc3_h = 4
    wc3 = weight_variable([wc3_w,wc3_h,out_channels2,out_channels3])
    padding='SAME'
    #stride=[]
    conv3 = tf.nn.convolution(conv2,wc3,padding)
    conv3 = tf.nn.relu(conv3)


    # Conv 4: conv 1x1 to go to a 3D vector for each pixel
    out_channels4=3
    wc4_w = 1
    wc4_h = 1
    wc4 = weight_variable([wc4_w,wc4_h,out_channels1,out_channels4])
    padding='SAME'
    #stride=[]
    conv4 = tf.nn.convolution(conv1,wc4,padding)
    conv4 = tf.nn.relu(conv4)
    """
    return conv1


def broadcast_matmul(A, B):
    "Compute A @ B, broadcasting over the first `N-2` ranks"
    with tf.variable_scope("broadcast_matmul"):
        return tf.reduce_sum(A[..., tf.newaxis] * B[..., tf.newaxis, :, :],
                             axis=-2)

def get_appearance_model_reg(x, adj, architecture, keep_prob, images_, calibs_):
    with tf.variable_scope("Image_Encoding"):
        ##Images coords ordering: [width,height] (cols,rows)
        ## First compute image features
        print("Images Input shape : ",images_.get_shape().as_list())
        encoded_images = encode_images(images_)
        print("Encoded Images shape : ",encoded_images.get_shape().as_list())

    with tf.variable_scope("Projection_Operation"):
        ## Recover Barycenters
        print("Data shape: ",x.get_shape().as_list())
        ## X: 3 first coords: normals, 3 lasts: barycenter
        normals , barycenters_originals = tf.split(x,[3,3],2,name='InputSplit_1')
        print("Normals shape:",normals.get_shape().as_list())
        print("Original Barycenters shape:",barycenters_originals.get_shape().as_list())
        ones_shape = tf.ones_like(barycenters_originals)
        ones, _ = tf.split(ones_shape,[1,2],2)
        barycenters = tf.concat([barycenters_originals,ones],axis=2)
        barycenters = tf.reshape(barycenters,[-1,4])
        normals = tf.concat([normals,ones],axis=2)
        normals = tf.reshape(normals,[-1,4])
        print("Barycenters shape:",barycenters.get_shape().as_list())

        ## Then BackProject Triangles barycenters
        #Multiply 3D coordinates with projection matrices
        num_cameras = calibs_.get_shape().as_list()[1]
        print("Projection matrices shape: ",calibs_.get_shape().as_list())

        # Repeat 3D coordinates for num_cameras
        stacked_barycenters = tf.stack([barycenters]*num_cameras,name='barycenter_stack')
        stacked_barycenters = tf.transpose(stacked_barycenters, [1,0,2],name='barycenters_transpose')
        stacked_barycenters = tf.reshape(stacked_barycenters,[-1,num_cameras,4,1])

        print("Stacked Barycenters shape:",stacked_barycenters.get_shape().as_list())

        #coords = tf.matmul(calibs_,stacked_barycenters)
        coords = broadcast_matmul(calibs_,stacked_barycenters)
        coords = tf.reshape(coords,[-1,num_cameras,3])
        print("Coords shape:",coords.get_shape().as_list())
        x_coords,y_coords,z_coords = tf.split(coords,[1,1,1],2)
        print("X shape: ",x_coords.get_shape().as_list()) # cols
        print("Y shape: ",y_coords.get_shape().as_list()) # rows
        print("Z shape: ",z_coords.get_shape().as_list())

        # trick for zero z_coords (removed so nans will appear for diagnostics)
        #non_zeros = tf.not_equal(adj_size, 0)
        #adj_size = tf.cast(adj_size, tf.float32)
        #adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))

        z_coords = tf.reciprocal(z_coords)
        x_coords = tf.multiply(x_coords,z_coords)
        y_coords = tf.multiply(y_coords,z_coords)

        #texture_coords = tf.stack([x_coords,y_coords],axis=2)
        #texture_coords = tf.reshape(texture_coords,[-1,num_cameras,2])
        #print("Texture coords shape: ",texture_coords.get_shape().as_list())

        ## Discard invalid tex_coords
        x_negatives = tf.greater(x_coords,tf.zeros_like(x_coords),name='x_negatives')
        x_overflow = tf.less(x_coords,tf.multiply(float(images_.get_shape().as_list()[2]),tf.ones_like(x_coords)),name='x_overflow')

        y_negatives = tf.greater(y_coords,tf.zeros_like(y_coords),name='y_negatives')
        y_overflow = tf.less(y_coords,tf.multiply(float(images_.get_shape().as_list()[3]),tf.ones_like(x_coords)),name='y_overflow')

        z_negatives = tf.greater(z_coords,tf.zeros_like(z_coords),name='z_negatives')

        mask = tf.logical_and(x_negatives,y_negatives) # X is inside image
        #mask = tf.logical_and(mask,x_overflow) # Y is inside image
        #mask = tf.logical_and(mask,y_overflow) # Y is inside image
        mask = tf.logical_and(mask,z_negatives) #triangle is in front of camera

        print("Mask Shape: ",mask.get_shape().as_list())
        tiled_mask = tf.stack([mask]*encoded_images.get_shape().as_list()[3])
        tiled_mask = tf.transpose(tiled_mask,[1,2,0,3])
        tiled_mask = tf.reshape(tiled_mask,[-1,num_cameras,encoded_images.get_shape().as_list()[3]])
        print("Tiled Mask Shape: ",tiled_mask.get_shape().as_list())

        ## Discard cameras behind triangle:
        #recover z coordinate of (barycenter + normal).
        #if z(barycenter +normal) - z(barycenter) > 0 discard (camera behind) else, accept

        #Multiply 3D coordinates with projection matrices
        num_cameras = calibs_.get_shape().as_list()[1]
        #calibs_ = tf.reshape(calibs_,[num_cameras,3,4,])
        print("Projection matrices shape: ",calibs_.get_shape().as_list())

        # Repeat 3D coordinates for num_cameras
        stacked_normals = tf.stack([normals]*num_cameras)
        stacked_normals = tf.transpose(stacked_normals, [1,0,2])
        stacked_normals = tf.reshape(stacked_normals,[-1,num_cameras,4,1])

        print("Stacked Normals shape:",stacked_normals.get_shape().as_list())
        summed = tf.add(stacked_normals,stacked_barycenters)
        #coords = tf.matmul(calibs_,stacked_barycenters)
        coords = broadcast_matmul(calibs_,summed)
        coords = tf.reshape(coords,[-1,num_cameras,3])
        print("Normal Coords shape:",coords.get_shape().as_list())
        normal_x_coords,normal_y_coords,normal_z_coords = tf.split(coords,[1,1,1],2)

        print("Normal Z shape: ",normal_z_coords.get_shape().as_list())

        # Discard invalid normal_z_coords
        #subtract normal's depths to barycenter's
        subtraction = tf.subtract(normal_z_coords,z_coords)
        visibility_mask = tf.less(subtraction,tf.zeros_like(subtraction),name='visibility_mask')
        #mask = tf.logical_and(mask,visibility_mask) #

        print("Mask Shape: ",mask.get_shape().as_list())
        tiled_mask = tf.stack([mask]*encoded_images.get_shape().as_list()[3])
        tiled_mask = tf.transpose(tiled_mask,[1,2,0,3])
        tiled_mask = tf.reshape(tiled_mask,[-1,num_cameras,encoded_images.get_shape().as_list()[3]],name='tiled_mask')
        print("Tiled Mask Shape: ",tiled_mask.get_shape().as_list())

        ## Aggregate Features
        round_x_coords = tf.round(x_coords)
        round_y_coords = tf.round(y_coords)

        round_x_coords = tf.reshape(round_x_coords,[-1, num_cameras])
        round_y_coords = tf.reshape(round_y_coords,[-1, num_cameras])

        round_x_coords = tf.cast(round_x_coords,dtype=tf.int32)
        round_y_coords = tf.cast(round_y_coords,dtype=tf.int32)

        dim = tf.shape(round_x_coords)[0]
        image_id = tf.constant(range(num_cameras))
        image_id = tf.expand_dims(image_id, axis=0)
        image_id = tf.tile(image_id, [dim,1])
        print("Image id shape: ",image_id.get_shape().as_list())
        indices = tf.stack([image_id,round_x_coords,round_y_coords],axis=2,name='Texture_Coords')
        print("Coords shape: ",indices.get_shape().as_list())
        aggregated_features = tf.gather_nd(encoded_images,indices,name='Aggregated_Features')
        print("Aggregated Features shape: ",aggregated_features.get_shape().as_list())
        aggregated_features = tf.where(tiled_mask,aggregated_features,tf.zeros_like(aggregated_features),name='Projection_Features') #mask features with zeros when camera is discarded
        print("Aggregated Features shape: ",aggregated_features.get_shape().as_list())

        ##Average Features
        summed_features = tf.reduce_sum(aggregated_features,axis=1)
        print("Summed Features shape:",summed_features.get_shape().as_list())
        summed_masks = tf.reduce_sum(tf.cast(tiled_mask,dtype=tf.int32),axis=1)
        print("Summed Masks shape:",summed_masks.get_shape().as_list())
        non_zeros = tf.not_equal(summed_masks, 0)
        summed_masks = tf.cast(summed_masks, tf.float32)
        averaging_weights = tf.where(non_zeros,tf.reciprocal(summed_masks),tf.zeros_like(summed_masks))
        averaged_features = tf.multiply(summed_features,averaging_weights)
        averaged_features = tf.reshape(averaged_features,[1,-1,encoded_images.get_shape().as_list()[3]])
        print("Averaged Features size:",averaged_features.get_shape().as_list())

        ## Concatenate resulting Features to normal+pos input
        #x = tf.concat([averaged_features,x],axis=2)
        x = tf.concat([averaged_features,barycenters_originals],axis=2,name='final_concat')

    print("Input Feature:",x.get_shape().as_list())

    """
    0 - input(3+feature_size) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
    bTransInvariant = False
    bRotInvariant = False

    # 3 conv layers, first one is translation invariant for position only. Position is only used for assignment

    # Conv1
    M_conv1 = 9
    out_channels_conv1 = 16
    h_conv1, _ = custom_conv2d_pos_for_assignment(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
    h_conv1_act = tf.nn.relu(h_conv1)

    # Conv2
    M_conv2 = 9
    out_channels_conv2 = 32
    h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
    h_conv2_act = tf.nn.relu(h_conv2)

    # Conv3
    M_conv3 = 9
    out_channels_conv3 = 64
    h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
    h_conv3_act = tf.nn.relu(h_conv3)

    # Lin(1024)
    out_channels_fc1 = 1024
    h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))

    # Lin(num_classes)
    out_channels_reg = 3
    y_conv = custom_lin(h_fc1, out_channels_reg)
    return y_conv

