from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
from utils import *
#import h5py

random_seed=0
# For meshes
std_dev= 0.05
std_dev_bias = 0.01

# # For images
# std_dev= 0.001
# std_dev_bias = 0.001

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
        initial = tf.random_normal(shape, stddev=std_dev_bias)
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
        _, num_points, K = adj.get_shape().as_list()
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
        #       batch_size, in_channels, num_points = x.get_shape().as_list()
        #       batch_size, num_points, K = adj.get_shape().as_list()
        #       M, in_channels = u.get_shape().as_list()
        #       # [batch_size, M, N]
        #       ux = tf.map_fn(lambda x: tf.matmul(u, x), x)
        #       vx = tf.map_fn(lambda x: tf.matmul(v, x), x)
        #       # [batch_size, N, M]
        #       vx = tf.transpose(vx, [0, 2, 1])
        #       # [batch_size, N, K, M]
        #       patches = get_patches(vx, adj)
        #       # [K, batch_size, M, N]
        #       patches = tf.transpose(patches, [2, 0, 3, 1])
        #       # [K, batch_size, M, N]
        #       patches = tf.add(ux, patches)
        #       # [K, batch_size, N, M]
        #       patches = tf.transpose(patches, [0, 1, 3, 2])
        #       patches = tf.add(patches, c)
        #       # [batch_size, N, K, M]
        #       patches = tf.transpose(patches, [1, 2, 0, 3])
        #       patches = tf.nn.softmax(patches)
        #       return patches

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


def get_slices(x, adj):     #adj is one-indexed
        batch_size, num_points, in_channels = x.get_shape().as_list()
        _, input_size, K = adj.get_shape().as_list()
        zeros = tf.zeros([batch_size, 1, in_channels], dtype=tf.float32)
        x = tf.concat([zeros, x], 1)
        slices = tf.gather(x, adj, axis=1)

        sliceList = []
        for b in range(batch_size):
            xb = x[b,:,:]
            # [N, ch]
            adjb = adj[b,:,:]
            # [N, K]
            sliceb = tf.gather(xb,adjb)
            # [N, K, ch]
            sliceList.append(sliceb)

        slices = tf.stack(sliceList, axis=0)
        # [ batch, N, K, ch]
        return slices

def get_patches(x, adj):
        batch_size, num_points, in_channels = x.get_shape().as_list()
        batch_size, num_points, K = adj.get_shape().as_list()
        patches = get_slices(x, adj)
        return patches


def batch_norm(x, fullNorm=True):
    batch_size, num_points, in_channels = x.get_shape().as_list()
    var_epsilon = 0.000001
    if fullNorm:
        gamma = weight_variable([in_channels])
        beta = weight_variable([in_channels])
        mean, var = tf.nn.moments(x, axes=[0,1])
        # [ch]
        bn_out = tf.nn.batch_normalization(x, mean, var, beta, gamma, var_epsilon)
    else:
        gamma = weight_variable([1, num_points, in_channels])
        beta = weight_variable([1, num_points, in_channels])
        mean, var = tf.nn.moments(x, axes=[0], keep_dims=True)
        # [1, N, ch]
        bn_out = tf.nn.batch_normalization(x, mean, var, beta, gamma, var_epsilon)

    return bn_out


def custom_conv2d(x, adj, out_channels, M, biasMask = True, translation_invariance=False, rotation_invariance=False):
        
        batch_size, input_size, in_channels = x.get_shape().as_list()
        W0 = weight_variable([M, out_channels, in_channels])
        b = bias_variable([out_channels])
        u = assignment_variable([M, in_channels])
        c = assignment_variable([M])
        adj_batch_size, input_size, K = adj.get_shape().as_list()
        # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
        adj_size = tf.count_nonzero(adj, 2)
        #deal with unconnected points: replace NaN with 0
        non_zeros = tf.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size_inv = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
        # [batch_size, input_size, 1, 1]
        #adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
        adj_size_inv = tf.reshape(adj_size_inv, [adj_batch_size, -1, 1, 1])
        

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

        # patches = patches + b
        patches = tf.multiply(adj_size_inv, patches)
        # Add elements for all m
        patches = tf.reduce_sum(patches, axis=2)
        # [batch_size, input_size, out]
        
        if biasMask:
            # NEW!! Add bias only to nodes with at least one active node in neighbourhood
            patches = tf.where(tf.tile(tf.expand_dims(non_zeros,axis=-1),[1,1,out_channels]), patches +b, patches)
        else:
            patches = patches + b

        # return patches, W0
        # return patches, tf.reshape(adj_size, [adj_batch_size,-1,1])
        return patches, [W0, u, c]


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


def image_conv2d(x, out_channels, kernel_size, mask=None, padding='VALID'):

    batch, in_height, in_width, in_channels = x.get_shape().as_list()
    W = weight_variable([kernel_size, kernel_size, in_channels, out_channels])
    b = bias_variable([out_channels])
    y = tf.nn.conv2d(x, W, [1,1,1,1], padding) + b

    if mask is not None:
        if padding=='VALID':
            mask = mask[:,1:-1,1:-1,:]
        mask = tf.tile(mask,[1,1,1,out_channels])
        y = tf.multiply(y, mask)
    return y, W

def image_upsampling(x):
    batch, height, width, channels = x.get_shape().as_list()

    xr = tf.expand_dims(x,axis=2)
    xr = tf.expand_dims(xr, axis=4)
    # [batch, height, 1, width, 1, channels]
    xt = tf.tile(xr,[1,1,2,1,2,1])
    # [batch, height, 2, width, 2, channels]
    xu = tf.reshape(xt,[batch,height*2,width*2,channels])

    return xu

def image_zeropadding(x,padWidth=1):
    batch, height, width, channels = x.get_shape().as_list()

    zeroCol = tf.zeros([batch,height,padWidth,channels])
    x1 = tf.concat([zeroCol,x,zeroCol],axis=2)
    zeroRow = tf.zeros([batch,padWidth,width+2*padWidth,channels])
    x2 = tf.concat([zeroRow,x1,zeroRow], axis=1)

    return x2

def image_paddingbot(x,padWidth=1):
    batch, height, width, channels = x.get_shape().as_list()

    zeroRow = tf.zeros([batch,padWidth,width,channels])
    x1 = tf.concat([x,zeroRow], axis=1)

    return x1

def image_paddingright(x,padWidth=1):
    batch, height, width, channels = x.get_shape().as_list()

    zeroCol = tf.zeros([batch,height,padWidth,channels])
    x1 = tf.concat([x,zeroCol],axis=2)

    return x1

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


def custom_conv2d_pos_for_assignment(x, adj, out_channels, M, biasMask = True, translation_invariance=False, rotation_invariance=False):
        
        batch_size, input_size, in_channels_ass = x.get_shape().as_list()
        in_channels_weights = in_channels_ass - 3
        #in_channels_ass = 6
        xn = tf.slice(x,[0,0,0],[-1,-1,in_channels_weights])    # take normals only
        W0 = weight_variable([M, out_channels, in_channels_weights])
        b = bias_variable([out_channels])
        u = assignment_variable([M, in_channels_ass])
        c = assignment_variable([M])
        adj_batch_size, input_size, K = adj.get_shape().as_list()
        # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
        adj_size = tf.count_nonzero(adj, 2)
        # [ N, K] ?
        #deal with unconnected points: replace NaN with 0
        non_zeros = tf.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size_inv = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
        # [batch_size, input_size, 1, 1]
        #adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
        
        # adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])
        adj_size_inv = tf.reshape(adj_size_inv, [adj_batch_size, -1, 1, 1])
        

        if (translation_invariance == False) and (rotation_invariance == False):
            vn = assignment_variable([M, in_channels_weights])
        elif translation_invariance == True:
            print("Translation-invariant\n")
            un = tf.slice(u,[0,0],[-1,in_channels_weights])
            vn = -un
            # # [batch_size, input_size, K, M]
            # q = get_weight_assigments_translation_invariance(x, adj, u, c)
        # elif rotation_invariance == True:
        #   print("Rotation-invariant\n")
        #   # [batch_size, input_size, K, M]
        #   if in_channels==3:
        #       q = get_weight_assigments_rotation_invariance(x, adj, u, c)
        #   elif in_channels==4:
        #       q = get_weight_assigments_rotation_invariance_with_area(x, adj, u, c)
        #   elif in_channels==6:
        #       q = get_weight_assigments_rotation_invariance_with_position(x, adj, u, c)


        # Make new assignement, that is translation invariant wrt position, but not normals
        # vn = assignment_variable([M, in_channels_weights])
        up = tf.slice(u,[0,in_channels_weights],[-1,-1])
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
        print("\npatches shape = "+str(patches.shape))
        patches = tf.multiply(q, patches)
        print("patches shape = "+str(patches.shape))
        patches = tf.transpose(patches, [1, 2, 3, 4, 0])
        print("patches shape = "+str(patches.shape))
        # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
        patches = tf.reduce_sum(patches, axis=2)
        patches = tf.multiply(adj_size_inv, patches)
        # Add add elements for all m
        patches = tf.reduce_sum(patches, axis=2)
        # [batch_size, input_size, out]
        if biasMask:
            # NEW!! Add bias only to nodes with at least one active node in neighbourhood
            patches = tf.where(tf.tile(tf.expand_dims(non_zeros,axis=-1),[1,1,out_channels]), patches +b, patches)
        else:
            patches = patches + b
        return patches, W0


def custom_conv2d_only_pos_for_assignment(x, adj, out_channels, M, translation_invariance=False, rotation_invariance=False):
        
        batch_size, input_size, in_channels_ass = x.get_shape().as_list()
        in_channels_weights = in_channels_ass - 3
        #in_channels_ass = 6
        xn = tf.slice(x,[0,0,0],[-1,-1,in_channels_weights])    # take normals only
        xp = tf.slice(x,[0,0,in_channels_weights],[-1,-1,-1])
        W0 = weight_variable([M, out_channels, in_channels_weights])
        b = bias_variable([out_channels])
        u = assignment_variable([M, 3])
        c = assignment_variable([M])
        if translation_invariance:
            q = get_weight_assigments_translation_invariance(xp, adj, u, c)
        else:
            v = assignment_variable([M,3])
            q = get_weight_assigments(tf.transpose(xp,[0,2,1],),adj,u,v,c)

        adj_batch_size, input_size, K = adj.get_shape().as_list()
        # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
        adj_size = tf.count_nonzero(adj, 2)
        # [ N, K] ?
        #deal with unconnected points: replace NaN with 0
        non_zeros = tf.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size_inv = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
        # [batch_size, input_size, 1, 1]
        #adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
        
        # adj_size = tf.reshape(adj_size, [batch_size, -1, 1, 1])
        adj_size_inv = tf.reshape(adj_size_inv, [adj_batch_size, -1, 1, 1])

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

        # Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]

        #patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
        patches = tf.reshape(patches, [batch_size, -1, K, M, out_channels])
        # [out, batch_size, input_size, K, M]
        patches = tf.transpose(patches, [4, 0, 1, 2, 3])
        print("\npatches shape = "+str(patches.shape))
        patches = tf.multiply(q, patches)
        print("patches shape = "+str(patches.shape))
        patches = tf.transpose(patches, [1, 2, 3, 4, 0])
        print("patches shape = "+str(patches.shape))
        # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
        patches = tf.reduce_sum(patches, axis=2)
        patches = tf.multiply(adj_size_inv, patches)
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

def custom_binary_tree_pooling(x, steps=1, pooltype='max'):

    batch_size, input_size, channels = x.get_shape().as_list()
    # print("x shape: "+str(x.shape))
    # print("batch_size = "+str(batch_size))
    # print("channels = "+str(channels))
    # Pairs of nodes should already be grouped together
    if pooltype=='max':
        x = tf.reshape(x,[batch_size,-1,int(math.pow(2,steps)),channels])
        outputs = tf.reduce_max(x,axis=2)
    elif pooltype=='avg':
        x = tf.reshape(x,[batch_size,-1,int(math.pow(2,steps)),channels])
        outputs = tf.reduce_mean(x,axis=2)
    elif pooltype=='avg_ignore_zeros':
        px = x
        for step in range(steps):
            px = tf.reshape(px,[batch_size,-1,2,channels])
            line0 = tf.slice(px,[0,0,0,0],[-1,-1,1,-1])
            line1 = tf.slice(px,[0,0,1,0],[-1,-1,-1,-1])

            z0 = tf.equal(line0,0)
            z1 = tf.equal(line1,0)

            z0 = tf.reduce_all(z0,axis=-1,keep_dims=True)
            z1 = tf.reduce_all(z1,axis=-1,keep_dims=True)

            z0 = tf.tile(z0,[1,1,1,channels])
            z1 = tf.tile(z1,[1,1,1,channels])

            cline0 = tf.where(z0,line1,line0)
            cline1 = tf.where(z1,line0,line1)

            cx = tf.concat([cline0,cline1],axis=2)

            px = tf.reduce_mean(cx,axis=2)
        outputs = px
    return outputs

def custom_upsampling(x, steps=1):
    batch_size, input_size, channels = x.get_shape().as_list()

    x = tf.expand_dims(x,axis=2)
    outputs = tf.tile(x,[1,1,int(math.pow(2,steps)),1])
    outputs = tf.reshape(outputs,[batch_size,-1,channels])

    return outputs


def filterAdj(x, adj, zeroValue):

    # batch_size, N, in_ch = x.get_shape().as_list()
    # # x : [batch, N, ch]
    # _, _, K = adj.get_shape().as_list()

    # zeroNodes = tf.reduce_all(tf.equal(x,zeroValue),axis=-1)
    # # [batch, N]
    # nodeIndices = tf.where(zeroNodes)
    # # [zeroNodesNum,2]
    # nodeIndicesB = nodeIndices[:,0]
    # nodeIndicesN = nodeIndices[:,1]
    # nodeIndicesN = nodeIndicesN+1   # Adj is one-indexed

    # nodeIndices = tf.stack((nodeIndicesB, nodeIndicesN),axis=1)

    # myRange = tf.range(N+1)
    # # [N]
    # myRange = tf.expand_dims(myRange,axis=0)
    # # [1, N]
    # # myRange = tf.Variable(tf.tile(myRange,[batch_size, 1]), trainable=False, expected_shape=[-1,N+1], validate_shape=False)
    # # [batch, N]
    # # myRange = tf.Variable(tf.tile(myRange,[batch_size, 1]), trainable=False, expected_shape=[-1,N+1])
    # myRange = tf.Variable(myRange)

    # zeroVec = tf.zeros_like(nodeIndicesN, dtype=tf.int32)
    # # [zeroNodesNum]


    # upRange = tf.scatter_nd_update(myRange,nodeIndices, zeroVec)
    # # [batch, N]
    # print("upRange shape = "+str(upRange.shape))
    # newAdjLst = []
    # for b in range(batch_size):
    #     upRangeb = upRange[b,:]
    #     print("upRangeb shape = "+str(upRangeb.shape))
    #     # [N]
    #     # adjb = adj[b,:,:]
    #     adjb = adj[0,:,:]
    #     # [N, K]
    #     newAdjb = tf.gather(upRangeb,adjb)
    #     print("newAdjb shape = "+str(newAdjb.shape))
    #     # [N, K]
    #     newAdjLst.append(newAdjb)

    

    batch_size, N, in_ch = x.get_shape().as_list()
    # x : [batch, N, ch]
    _, _, K = adj.get_shape().as_list()

    myRange = tf.range(N+1)
    # [N]
    # myRange = tf.Variable(myRange, name='rangeVariable')
    adjb = adj[0,:,:]
    newAdjLst = []
    for b in range(batch_size):
        xb = x[b,:,:]
        # [N, ch]
        zeroNodes = tf.reduce_all(tf.equal(xb,-3.0),axis=-1)
        # [N]
        nodeIndices = tf.where(zeroNodes)
        nodeIndices = nodeIndices + 1
        # [zeroNodesNum,1]
        nodeIndicesN = nodeIndices[:,0]
        zeroVec = tf.zeros_like(nodeIndicesN, dtype=tf.int32)
        oneVec = tf.ones_like(nodeIndicesN,dtype=tf.int32)
        # myRangeVar = tf.Variable(myRange, name='rangeVariable')

        # upRangeb = tf.tensor_scatter_update(myRange,nodeIndices, zeroVec)
        zeroValOnes = tf.scatter_nd(nodeIndices,oneVec, [N+1])

        zeroVal = tf.equal(zeroValOnes,1)

        upRangeb = tf.where(zeroVal, tf.zeros([N+1],dtype=tf.int32), myRange)
        # upRangeb = tf.where(zeroVal, tf.zeros([N+1],dtype=tf.int32), tf.zeros([N+1],dtype=tf.int32))


        # upRangeb = myRange

        newAdjb = tf.gather(upRangeb,adjb)
        # newAdjb = adjb
        newAdjLst.append(newAdjb)


    newAdj = tf.stack(newAdjLst, axis=0)

    return newAdj


def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def lrelu6(x, alpha):
  return tf.nn.relu6(x) - alpha * tf.nn.relu6(-x)

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
        default_M_conv = 18
        bTransInvariant = False
        bRotInvariant = False
        if architecture == 0:       # Original Nitika's architecture (3 conv layers)
                
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

        if architecture == 1:       # Copy of original architecture, with relus replaced by sigmoids
                
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

        if architecture == 2:       # one big linear hidden layer

            out_channels_fc0 = 60000
            h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0),name="h_fc0")
            nan0 = tf.is_nan(h_fc0,name="nan0")

            out_channels_reg = 1
            y_conv = custom_lin(h_fc0, out_channels_reg)
            return y_conv

        if architecture == 3:       # Two smaller linear layers

            out_channels_fc0 = 1800
            h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))

            out_channels_fc1 = 1800
            h_fc1 = tf.nn.relu(custom_lin(h_fc0, out_channels_fc1))

            out_channels_reg = 1
            y_conv = custom_lin(h_fc1, out_channels_reg)
            return y_conv

        if architecture == 4:       #Six small linear layers (+ dropout)

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

        if architecture == 5:       # Reusable archi 4 (w/o dropout)

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

        if architecture == 6:       # One conv layer, concatenated w/ output of previous layer

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

        if architecture == 7:       # Kind of like 6, with one extra conv layer

            out_channels_fc0 = 16
            h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
            
            # Conv1
            M_conv1 = 9
            out_channels_conv1 = 32
            h_conv1, _ = custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1,translation_invariance=bTransInvariant)
            h_conv1_act = tf.nn.relu(h_conv1)

            # Conv2
            M_conv2 = 9
            out_channels_conv2 = 40     #64
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

        if architecture == 8:       # Reusable archi 0 for iterative network
            
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

        if architecture == 9:       # Auto-encoder test
                
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

        if architecture == 10:      # 3 conv layers, first one is rotation invariant

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

        if architecture == 11:      # 4 conv layers, first one is rotation invariant

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


        if architecture == 12:      # 4 conv layers, first one is rotation invariant

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

        if architecture == 13:      # Like 10, with MORE WEIGHTS!

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

        if architecture == 14:      # 3 conv layers, first one is translation invariant for position only. Position is only used for assignment

            # Conv1
            M_conv1 = default_M_conv
            out_channels_conv1 = 16
            h_conv1, _ = custom_conv2d_pos_for_assignment(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
            h_conv1_act = tf.nn.relu(h_conv1)

            # Conv2
            M_conv2 = default_M_conv
            out_channels_conv2 = 32
            h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv2_act = tf.nn.relu(h_conv2)

            # Conv3
            M_conv3 = default_M_conv
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

        if architecture == 15:      # Same as 14, with concatenation at every layer

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


        if architecture == 16:      # Like 14, with smaller weights (and bigger M)

            # Conv1
            M_conv1 = default_M_conv
            out_channels_conv1 = 8
            h_conv1, _ = custom_conv2d_pos_for_assignment(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
            h_conv1_act = tf.nn.relu(h_conv1)

            # Conv2
            M_conv2 = default_M_conv
            out_channels_conv2 = 16
            h_conv2, _ = custom_conv2d(h_conv1_act, adj, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv2_act = tf.nn.relu(h_conv2)

            # Conv3
            M_conv3 = default_M_conv
            out_channels_conv3 = 32
            h_conv3, _ = custom_conv2d(h_conv2_act, adj, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv3_act = tf.nn.relu(h_conv3)

            # Lin(1024)
            out_channels_fc1 = 1024
            h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))
            
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
    alpha = 0.2
    
    coarsening_steps=2

    if architecture == 0:       # Multi-scale, like in FeaStNet paper (figure 3)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)
        # [batch, N/4, 64]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = tf.nn.relu(dconv3)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        #DeConv2
        concat2 = tf.concat([upsamp2, h_conv2_act], axis=-1)
        # [batch, N/2, 64]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = tf.nn.relu(dconv2)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(concat1, out_channels_fc1))
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 1:       # Multi-scale, w/ extra convolutions in encoder

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)
        # [batch, N, 8]

        # Conv1_2
        M_conv1_2 = 9
        out_channels_conv1_2 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1_2, _ = custom_conv2d(h_conv1_act, adjs[0], out_channels_conv1_2, M_conv1_2, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_2_act = tf.nn.relu(h_conv1_2)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_2_act, steps=coarsening_steps)   # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)
        # [batch, N/2, 32]

        # Conv2_2
        M_conv2_2 = 9
        out_channels_conv2_2 = 32
        h_conv2_2, _ = custom_conv2d(h_conv2_act, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_2_act = tf.nn.relu(h_conv2_2)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)
        # [batch, N/4, 64]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = tf.nn.relu(dconv3)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        #DeConv2
        concat2 = tf.concat([upsamp2, h_conv2_2_act], axis=-1)
        # [batch, N/2, 64]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = tf.nn.relu(dconv2)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_2_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(concat1, out_channels_fc1))
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 2:       # Like 0, with decoder weights preset by encoder. No skip-connections

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, conv1_W = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, conv2_W = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, conv3_W = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)
        # [batch, N/4, 64]

        # --- Central features ---

        conv3_Wp = tf.transpose(conv3_W,[0,2,1])
        dconv3 = decoding_layer(h_conv3_act, adjs[2], conv3_Wp, translation_invariance=bTransInvariant)
        dconv3_act = tf.nn.relu(dconv3)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        conv2_Wp = tf.transpose(conv2_W,[0,2,1])
        dconv2 = decoding_layer(upsamp2, adjs[1], conv2_Wp, translation_invariance=bTransInvariant)
        dconv2_act = tf.nn.relu(dconv2)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        
        conv1_Wp = tf.transpose(conv1_W,[0,2,1])
        dconv1 = decoding_layer(upsamp1, adjs[0], conv1_Wp, translation_invariance=bTransInvariant)
        dconv1_act = tf.nn.relu(dconv1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(dconv1_act, out_channels_fc1))
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 3:       # Like 0, but w/ pos for assignment

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0,pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1,pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)
        # [batch, N/4, 64]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = tf.nn.relu(dconv3)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        #DeConv2
        concat2 = tf.concat([upsamp2, h_conv2_act], axis=-1)
        # [batch, N/2, 64]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = tf.nn.relu(dconv2)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(concat1, out_channels_fc1))
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 4:       # Like 3, but w/ pos for assignment only for 1st convolution, to try and fine-tune archi 14 (non multi-scale)

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0,pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1,pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)
        # [batch, N/4, 64]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = tf.nn.relu(dconv3)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        #DeConv2
        concat2 = tf.concat([upsamp2, h_conv2_act], axis=-1)
        # [batch, N/2, 64]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = tf.nn.relu(dconv2)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(concat1, out_channels_fc1))
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 5:       # copy of archi 14, w/ extra pooling + conv + upsampling layer

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(h_conv1_act, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, _ = custom_conv2d(h_conv2_act, adjs[0], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 64]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 128
        h_conv4, _ = custom_conv2d(pool1, adjs[1], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = tf.nn.relu(h_conv4)
        # [batch, N/2, 128]

        # Conv5
        M_conv5 = 9
        out_channels_conv5 = 64
        h_conv5, _ = custom_conv2d(pool1, adjs[1], out_channels_conv5, M_conv5,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv5_act = tf.nn.relu(h_conv5)
        # [batch, N/2, 64]

        #Upsampling2
        upsamp2 = custom_upsampling(h_conv5_act, steps=coarsening_steps)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(h_conv3_act, out_channels_fc1))
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv


    if architecture == 6:       # Like 0, w/ leaky ReLU
        x = tf.slice(x,[0,0,0],[-1,-1,3])   #Take normals only
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/4, 64]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        #DeConv2
        concat2 = tf.concat([upsamp2, h_conv2_act], axis=-1)
        # [batch, N/2, 64]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(concat1, out_channels_fc1),alpha)
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 7:       # Like 3, but w/ leaky ReLU

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/2, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/4, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/4, 64]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv2, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/4, 32]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/2, 32]

        #DeConv2
        concat2 = tf.concat([upsamp2, h_conv2_act], axis=-1)
        # [batch, N/2, 64]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/2, 16]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(concat1, out_channels_fc1),alpha)
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 8:       # Like 6, w/ 1 pooling only, and normals + pos
        #x = tf.slice(x,[0,0,0],[-1,-1,3])  #Take normals only
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/2, 32]

        # --- Central features ---

        #DeConv2
        dconv2, _ = custom_conv2d(h_conv2_act, adjs[1], out_channels_conv1, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 32]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv1_act], axis=-1)

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(concat1, out_channels_fc1),alpha)
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 9:       # Inspired by U-net. kind of like 7 w/ more weights, and extra conv after upsampling and before concatenating (no ReLU)
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 10:      # Like 9, without the position for assignment
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        x = tf.slice(x,[0,0,0],[-1,-1,3])
        # pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 11:      # Normals, conv16, conv32, pool4, conv64, conv32, upsamp4, Lin256, Lin3
        x = tf.slice(x,[0,0,0],[-1,-1,3])   #Take normals only
        coarsening_steps=2
        M_conv = 6

        # Conv1
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 16]

        out_channels_conv2 = 16
        h_conv2, _ = custom_conv2d(h_conv1_act, adjs[0], out_channels_conv2, M_conv, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv2_act = lrelu(h_conv2,alpha)

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        # Conv3
        out_channels_conv3 = 24
        h_conv3, _ = custom_conv2d(pool1, adjs[1], out_channels_conv3, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/2, 32]

        # --- Central features ---

        #DeConv2
        dconv2, _ = custom_conv2d(h_conv3_act, adjs[1], out_channels_conv2, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 32]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        concat1 = tf.concat([upsamp1, h_conv2_act], axis=-1)

        #DeConv1
        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)

        # Lin(1024)
        out_channels_fc1 = 128
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 12:      # Branching out before 1st pooling, classification leading to 4 separate branches.
        x = tf.slice(x,[0,0,0],[-1,-1,3])   #Take normals only
        coarsening_steps=2
        M_conv = 6

        # Conv1
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 16]

        out_channels_conv2 = 12
        h_conv2, _ = custom_conv2d(h_conv1_act, adjs[0], out_channels_conv2, M_conv, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv2_act = lrelu(h_conv2,alpha)

        # Classification
        branches_num = 4
        p_classes = lrelu(custom_lin(h_conv2_act, branches_num),alpha)
        p_classes = tf.nn.softmax(p_classes)
        # [batch, N, 4]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/2, 16]

        out_channels_conv3 = 16
        out_channels_fc1 = 64
        # ---branches: ---

        #Branch 0
        # Conv3
        h_conv3_0, _ = custom_conv2d(pool1, adjs[1], out_channels_conv3, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_0_act = lrelu(h_conv3_0,alpha)

        #DeConv2
        dconv2_0, _ = custom_conv2d(h_conv3_0_act, adjs[1], out_channels_conv2, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_0_act = lrelu(dconv2_0,alpha)
        # [batch, N/4, 32]

        #Upsampling1
        upsamp1_0 = custom_upsampling(dconv2_0_act, steps=coarsening_steps)
        concat1_0 = tf.concat([upsamp1_0, h_conv2_act], axis=-1)

        #DeConv1
        dconv1_0, _ = custom_conv2d(concat1_0, adjs[0], out_channels_conv1, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_0_act = lrelu(dconv1_0,alpha)

        # Lin
        h_fc1_0 = lrelu(custom_lin(dconv1_0_act, out_channels_fc1),alpha)
        
        #Branch 1
        # Conv3
        h_conv3_1, _ = custom_conv2d(pool1, adjs[1], out_channels_conv3, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_1_act = lrelu(h_conv3_1,alpha)

        #DeConv2
        dconv2_1, _ = custom_conv2d(h_conv3_1_act, adjs[1], out_channels_conv2, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_1_act = lrelu(dconv2_1,alpha)
        # [batch, N/4, 32]

        #Upsampling1
        upsamp1_1 = custom_upsampling(dconv2_1_act, steps=coarsening_steps)
        concat1_1 = tf.concat([upsamp1_1, h_conv2_act], axis=-1)

        #DeConv1
        dconv1_1, _ = custom_conv2d(concat1_1, adjs[0], out_channels_conv1, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_1_act = lrelu(dconv1_1,alpha)

        # Lin
        h_fc1_1 = lrelu(custom_lin(dconv1_1_act, out_channels_fc1),alpha)

        #Branch 2
        # Conv3
        h_conv3_2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv3, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_2_act = lrelu(h_conv3_2,alpha)

        #DeConv2
        dconv2_2, _ = custom_conv2d(h_conv3_2_act, adjs[1], out_channels_conv2, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_2_act = lrelu(dconv2_2,alpha)
        # [batch, N/4, 32]

        #Upsampling1
        upsamp1_2 = custom_upsampling(dconv2_2_act, steps=coarsening_steps)
        concat1_2 = tf.concat([upsamp1_2, h_conv2_act], axis=-1)

        #DeConv1
        dconv1_2, _ = custom_conv2d(concat1_2, adjs[0], out_channels_conv1, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_2_act = lrelu(dconv1_2,alpha)

        # Lin
        h_fc1_2 = lrelu(custom_lin(dconv1_2_act, out_channels_fc1),alpha)

        #Branch 3
        # Conv3
        h_conv3_3, _ = custom_conv2d(pool1, adjs[1], out_channels_conv3, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_3_act = lrelu(h_conv3_3,alpha)

        #DeConv2
        dconv2_3, _ = custom_conv2d(h_conv3_3_act, adjs[1], out_channels_conv2, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_3_act = lrelu(dconv2_3,alpha)
        # [batch, N/4, 32]

        #Upsampling1
        upsamp1_3 = custom_upsampling(dconv2_3_act, steps=coarsening_steps)
        concat1_3 = tf.concat([upsamp1_3, h_conv2_act], axis=-1)

        #DeConv1
        dconv1_3, _ = custom_conv2d(concat1_3, adjs[0], out_channels_conv1, M_conv,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_3_act = lrelu(dconv1_3,alpha)

        # Lin
        h_fc1_3 = lrelu(custom_lin(dconv1_3_act, out_channels_fc1),alpha)
        # [batch, N, 3]

        # merging

        h_fc1_0 = tf.expand_dims(h_fc1_0,axis=-1)
        # [batch, N, 128, 1]
        h_fc1_1 = tf.expand_dims(h_fc1_1,axis=-1)
        h_fc1_2 = tf.expand_dims(h_fc1_2,axis=-1)
        h_fc1_3 = tf.expand_dims(h_fc1_3,axis=-1)


        h_fc_final = tf.concat([h_fc1_0,h_fc1_1,h_fc1_2,h_fc1_3], axis=-1)


        p_classes = tf.expand_dims(p_classes,axis=2)
        # [batch, N, 1, 4]
        p_classes = tf.tile(p_classes,[1,1,out_channels_fc1,1])
        # [batch, N, 128, 4]

        h_fc_final = tf.multiply(h_fc_final,p_classes)

        h_fc_final = tf.reduce_sum(h_fc_final,axis=-1)
        # Lin(num_classes)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc_final, out_channels_reg)
        return y_conv


    if architecture == 13:      # Like 9, w/ fewer weights everywhere
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 256
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 14:      # Like 9, with 3 coarsening steps between each level
        alpha = 0.1
        coarsening_steps = 3
        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv

    if architecture == 15:      # Like 14, but multi-scale estimation
        alpha = 0.1
        coarsening_steps = 3
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        # Lin(1024)
        out_channels_fc2 = 1024
        h_fc2 = lrelu(custom_lin(dconv3_act, out_channels_fc2),alpha)
        # Lin(3)
        y_conv2 = custom_lin(h_fc2, out_channels_reg)

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]


        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv2_act, out_channels_fc1),alpha)
        # Lin(3)
        y_conv1 = custom_lin(h_fc1, out_channels_reg)

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc0 = 1024
        h_fc0 = lrelu(custom_lin(dconv1_act, out_channels_fc0),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, y_conv1, y_conv2

    if architecture == 16:      # Like 15, w/ coarsening_steps=2
        alpha = 0.1
        coarsening_steps = 2
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        # Lin(1024)
        out_channels_fc2 = 1024
        h_fc2 = lrelu(custom_lin(dconv3_act, out_channels_fc2),alpha)
        # Lin(3)
        y_conv2 = custom_lin(h_fc2, out_channels_reg)

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]


        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv2_act, out_channels_fc1),alpha)
        # Lin(3)
        y_conv1 = custom_lin(h_fc1, out_channels_reg)

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc0 = 1024
        h_fc0 = lrelu(custom_lin(dconv1_act, out_channels_fc0),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, y_conv1, y_conv2

    if architecture == 17:      # Like 16, w/ dropout
        alpha = 0.1
        coarsening_steps = 2
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        dropout0 = tf.nn.dropout(h_conv1,keep_prob)
        h_conv1_act = lrelu(dropout0,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout1 = tf.nn.dropout(h_conv2,keep_prob)
        h_conv2_act = lrelu(dropout1,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout2 = tf.nn.dropout(h_conv3,keep_prob)
        h_conv3_act = lrelu(dropout2,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout3 = tf.nn.dropout(dconv3,keep_prob)
        dconv3_act = lrelu(dropout3,alpha)
        # [batch, N/16, 256]

        # Lin(1024)
        out_channels_fc2 = 1024
        h_fc2 = lrelu(tf.nn.dropout(custom_lin(dconv3_act, out_channels_fc2),keep_prob),alpha)
        # Lin(3)
        y_conv2 = custom_lin(h_fc2, out_channels_reg)

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout5 = tf.nn.dropout(upconv2,keep_prob)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([dropout5, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout6 = tf.nn.dropout(dconv2,keep_prob)
        dconv2_act = lrelu(dropout6,alpha)
        # [batch, N/4, 128]


        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(tf.nn.dropout(custom_lin(dconv2_act, out_channels_fc1),keep_prob),alpha)
        # Lin(3)
        y_conv1 = custom_lin(h_fc1, out_channels_reg)

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout8 = tf.nn.dropout(upconv1,keep_prob)
        # [batch, N, 64]

        concat1 = tf.concat([dropout8, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout9 = tf.nn.dropout(dconv1,keep_prob)
        dconv1_act = lrelu(dropout9,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc0 = 1024
        h_fc0 = lrelu(tf.nn.dropout(custom_lin(dconv1_act, out_channels_fc0),keep_prob),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, y_conv1, y_conv2

    if architecture == 18:      # Like 17, w/ 2 extra convolutions at coarsest level
        alpha = 0.1
        coarsening_steps = 2
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        dropout0 = tf.nn.dropout(h_conv1,keep_prob)
        h_conv1_act = lrelu(dropout0,alpha)
        # [batch, N, 16]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 16]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout1 = tf.nn.dropout(h_conv2,keep_prob)
        h_conv2_act = lrelu(dropout1,alpha)
        # [batch, N/4, 32]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 32]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout2 = tf.nn.dropout(h_conv3,keep_prob)
        h_conv3_act = lrelu(dropout2,alpha)
        # [batch, N/16, 64]

        # Conv3_2
        # Add position:
        pool2_2 = tf.concat([h_conv3_act,pos2],axis=-1)
        h_conv3_2, _ = custom_conv2d_pos_for_assignment(pool2_2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout2_2 = tf.nn.dropout(h_conv3_2,keep_prob)
        h_conv3_2_act = lrelu(dropout2_2,alpha)
        # [batch, N/16, 64]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_2_act = tf.concat([h_conv3_2_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_2_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout3 = tf.nn.dropout(dconv3,keep_prob)
        dconv3_act = lrelu(dropout3,alpha)
        # [batch, N/16, 64]

        #DeConv3_2
        # Add position:
        dconv3_act = tf.concat([dconv3_act,pos2],axis=-1)
        dconv3_2, _ = custom_conv2d_pos_for_assignment(dconv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout3_2 = tf.nn.dropout(dconv3_2,keep_prob)
        dconv3_2_act = lrelu(dropout3_2,alpha)
        # [batch, N/16, 64]

        # Lin(1024)
        out_channels_fc2 = 256
        h_fc2 = lrelu(tf.nn.dropout(custom_lin(dconv3_2_act, out_channels_fc2),keep_prob),alpha)
        # Lin(3)
        y_conv2 = custom_lin(h_fc2, out_channels_reg)

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_2_act, steps=coarsening_steps)
        # [batch, N/4, 64]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout5 = tf.nn.dropout(upconv2,keep_prob)
        # [batch, N/4, 32]

        #DeConv2
        concat2 = tf.concat([dropout5, h_conv2_act], axis=-1)
        # [batch, N/4, 64]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout6 = tf.nn.dropout(dconv2,keep_prob)
        dconv2_act = lrelu(dropout6,alpha)
        # [batch, N/4, 32]


        # Lin(1024)
        out_channels_fc1 = 256
        h_fc1 = lrelu(tf.nn.dropout(custom_lin(dconv2_act, out_channels_fc1),keep_prob),alpha)
        # Lin(3)
        y_conv1 = custom_lin(h_fc1, out_channels_reg)

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 32]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout8 = tf.nn.dropout(upconv1,keep_prob)
        # [batch, N, 16]

        concat1 = tf.concat([dropout8, h_conv1_act], axis=-1)
        # [batch, N, 32]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout9 = tf.nn.dropout(dconv1,keep_prob)
        dconv1_act = lrelu(dropout9,alpha)
        # [batch, N, 16]

        # Lin(1024)
        out_channels_fc0 = 256
        h_fc0 = lrelu(tf.nn.dropout(custom_lin(dconv1_act, out_channels_fc0),keep_prob),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, y_conv1, y_conv2

    if architecture == 19:      # Like 15, but 2 * 3-channels output
        alpha = 0.1
        coarsening_steps = 3
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]


        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc0 = 1024
        h_fc0 = lrelu(custom_lin(dconv1_act, out_channels_fc0),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)
        disp = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, disp


    if architecture == 20:      # Like 19, but no pos for assignment (and pos used as input for 1st layer)
        alpha = 0.1
        coarsening_steps = 3
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]


        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc0 = 1024
        h_fc0 = lrelu(custom_lin(dconv1_act, out_channels_fc0),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)
        disp = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, disp


    if architecture == 21:      # Like 20, but more convolutions, more filters, and less channels
        alpha = 0.1
        coarsening_steps = 3
        out_channels_reg = 3

        _, _,in_channels = x.get_shape().as_list()
        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 18
        out_channels_conv3 = 32
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Conv3_2
        h_conv3_2, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_2_act = lrelu(h_conv3_2,alpha)
        # [batch, N/16, 256]


        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_2_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #DeConv3_2
        dconv3_2, _ = custom_conv2d(dconv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_2_act = lrelu(dconv3_2,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_2_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc0 = 1024
        h_fc0 = lrelu(custom_lin(dconv1_act, out_channels_fc0),alpha)
        # Lin(3)
        y_conv0 = custom_lin(h_fc0, out_channels_reg)
        disp = custom_lin(h_fc0, out_channels_reg)

        return y_conv0, disp
# End


def get_model_shape_reg(x, adjs, architecture, keep_prob, mode='shape'):
    bTransInvariant = False
    bRotInvariant = False

    alpha = 0.1
    coarsening_steps = 3
    batch_size,_,in_channels = x.get_shape().as_list()
    print("in_channels = "+str(in_channels))
    pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
    pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
    pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
    if architecture<26:
        pos3 = custom_binary_tree_pooling(pos2, steps=coarsening_steps, pooltype='avg_ignore_zeros')

    if architecture==21 or architecture==19 or architecture==22:
        xUV = tf.slice(x,[0,0,0],[-1,-1,in_channels-3])
    else:
        xUV = tf.slice(x,[0,0,0],[-1,-1,2])

    print("xUV shape = "+str(xUV.shape))
    print("pos0 shape = "+str(pos0.shape))

    if architecture == 0:      # copied from architecture 14 above, one extra coarsening layer
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 1:      # like 0, w/o position for assignment
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)


        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)
    
    if architecture == 2:      # like 0, with two extra lin for image patch (i.e. extracting image features (end-to-end) before the main network)
        im_patch = tf.slice(x,[0,0,2],[-1,-1,27])

        # Image patch feature extraction:
        patch1 = lrelu(custom_lin(im_patch,27),alpha)
        patch2 = lrelu(custom_lin(patch1, 27),alpha)
        patch3 = lrelu(custom_lin(patch2,10),alpha)

        newX = tf.concat((xUV,patch3,pos0),axis=-1)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 3:      # like 2, with extra channels for hidden layers!!!!!!!
        im_patch = tf.slice(x,[0,0,2],[-1,-1,27])

        # Image patch feature extraction:
        patch1 = lrelu(custom_lin(im_patch,27),alpha)
        patch2 = lrelu(custom_lin(patch1, 27),alpha)
        patch3 = lrelu(custom_lin(patch2,10),alpha)

        newX = tf.concat((xUV,patch3,pos0),axis=-1)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 128
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 4:      # like 3, without color input, and translation invariance  
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 128
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 5:      # like 4, with smaller features
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 6:      # like 5, with ReLU (not leaky)
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.relu(h_conv1)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.relu(h_conv2)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.relu(h_conv3)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = tf.nn.relu(h_conv4)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = tf.nn.relu(dconv4)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = tf.nn.relu(custom_lin(fcIn, out_channels_fc1))

    if architecture == 7:      # like 5, with deactivated 0-nodes
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        for i in range(1,4):
            adjs[i] = tf.tile(adjs[i],[batch_size,1,1])



        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos3],axis=-1)
        dconv4, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 8:      # like 7, without pos for assignment, but with pos as input
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        for i in range(1,4):
            adjs[i] = tf.tile(adjs[i],[batch_size,1,1])



        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 9:      # like 8, without translation invariance
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        for i in range(1,4):
            adjs[i] = tf.tile(adjs[i],[batch_size,1,1])



        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 10:      # like 9, with "node killing" at every scale (w/ max pooling)
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 11:      # like 10, with more channels
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))
        print("xUV1 shape = "+str(xUV1.shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        print("new adjs1 shape = "+str(adjs[1].shape))

        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)
        print("new adjs3 shape = "+str(adjs[3].shape))


        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 128
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 12:      # like 10, with dropout
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        dropout1 = tf.nn.dropout(h_conv1,keep_prob)
        h_conv1_act = lrelu(dropout1,alpha)

        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout2 = tf.nn.dropout(h_conv2,keep_prob)
        h_conv2_act = lrelu(dropout2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout3 = tf.nn.dropout(h_conv3,keep_prob)
        h_conv3_act = lrelu(dropout3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout4 = tf.nn.dropout(h_conv4,keep_prob)
        h_conv4_act = lrelu(dropout4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout5 = tf.nn.dropout(dconv4,keep_prob)
        dconv4_act = lrelu(dropout5,alpha)
        
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)
        h_fc1 = tf.nn.dropout(h_fc1,keep_prob)

    if architecture == 13:      # like 12, with less weights
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 6
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        dropout1 = tf.nn.dropout(h_conv1,keep_prob)
        h_conv1_act = lrelu(dropout1,alpha)

        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout2 = tf.nn.dropout(h_conv2,keep_prob)
        h_conv2_act = lrelu(dropout2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout3 = tf.nn.dropout(h_conv3,keep_prob)
        h_conv3_act = lrelu(dropout3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 32
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout4 = tf.nn.dropout(h_conv4,keep_prob)
        h_conv4_act = lrelu(dropout4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dropout5 = tf.nn.dropout(dconv4,keep_prob)
        dconv4_act = lrelu(dropout5,alpha)
        
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 512
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)
        h_fc1 = tf.nn.dropout(h_fc1,keep_prob)

    if architecture == 14:      # like 10, with translation invariance for 1st layer
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 15:      # like 10, with pos for assignment
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 16:      # like 15, with translation invariance for 1st layer
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 17:      # like 15, with ONLY pos for assignment, and translation invariance
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 18:      # like 17, without translation invariance
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=False, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=False, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=False, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=False, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=False, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 19:      # like 17, with image patches
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 20:      # like 16, with max pooling instead of FC layer
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))

        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        # max pool
        fcIn = tf.reduce_max(dconv4_act,axis=1, keep_dims=True)

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 21:      # like 19, with max pooling instead of FC layer, and with leaky ReLU6
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu6(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 16
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu6(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 32
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu6(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu6(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu6(dconv4,alpha)
        # [batch, N/16, 256]

        # max pool
        fcIn = tf.reduce_max(dconv4_act,axis=1, keep_dims=True)

        out_channels_fc1 = 1024
        h_fc1 = lrelu6(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 22:      # like 21, with drastically less weights (M too)
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 6
        out_channels_conv1 = 6
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu6(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 6
        out_channels_conv2 = 12
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu6(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 6
        out_channels_conv3 = 16
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu6(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 6
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu6(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu6(dconv4,alpha)
        # [batch, N/16, 256]

        # max pool
        fcIn = tf.reduce_max(dconv4_act,axis=1, keep_dims=True)

        out_channels_fc1 = 1024
        h_fc1 = lrelu6(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 23:      # like 22, with 2 input channels
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))

        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 6
        out_channels_conv1 = 6
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu6(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 6
        out_channels_conv2 = 12
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu6(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 6
        out_channels_conv3 = 16
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu6(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 6
        out_channels_conv4 = 64
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu6(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu6(dconv4,alpha)
        # [batch, N/16, 256]

        # max pool
        fcIn = tf.reduce_max(dconv4_act,axis=1, keep_dims=True)

        out_channels_fc1 = 1024
        h_fc1 = lrelu6(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 24:      # like 23, with M=9 and less weights
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 5
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d_only_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu6(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 10
        h_conv2, adj_size2 = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu6(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 15
        h_conv3, adj_size3 = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu6(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Add position:
        pool3 = tf.concat([pool3,pos3],axis=-1)
        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 32
        h_conv4, _ = custom_conv2d_only_pos_for_assignment(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu6(h_conv4,alpha)
        # [batch, N/16, 256]

        # Add position:
        pool4 = tf.concat([h_conv4_act,pos3],axis=-1)
        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d_only_pos_for_assignment(pool4, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu6(dconv4,alpha)
        # [batch, N/16, 256]

        # max pool
        fcIn = tf.reduce_max(dconv4_act,axis=1, keep_dims=True)

        out_channels_fc1 = 128
        h_fc1 = lrelu6(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 25:      # like 24, with pos as input for 1st layer
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 5
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu6(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 10
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu6(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 15
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu6(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 32
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu6(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu6(dconv4,alpha)
        # [batch, N/16, 256]

        # max pool
        fcIn = tf.reduce_max(dconv4_act,axis=1, keep_dims=True)

        out_channels_fc1 = 128
        h_fc1 = lrelu6(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 26:       # Copied from function above (previous work) Inspired by U-net.
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d_pos_for_assignment(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 4
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv[:,:,:3], tf.expand_dims(y_conv[:,:,3],axis=-1)


    if architecture == 27:       # Like 26 with pos as input (every layer)
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        # Add position:
        pool1 = tf.concat([pool1,pos1],axis=-1)
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        # Add position:
        pool2 = tf.concat([pool2,pos2],axis=-1)
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]
        
        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 4
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv[:,:,:3], tf.expand_dims(y_conv[:,:,3],axis=-1)

    if architecture == 28:       # Like 26 with pos as input (first layer)
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)




        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]
        
        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 4
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv[:,:,:3], tf.expand_dims(y_conv[:,:,3],axis=-1)

    if architecture == 29:       # Like 28, with full batch normalization
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]
        h_conv1_act = batch_norm(h_conv1_act)

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]
        h_conv2_act = batch_norm(h_conv2_act)

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]
        h_conv3_act = batch_norm(h_conv3_act)

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]
        dconv3_act = batch_norm(dconv3_act)

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]
        dconv2_act = batch_norm(dconv2_act)

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]
        dconv1_act = batch_norm(dconv1_act)
        
        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 4
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv[:,:,:3], tf.expand_dims(y_conv[:,:,3],axis=-1)

    if architecture == 30:       # Like 28 without node input filtering
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        # adjs[0] = filterAdj(xUV, adjs[0], -3)
        # print("new adjs0 shape = "+str(adjs[0].shape))

        # adjs[1] = filterAdj(xUV1, adjs[1], -3)
        # adjs[2] = filterAdj(xUV2, adjs[2], -3)
        for adjnum in range(len(adjs)):
            adjs[adjnum] = tf.tile(adjs[adjnum], [batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]
        
        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 4
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv[:,:,:3], tf.expand_dims(y_conv[:,:,3],axis=-1)


    if architecture == 31:       # Like 28 with normal adjacency for "decoder" part
        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')

        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        filterAdj0 = filterAdj(xUV, adjs[0], -3)
        filterAdj1 = filterAdj(xUV, adjs[1], -3)
        filterAdj2 = filterAdj(xUV, adjs[2], -3)

        for adjnum in range(len(adjs)):
            adjs[adjnum] = tf.tile(adjs[adjnum], [batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 8
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(newX, filterAdj0, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 8
        h_conv2, _ = custom_conv2d(pool1, filterAdj1, out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 16
        h_conv3, _ = custom_conv2d(pool2, filterAdj2, out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        dconv2, _ = custom_conv2d(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]
        
        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 4
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv[:,:,:3], tf.expand_dims(y_conv[:,:,3],axis=-1)

    if mode=='shape':
        # Lin(3)
        out_channels_reg = 10
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # [batch, 1, 10]
        finalFeature = tf.reshape(finalFeature,[batch_size,-1])
        # [batch, 10]
        # return finalFeature, tf.concat((adj_size,custom_upsampling(adj_size2,coarsening_steps),custom_upsampling(adj_size3,coarsening_steps*2)),axis=-1)
        # return finalFeature, tf.tile(adj_size,[1,1,3])
        # return finalFeature, tf.tile(custom_upsampling(adj_size3,coarsening_steps*2),[1,1,3])
        return finalFeature, custom_upsampling(h_conv3_act, coarsening_steps*2)
        # return finalFeature, h_conv1_act

    elif mode=='gender':
        # Lin(3)
        out_channels_reg = 1
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.sigmoid(finalFeature)
        # [batch, 1, 1]
        finalProba = tf.reshape(finalProba,[batch_size,-1])
        # [batch]
        # return finalProba, custom_upsampling(h_conv3_act, steps=coarsening_steps*2)
        return finalProba, custom_upsampling(h_conv2_act, coarsening_steps)
        # return finalProba, h_conv1_act

    elif mode=='shapeClass':
        # Lin(3)
        out_channels_reg = 2
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.nn.softmax(finalFeature)
        # [batch, 1, 1]
        finalProba = tf.reshape(finalProba,[batch_size,-1])
        # [batch]
        # return finalProba, custom_upsampling(h_conv3_act, steps=coarsening_steps*2)
        return finalProba, custom_upsampling(h_conv2_act, coarsening_steps)
        # return finalProba, h_conv1_act
    elif mode=='bodyratio':
        # Lin(3)
        out_channels_reg = 1
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.sigmoid(finalFeature)
        # [batch, 1, 1]
        finalProba = tf.reshape(finalProba,[batch_size,-1])
        # [batch]
        # return finalProba, custom_upsampling(h_conv3_act, steps=coarsening_steps*2)
        return finalProba, custom_upsampling(h_conv2_act, coarsening_steps)
        # return finalProba, h_conv1_act


def get_image_conv_model_gender_class(x, architecture, keep_prob, mode='shape'):
    global std_dev, std_dev_bias
    # For images
    std_dev= 0.001
    std_dev_bias = 0.001
    alpha = 0.1

    batch_size, width, height, in_channels = x.get_shape().as_list()
    print("x shape = "+str(x.shape))
    # [batch, W, H, ch]
    myact = lrelu

    if architecture==3:
        myact = lrelu6
        architecture=2

    if architecture == 0:
        padding_type = 'valid'
        out_channels_conv1 = 32
        kernel_conv1 = 3
        conv1 = tf.layers.conv2d(inputs=x, filters=out_channels_conv1, kernel_size=[kernel_conv1,kernel_conv1], padding=padding_type)
        conv1_act = lrelu(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2 = tf.layers.conv2d(inputs=conv1_act, filters=out_channels_conv1, kernel_size=[kernel_conv1,kernel_conv1], padding=padding_type)
        conv1_2_act = lrelu(conv1_2,alpha)

        pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)

        out_channels_conv2 = 64
        kernel_conv2 = 3
        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=out_channels_conv2, kernel_size=[kernel_conv2,kernel_conv2], padding=padding_type)
        conv2_1_act = lrelu(conv2_1,alpha)

        conv2_2 = tf.layers.conv2d(inputs=conv2_1_act, filters=out_channels_conv2, kernel_size=[kernel_conv2,kernel_conv2], padding=padding_type)
        conv2_2_act = lrelu(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 128
        kernel_conv3 = 3
        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=out_channels_conv3, kernel_size=[kernel_conv3,kernel_conv3], padding=padding_type)
        conv3_1_act = lrelu(conv3_1,alpha)

        conv3_2 = tf.layers.conv2d(inputs=conv3_1_act, filters=out_channels_conv3, kernel_size=[kernel_conv2,kernel_conv3], padding=padding_type)
        conv3_2_act = lrelu(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 256
        kernel_conv4 = 3
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=out_channels_conv4, kernel_size=[kernel_conv4,kernel_conv4], padding=padding_type)
        conv4_1_act = lrelu(conv4_1,alpha)

        conv4_2 = tf.layers.conv2d(inputs=conv4_1_act, filters=out_channels_conv4, kernel_size=[kernel_conv4,kernel_conv4], padding=padding_type)
        conv4_2_act = lrelu(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 256
        kernel_conv5 = 3
        conv5 = tf.layers.conv2d(inputs=pool4, filters=out_channels_conv5, kernel_size=[kernel_conv5,kernel_conv5], padding=padding_type)
        conv5_act = lrelu(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fully connected layer
        fcIn = tf.reshape(conv5_act,[batch_size,1,-1])
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))



    # Using tf.nn instead of tf.layers (lower level)
    if architecture == 1:
        padding_type = 'VALID'
        out_channels_conv1 = 32
        kernel_conv1 = 3
        conv1 = image_conv2d(x, out_channels_conv1, kernel_conv1, padding=padding_type)
        conv1_act = lrelu(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2 = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, padding=padding_type)
        conv1_2_act = lrelu(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 64
        kernel_conv2 = 3
        conv2_1 = image_conv2d(pool1, out_channels_conv2, kernel_conv2, padding=padding_type)
        conv2_1_act = lrelu(conv2_1,alpha)

        conv2_2 = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, padding=padding_type)
        conv2_2_act = lrelu(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 128
        kernel_conv3 = 3
        conv3_1 = image_conv2d(pool2, out_channels_conv3, kernel_conv3, padding=padding_type)
        conv3_1_act = lrelu(conv3_1,alpha)

        conv3_2 = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, padding=padding_type)
        conv3_2_act = lrelu(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 256
        kernel_conv4 = 3
        conv4_1 = image_conv2d(pool3, out_channels_conv4, kernel_conv4, padding=padding_type)
        conv4_1_act = lrelu(conv4_1,alpha)

        conv4_2 = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, padding=padding_type)
        conv4_2_act = lrelu(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 256
        kernel_conv5 = 3
        conv5 = image_conv2d(pool4, out_channels_conv5, kernel_conv5, padding=padding_type)
        conv5_act = lrelu(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # fully connected layer
        fcIn = tf.reshape(fcIn,[batch_size,1,-1])
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

    
    # Like 1, with masking blank parts of the input (at all levels)
    if architecture == 2:
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 32
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 64
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 128
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 256
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 256
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # fully connected layer
        fcIn = tf.reshape(fcIn,[batch_size,1,-1])
        out_channels_fc1 = 1024
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))


    if architecture == 4:   # Like 2, with max pooling instead of FC layer
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 32
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 64
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 128
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 256
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 256
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # max pooling, then 1x1 conv
        fcIn = tf.reduce_max(fcIn, axis=1)
        fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        out_channels_fc1 = 1024
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

    if architecture == 5:   # Like 4, with drastically less weights
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 6
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 12
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 18
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 24
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 64
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # max pooling, then 1x1 conv
        fcIn = tf.reduce_max(fcIn, axis=1)
        fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        out_channels_fc1 = 1024
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

    if architecture == 6:   # Like 5, with even less parameters
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 5
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 10
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 15
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 20
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 32
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # max pooling, then 1x1 conv
        fcIn = tf.reduce_max(fcIn, axis=1)
        fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        out_channels_fc1 = 128
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))


    if architecture == 7:   # Like 6, with silhouettes only (1 binary channel)
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 5
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(mask0, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 10
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 15
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 20
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 32
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # max pooling, then 1x1 conv
        fcIn = tf.reduce_max(fcIn, axis=1)
        fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        out_channels_fc1 = 128
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

    if mode=='shape':
        # Lin(3)
        out_channels_reg = 10
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # [batch, 1, 10]
        finalFeature = tf.reshape(finalFeature,[batch_size,-1])
        # [batch, 10]
        # return finalFeature, tf.concat((adj_size,custom_upsampling(adj_size2,coarsening_steps),custom_upsampling(adj_size3,coarsening_steps*2)),axis=-1)
        # return finalFeature, tf.tile(adj_size,[1,1,3])
        # return finalFeature, tf.tile(custom_upsampling(adj_size3,coarsening_steps*2),[1,1,3])
        return finalFeature, conv2_2_act
        # return finalFeature, h_conv1_act

    elif mode=='gender':
        # Lin(3)
        out_channels_reg = 1
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.sigmoid(finalFeature)
        # [batch, 1, 1]
        finalProba = tf.reshape(finalProba,[batch_size,-1])
        # [batch]
        # return finalProba, custom_upsampling(h_conv3_act, steps=coarsening_steps*2)
        return finalProba, conv5_act
        # return finalProba, h_conv1_act

    elif mode=='bodyratio':
        # Lin(3)
        out_channels_reg = 1
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.sigmoid(finalFeature)
        # [batch, 1, 1]
        finalProba = tf.reshape(finalProba,[batch_size,-1])
        # [batch]
        # return finalProba, custom_upsampling(h_conv3_act, steps=coarsening_steps*2)
        return finalProba, conv5_act
        # return finalProba, h_conv1_act


def get_image_encoder(x, architecture, keep_prob):
    global std_dev, std_dev_bias
    # For images
    std_dev= 0.001
    std_dev_bias = 0.001
    alpha = 0.1

    batch_size, width, height, in_channels = x.get_shape().as_list()
    print("x shape = "+str(x.shape))
    # [batch, W, H, ch]
    myact = lrelu6
    myact = lambda x,alpha: tf.nn.relu(x)
    
    if architecture == 0:
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 8
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 8
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 16
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 16
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 32
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # # max pooling, then 1x1 conv
        # fcIn = tf.reduce_max(fcIn, axis=1)
        # fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        # out_channels_fc1 = 1024
        # h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)

        # fully connected layer
        fcIn = tf.reshape(fcIn,[batch_size,1,-1])
        out_channels_fc1 = 512
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

        # --- Encoded image ---

        dconv5 = myact(custom_lin(h_fc1, 9*14*out_channels_conv5), alpha)

        dconv5 = tf.reshape(dconv5, [batch_size, 9, 14, out_channels_conv5])
        # [9, 14]
        dconv5 = image_zeropadding(dconv5)
        # [11, 16]

        upsamp4 = image_upsampling(dconv5)
        # [22, 32]
        upsamp4 = image_zeropadding(upsamp4,1)
        # [24, 34]
        dconv4_1, _ = image_conv2d(upsamp4, out_channels_conv4, kernel_conv4, padding='SAME')
        upsamp4_2 = image_zeropadding(dconv4_1,1)
        # [26, 36]
        dconv4_2, _ = image_conv2d(upsamp4_2, out_channels_conv4, kernel_conv4, padding='SAME')

        upsamp3 = image_upsampling(dconv4_2)
        # [52, 72]
        upsamp3 = image_paddingright(upsamp3)
        upsamp3 = image_paddingbot(upsamp3)
        # [53, 73]

        upsamp3 = image_zeropadding(upsamp3)
        # [55, 75]
        dconv3_1, _ = image_conv2d(upsamp3, out_channels_conv3, kernel_conv3, padding='SAME')
        upsamp3_2 = image_zeropadding(dconv3_1,1)
        # [57, 77]
        dconv3_2, _ = image_conv2d(upsamp3_2, out_channels_conv3, kernel_conv3, padding='SAME')

        upsamp2 = image_upsampling(dconv3_2)
        # [114, 154]
        upsamp2 = image_zeropadding(upsamp2)
        # [116, 156]
        dconv2_1, _ = image_conv2d(upsamp2, out_channels_conv2, kernel_conv2, padding='SAME')
        upsamp2_2 = image_zeropadding(dconv2_1,1)
        # [118, 158]
        dconv2_2, _ = image_conv2d(upsamp2_2, out_channels_conv2, kernel_conv2, padding='SAME')

        upsamp1 = image_upsampling(dconv2_2)
        # [236, 316]
        upsamp1 = image_zeropadding(upsamp1)
        # [238, 318]
        dconv1_1, _ = image_conv2d(upsamp1, out_channels_conv1, kernel_conv1, padding='SAME')
        upsamp1_2 = image_zeropadding(dconv1_1,1)
        # [240, 320]

        final_conv, _ = image_conv2d(upsamp1_2, in_channels, kernel_conv1, padding='SAME')
        print("final_conv shape = "+str(final_conv.shape))
        final_conv = tf.sigmoid(final_conv)
        print("final_conv shape = "+str(final_conv.shape))
        return final_conv, h_fc1


    if architecture == 1:   # Like 0, with more weights
        padding_type = 'VALID'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0[:,1:-1,1:-1,:]
        mask0_3 = mask0_2[:,1:-1,1:-1,:]
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1[:,1:-1,1:-1,:]
        mask1_3 = mask1_2[:,1:-1,1:-1,:]
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2[:,1:-1,1:-1,:]
        mask2_3 = mask2_2[:,1:-1,1:-1,:]
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3[:,1:-1,1:-1,:]
        mask3_3 = mask3_2[:,1:-1,1:-1,:]
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 8
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 16
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 16
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 32
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 64
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # # max pooling, then 1x1 conv
        # fcIn = tf.reduce_max(fcIn, axis=1)
        # fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        # out_channels_fc1 = 1024
        # h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)

        # fully connected layer
        fcIn = tf.reshape(fcIn,[batch_size,1,-1])
        out_channels_fc1 = 1024
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

        # --- Encoded image ---

        dconv5 = myact(custom_lin(h_fc1, 9*14*out_channels_conv5), alpha)

        dconv5 = tf.reshape(dconv5, [batch_size, 9, 14, out_channels_conv5])
        # [9, 14]
        dconv5 = image_zeropadding(dconv5)
        # [11, 16]

        upsamp4 = image_upsampling(dconv5)
        # [22, 32]
        upsamp4 = image_zeropadding(upsamp4,1)
        # [24, 34]
        dconv4_1, _ = image_conv2d(upsamp4, out_channels_conv4, kernel_conv4, padding='SAME')
        upsamp4_2 = image_zeropadding(dconv4_1,1)
        # [26, 36]
        dconv4_2, _ = image_conv2d(upsamp4_2, out_channels_conv4, kernel_conv4, padding='SAME')

        upsamp3 = image_upsampling(dconv4_2)
        # [52, 72]
        upsamp3 = image_paddingright(upsamp3)
        upsamp3 = image_paddingbot(upsamp3)
        # [53, 73]

        upsamp3 = image_zeropadding(upsamp3)
        # [55, 75]
        dconv3_1, _ = image_conv2d(upsamp3, out_channels_conv3, kernel_conv3, padding='SAME')
        upsamp3_2 = image_zeropadding(dconv3_1,1)
        # [57, 77]
        dconv3_2, _ = image_conv2d(upsamp3_2, out_channels_conv3, kernel_conv3, padding='SAME')

        upsamp2 = image_upsampling(dconv3_2)
        # [114, 154]
        upsamp2 = image_zeropadding(upsamp2)
        # [116, 156]
        dconv2_1, _ = image_conv2d(upsamp2, out_channels_conv2, kernel_conv2, padding='SAME')
        upsamp2_2 = image_zeropadding(dconv2_1,1)
        # [118, 158]
        dconv2_2, _ = image_conv2d(upsamp2_2, out_channels_conv2, kernel_conv2, padding='SAME')

        upsamp1 = image_upsampling(dconv2_2)
        # [236, 316]
        upsamp1 = image_zeropadding(upsamp1)
        # [238, 318]
        dconv1_1, _ = image_conv2d(upsamp1, out_channels_conv1, kernel_conv1, padding='SAME')
        upsamp1_2 = image_zeropadding(dconv1_1,1)
        # [240, 320]

        final_conv, _ = image_conv2d(upsamp1_2, in_channels, kernel_conv1, padding='SAME')
        print("final_conv shape = "+str(final_conv.shape))
        final_conv = tf.sigmoid(final_conv)
        print("final_conv shape = "+str(final_conv.shape))
        return final_conv, h_fc1

    if architecture == 2:   # Like 0, with more weights, and SAME padding
        padding_type = 'SAME'

            # mask0 = tf.reduce_all(tf.equal(x,-1),axis=-1)
        mask0_bool = tf.reduce_any(tf.not_equal(x,-1), axis=-1)
        # [batch, width, height]
        mask0 = tf.where(mask0_bool, tf.ones([batch_size, width, height], dtype=tf.float32), tf.zeros([batch_size, width, height], dtype=tf.float32))
        mask0 = tf.expand_dims(mask0,axis=-1)
        # [batch, width, height, 1]
        mask0_2 = mask0
        mask0_3 = mask0_2
        mask1 = tf.nn.max_pool(mask0_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask1_2 = mask1
        mask1_3 = mask1_2
        mask2 = tf.nn.max_pool(mask1_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask2_2 = mask2
        mask2_3 = mask2_2
        mask3 = tf.nn.max_pool(mask2_3, [1,2,2,1],[1,2,2,1],padding=padding_type)
        mask3_2 = mask3
        mask3_3 = mask3_2
        mask4 = tf.nn.max_pool(mask3_3, [1,2,2,1],[1,2,2,1],padding=padding_type)


        out_channels_conv1 = 8
        kernel_conv1 = 3
        conv1, Wtest = image_conv2d(x, out_channels_conv1, kernel_conv1, mask0, padding=padding_type)
        conv1_act = myact(conv1,alpha)
        print("conv1_act shape = "+str(conv1_act.shape))

        conv1_2, _ = image_conv2d(conv1_act, out_channels_conv1, kernel_conv1, mask0_2, padding=padding_type)
        conv1_2_act = myact(conv1_2,alpha)

        # pool1 = tf.layers.max_pooling2d(inputs=conv1_2_act, pool_size=[2,2], strides=2)
        pool1 = tf.nn.max_pool(conv1_2_act, [1,2,2,1], [1,2,2,1], padding_type)

        out_channels_conv2 = 16
        kernel_conv2 = 3
        conv2_1, _ = image_conv2d(pool1, out_channels_conv2, kernel_conv2, mask1, padding=padding_type)
        conv2_1_act = myact(conv2_1,alpha)

        conv2_2, _ = image_conv2d(conv2_1_act, out_channels_conv2, kernel_conv2, mask1_2, padding=padding_type)
        conv2_2_act = myact(conv2_2,alpha)
        print("conv2_2_act shape = "+str(conv2_2_act.shape))

        pool2 = tf.layers.max_pooling2d(inputs=conv2_2_act, pool_size=[2,2], strides=2)

        out_channels_conv3 = 16
        kernel_conv3 = 3
        conv3_1, _ = image_conv2d(pool2, out_channels_conv3, kernel_conv3, mask2, padding=padding_type)
        conv3_1_act = myact(conv3_1,alpha)

        conv3_2, _ = image_conv2d(conv3_1_act, out_channels_conv3, kernel_conv2, mask2_2, padding=padding_type)
        conv3_2_act = myact(conv3_2,alpha)
        print("conv3_2_act shape = "+str(conv3_2_act.shape))

        pool3 = tf.layers.max_pooling2d(inputs=conv3_2_act, pool_size=[2,2], strides=2)

        out_channels_conv4 = 32
        kernel_conv4 = 3
        conv4_1, _ = image_conv2d(pool3, out_channels_conv4, kernel_conv4, mask3, padding=padding_type)
        conv4_1_act = myact(conv4_1,alpha)

        conv4_2, _ = image_conv2d(conv4_1_act, out_channels_conv4, kernel_conv4, mask3_2, padding=padding_type)
        conv4_2_act = myact(conv4_2,alpha)
        print("conv4_2_act shape = "+str(conv4_2_act.shape))

        pool4 = tf.layers.max_pooling2d(inputs=conv4_2_act, pool_size=[2,2], strides=2)

        out_channels_conv5 = 64
        kernel_conv5 = 3
        conv5, _ = image_conv2d(pool4, out_channels_conv5, kernel_conv5, mask4, padding=padding_type)
        conv5_act = myact(conv5,alpha)
        print("conv5_act shape = "+str(conv5_act.shape))

        # fcIn = tf.reduce_max(tf.reduce_max(conv5_act,axis=1),axis=1)
        fcIn = conv5_act
        
        # # max pooling, then 1x1 conv
        # fcIn = tf.reduce_max(fcIn, axis=1)
        # fcIn = tf.reduce_max(fcIn, axis=1, keep_dims=True)
        # out_channels_fc1 = 1024
        # h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)

        # fully connected layer
        fcIn = tf.reshape(fcIn,[batch_size,1,-1])
        out_channels_fc1 = 1024
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)
        print("h_fc1 shape = "+str(h_fc1.shape))

        # --- Encoded image ---

        dconv5 = myact(custom_lin(h_fc1, 15*20*out_channels_conv5), alpha)

        dconv5 = tf.reshape(dconv5, [batch_size, 15, 20, out_channels_conv5])
        # [15, 20]

        upsamp4 = image_upsampling(dconv5)
        # [30, 40]
        dconv4_1, _ = image_conv2d(upsamp4, out_channels_conv4, kernel_conv4, padding='SAME')
        upsamp4_2 = dconv4_1
        # [30, 40]
        dconv4_2, _ = image_conv2d(upsamp4_2, out_channels_conv4, kernel_conv4, padding='SAME')

        upsamp3 = image_upsampling(dconv4_2)
        # [60, 80]

        dconv3_1, _ = image_conv2d(upsamp3, out_channels_conv3, kernel_conv3, padding='SAME')
        upsamp3_2 = dconv3_1
        # [60, 80]
        dconv3_2, _ = image_conv2d(upsamp3_2, out_channels_conv3, kernel_conv3, padding='SAME')

        upsamp2 = image_upsampling(dconv3_2)
        # [120, 160]
        dconv2_1, _ = image_conv2d(upsamp2, out_channels_conv2, kernel_conv2, padding='SAME')
        upsamp2_2 = dconv2_1
        # [120, 160]
        dconv2_2, _ = image_conv2d(upsamp2_2, out_channels_conv2, kernel_conv2, padding='SAME')

        upsamp1 = image_upsampling(dconv2_2)
        # [240, 320]

        dconv1_1, _ = image_conv2d(upsamp1, out_channels_conv1, kernel_conv1, padding='SAME')
        upsamp1_2 = dconv1_1
        # [240, 320]

        final_conv, _ = image_conv2d(upsamp1_2, in_channels, kernel_conv1, padding='SAME')
        print("final_conv shape = "+str(final_conv.shape))
        final_conv = tf.sigmoid(final_conv)
        print("final_conv shape = "+str(final_conv.shape))
        return final_conv, h_fc1


def get_encoded_reg(code, architecture):

    batch_size, _, code_len = code.get_shape().as_list()

    if architecture==0:         # Very simple, no hidden layer
        out_ch = 1
        finalFeature = custom_lin(code, out_ch)
        # [batch, 1, 1]
        finalOut = tf.sigmoid(finalFeature)

        return tf.reshape(finalOut, [batch_size, 1])

    elif architecture==1:       # One hidden layer

        out_ch = 1
        ch1 = 512
        hf1 = custom_lin(code, ch1)

        finalFeature = custom_lin(hf1, out_ch)
        # [batch, 1, 1]
        finalOut = tf.sigmoid(finalFeature)

        return tf.reshape(finalOut, [batch_size, 1])


def get_mesh_encoder(x, adjs, architecture, keep_prob, mode='shape'):
    bTransInvariant = False
    bRotInvariant = False
    alpha = 0.1
    if architecture == 0:      # copied from above, not complete
        coarsening_steps = 3
        batch_size,_,in_channels = x.get_shape().as_list()

        print("in_channels = "+str(in_channels))

        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos3 = custom_binary_tree_pooling(pos2, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        xUV = tf.slice(x,[0,0,0],[-1,-1,2])

        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        print("xUV shape = "+str(xUV.shape))
        print("pos0 shape = "+str(pos0.shape))
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        adjs[0] = filterAdj(xUV, adjs[0], -3)
        print("new adjs0 shape = "+str(adjs[0].shape))

        adjs[1] = filterAdj(xUV1, adjs[1], -3)
        adjs[2] = filterAdj(xUV2, adjs[2], -3)
        adjs[3] = filterAdj(xUV3, adjs[3], -3)

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 5
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu6(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 10
        h_conv2, adj_size2 = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = lrelu6(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 15
        h_conv3, adj_size3 = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = lrelu6(h_conv3,alpha)
        # [batch, N/16, 256]

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 32
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        h_conv4_act = lrelu6(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=True, rotation_invariance=False)
        dconv4_act = lrelu6(dconv4,alpha)
        # [batch, N/16, 256]


        # fully connected layer
        fcIn = tf.reshape(dconv4_act,(batch_size,1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

    if architecture == 1:      # like 0 (if it was finished) w/ one less coarsening step
        myact = lrelu
        coarsening_steps = 3
        batch_size,_,in_channels = x.get_shape().as_list()

        print("in_channels = "+str(in_channels))

        pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos3 = custom_binary_tree_pooling(pos2, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        xUV = tf.slice(x,[0,0,0],[-1,-1,2])

        xUV1 = custom_binary_tree_pooling(xUV, steps=coarsening_steps, pooltype='max')
        xUV2 = custom_binary_tree_pooling(xUV1, steps=coarsening_steps, pooltype='max')
        xUV3 = custom_binary_tree_pooling(xUV2, steps=coarsening_steps, pooltype='max')

        print("xUV shape = "+str(xUV.shape))
        print("pos0 shape = "+str(pos0.shape))
        newX = tf.concat((xUV,pos0),axis=-1)
        print("newX shape = "+str(newX.shape))


        # adjs[0] = filterAdj(xUV, adjs[0], -3)
        # print("new adjs0 shape = "+str(adjs[0].shape))

        # adjs[1] = filterAdj(xUV1, adjs[1], -3)
        # adjs[2] = filterAdj(xUV2, adjs[2], -3)
        # adjs[3] = filterAdj(xUV3, adjs[3], -3)

        filterAdj0 = filterAdj(xUV, adjs[0], -3)
        filterAdj1 = filterAdj(xUV, adjs[1], -3)
        filterAdj2 = filterAdj(xUV, adjs[2], -3)
        filterAdj3 = filterAdj(xUV, adjs[3], -3)

        for adjnum in range(len(adjs)):
            adjs[adjnum] = tf.tile(adjs[adjnum], [batch_size,1,1])

        # for i in range(1,4):
        #     adjs[i] = tf.tile(adjs[i],[batch_size,1,1])

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 6
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, adj_size = custom_conv2d(newX, filterAdj0, out_channels_conv1, M_conv1, translation_invariance=True, rotation_invariance=bRotInvariant)
        h_conv1_act = myact(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 12
        h_conv2, adj_size2 = custom_conv2d(pool1,filterAdj1, out_channels_conv2, M_conv2,translation_invariance=True, rotation_invariance=False)
        h_conv2_act = myact(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]
        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 18
        h_conv3, adj_size3 = custom_conv2d(pool2, filterAdj2, out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        h_conv3_act = myact(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        # One extra conv
        #DeConv4
        hconv3_2, _ = custom_conv2d(h_conv3_act, filterAdj2, out_channels_conv3, M_conv3,translation_invariance=True, rotation_invariance=False)
        hconv3_2_act = myact(hconv3_2,alpha)
        # [batch, N/16, 256]


        # fully connected layer
        fcIn = tf.reshape(hconv3_2_act,(batch_size,1,-1))

        out_channels_fc1 = 512
        h_fc1 = myact(custom_lin(fcIn, out_channels_fc1), alpha)

        dconv3 = myact(custom_lin(h_fc1, 512*out_channels_conv3), alpha)

        dconv3 = tf.reshape(dconv3, [batch_size, 512, out_channels_conv3])

        dconv3_2, _ = custom_conv2d(dconv3, adjs[2], out_channels_conv3, M_conv3, translation_invariance=True, biasMask=False)

        upsamp2 = custom_upsampling(dconv3_2, coarsening_steps)

        dconv2, _ = custom_conv2d(upsamp2, adjs[1], out_channels_conv2, M_conv2, translation_invariance=True, biasMask=False)
        dconv2_act = myact(dconv2, alpha)

        upsamp1 = custom_upsampling(dconv2_act, coarsening_steps)

        dconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1, translation_invariance=True, biasMask=False)
        dconv1_act = myact(dconv1, alpha)

        final_conv, _ = custom_conv2d(dconv1_act, adjs[0], 2, M_conv1, translation_invariance=True, biasMask=False)

        return tf.sigmoid(final_conv), h_fc1
# 
        # return tf.sigmoid(final_conv), [adj_size, newX, h_conv1]

# End of file