from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
from utils import *
from settings import *
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
    # with tf.variable_scope('Conv'):
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
        

# BiasMask is not intuitive:
# We actually discriminate nodes with no connection to the graph, instead of input equal to 0
# The adjacency matrix is filtered upstream based on input, thanks to the function (filterAdj).
def reusable_custom_conv2d(x, adj, out_channels, M, biasMask = True, translation_invariance=False, name="conv2d"):
    

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

    if biasMask:
        # NEW!! Add bias only to nodes with at least one active node in neighbourhood
        patches = tf.where(tf.tile(tf.expand_dims(non_zeros,axis=-1),[1,1,out_channels]), patches +b, patches)
    else:
        patches = patches + b

    # patches = patches + b
    return patches

def cycle_conv2d(x, adj, out_channels, M, biasMask=True, translation_invariance=False, rotation_invariance=False, name="conv2d"):    
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

    # ---
    # This is where the trick happens:
    # Roll tensors to align cycles on columns
    # Then, sum over columns and maxpool over rows
    # ---
    sliceList = []
    for neighbourInd in range(K):   # I don't know any better than to loop and roll each row independently...
        patchSlice = patches[:,:,neighbourInd,:]
        newSlice = tf.manip.roll(patchSlice,-neighbourInd,axis=-1)
        sliceList.append(newSlice)
    patches = tf.stack(sliceList,axis=2)


    # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
    patches = tf.reduce_sum(patches, axis=2)
    patches = tf.multiply(adj_size, patches)

    # Now, (soft)max pool over all possible rotations instead of summing
    # patches = tf.reduce_sum(patches, axis=2)
    patches = softmaxPooling(patches, axis=2)
    # [batch_size, input_size, out]
    # print("Your patches shape is "+str(patches.shape))
    if biasMask:
        # NEW!! Add bias only to nodes with at least one active node in neighbourhood
        patches = tf.where(tf.tile(tf.expand_dims(non_zeros,axis=-1),[1,1,out_channels]), patches +b, patches)
    else:
        patches = patches + b

    return patches








def custom_lin(input, out_channels):
    with tf.variable_scope('MLP'):
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
    with tf.variable_scope('Pooling'):
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
    with tf.variable_scope('Upsampling'):
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
        zeroNodes = tf.reduce_all(tf.equal(xb,zeroValue),axis=-1)
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
    with tf.variable_scope('lReLU'):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def lrelu6(x, alpha):
  return tf.nn.relu6(x) - alpha * tf.nn.relu6(-x)


def softmaxPooling(x,axis=-1):

    # softmax (no axis support in my tf version)
    # weights = tf.math.softmax(x,axis=dim)
    weights = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=axis, keepdims=True)
    weighted_x = tf.multiply(x,weights)
    return tf.reduce_sum(weighted_x,axis=axis)



# For this function, we give a pyramid of adjacency matrix, from detailed to coarse
# (This is used for the pooling layers)
# Edge weights????

def get_model_reg_multi_scale(x, adjs, architecture, keep_prob, coarsening_steps=COARSENING_STEPS):
    """ 
    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
    bTransInvariant = False
    bRotInvariant = False
    alpha = 0.2

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

    
    if architecture == 15:      # Like 9, but multi-scale estimation
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

    if architecture == 22:      # Like 10, with normals + pos as input
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
        # pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        with tf.variable_scope('Level0'):
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

        with tf.variable_scope('Level1'):
            # Conv2
            M_conv2 = 9
            out_channels_conv2 = 64
            h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv2_act = lrelu(h_conv2,alpha)
            # [batch, N/4, 128]

            # Pooling 2
            pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
            # [batch, N/16, 128]

        with tf.variable_scope('Level2'):
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
        with tf.variable_scope('Level1'):
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
        with tf.variable_scope('Level0'):
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

    if architecture == 23:       # Like 9, with only pos for assignment (and translation invariance)
        bTransInvariant = True
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
        h_conv1, _ = custom_conv2d_only_pos_for_assignment(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
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
        h_conv2, _ = custom_conv2d_only_pos_for_assignment(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
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
        h_conv3, _ = custom_conv2d_only_pos_for_assignment(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = lrelu(h_conv3,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv3_act = tf.concat([h_conv3_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_only_pos_for_assignment(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = lrelu(dconv3,alpha)
        # [batch, N/16, 256]

        #Upsampling2
        upsamp2 = custom_upsampling(dconv3_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upsamp2 = tf.concat([upsamp2,pos1],axis=-1)
        upconv2, _ = custom_conv2d_only_pos_for_assignment(upsamp2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
        # [batch, N/4, 256]
        # Add position:
        concat2 = tf.concat([concat2,pos1],axis=-1)
        dconv2, _ = custom_conv2d_only_pos_for_assignment(concat2, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv2_act = lrelu(dconv2,alpha)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upsamp1 = tf.concat([upsamp1,pos0],axis=-1)
        upconv1, _ = custom_conv2d_only_pos_for_assignment(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        concat1 = tf.concat([concat1,pos0],axis=-1)
        dconv1, _ = custom_conv2d_only_pos_for_assignment(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = lrelu(dconv1,alpha)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
        
        # Lin(3)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv


    if architecture == 24:       # Like 9, w/ 1 extra layer in the middle
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


        # Conv4
        # Add position:
        h_conv3_act_pos = tf.concat([h_conv3_act,pos2],axis=-1)
        h_conv4, _ = custom_conv2d_pos_for_assignment(h_conv3_act_pos, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        # Add position:
        h_conv4_act = tf.concat([h_conv4_act,pos2],axis=-1)
        dconv3, _ = custom_conv2d_pos_for_assignment(h_conv4_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
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

    if architecture == 25:      # Like 22, with multi-scale estimation
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        out_channels_reg = 3
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
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

        # Lin(1024)
        out_channels_fc2 = 1024
        h_fc2 = lrelu(custom_lin(dconv3_act, out_channels_fc2),alpha)
        # Lin(3)
        y_conv2 = custom_lin(h_fc2, out_channels_reg)

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

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(dconv2_act, out_channels_fc1),alpha)
        # Lin(3)
        y_conv1 = custom_lin(h_fc1, out_channels_reg)

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
        
        y_conv0 = custom_lin(h_fc1, out_channels_reg)
        return y_conv0, y_conv1 ,y_conv2

    if architecture == 26:      # Like 22, with one extra pooling/unpooling scale!
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
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

        # Pooling 3
        pool3 = custom_binary_tree_pooling(h_conv3_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv4
        M_conv4 = 9
        out_channels_conv4 = 256
        h_conv4, _ = custom_conv2d(pool3, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv4_act = lrelu(h_conv4,alpha)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        #Upsampling3
        upsamp3 = custom_upsampling(dconv4_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upconv3, _ = custom_conv2d(upsamp3, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat3 = tf.concat([upconv3, h_conv3_act], axis=-1)

        #DeConv3
        dconv3, _ = custom_conv2d(concat3, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
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

    if architecture == 27:      # Like 26, w/ less weights
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
        # pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 16
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = lrelu(h_conv1,alpha)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 32
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = lrelu(h_conv2,alpha)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 64
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
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

        #DeConv4
        dconv4, _ = custom_conv2d(h_conv4_act, adjs[3], out_channels_conv4, M_conv4,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv4_act = lrelu(dconv4,alpha)
        # [batch, N/16, 256]

        #Upsampling3
        upsamp3 = custom_upsampling(dconv4_act, steps=coarsening_steps)
        # [batch, N/4, 256]

        upconv3, _ = custom_conv2d(upsamp3, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N/4, 128]

        #DeConv2
        concat3 = tf.concat([upconv3, h_conv3_act], axis=-1)

        #DeConv3
        dconv3, _ = custom_conv2d(concat3, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
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

    if architecture == 28:      # Like 22, with eLu
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
        # pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

        # Conv1
        M_conv1 = 9
        out_channels_conv1 = 32
        #h_conv1, _ = custom_conv2d_norm_pos(x, adj, out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=True)
        h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
        h_conv1_act = tf.nn.elu(h_conv1)
        # [batch, N, 64]

        # Pooling 1
        pool1 = custom_binary_tree_pooling(h_conv1_act, steps=coarsening_steps) # TODO: deal with fake nodes??
        # [batch, N/4, 64]

        # Conv2
        M_conv2 = 9
        out_channels_conv2 = 64
        h_conv2, _ = custom_conv2d(pool1, adjs[1], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv2_act = tf.nn.elu(h_conv2)
        # [batch, N/4, 128]

        # Pooling 2
        pool2 = custom_binary_tree_pooling(h_conv2_act, steps=coarsening_steps)
        # [batch, N/16, 128]

        # Conv3
        M_conv3 = 9
        out_channels_conv3 = 128
        h_conv3, _ = custom_conv2d(pool2, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        h_conv3_act = tf.nn.elu(h_conv3)
        # [batch, N/16, 256]

        # --- Central features ---

        #DeConv3
        dconv3, _ = custom_conv2d(h_conv3_act, adjs[2], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv3_act = tf.nn.elu(dconv3)
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
        dconv2_act = tf.nn.elu(dconv2)
        # [batch, N/4, 128]

        #Upsampling1
        upsamp1 = custom_upsampling(dconv2_act, steps=coarsening_steps)
        # [batch, N, 128]
        upconv1, _ = custom_conv2d(upsamp1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        # [batch, N, 64]

        concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
        # [batch, N, 128]

        dconv1, _ = custom_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
        dconv1_act = tf.nn.elu(dconv1)
        # [batch, N, 64]

        # Lin(1024)
        out_channels_fc1 = 1024
        h_fc1 = tf.nn.elu(custom_lin(dconv1_act, out_channels_fc1))
        
        # Lin(3)
        out_channels_reg = 3
        y_conv = custom_lin(h_fc1, out_channels_reg)
        return y_conv


    if architecture == 29:      # Like 22, without pooling/upsampling
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
        # pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        with tf.variable_scope('Level0'):
            # Conv1
            M_conv1 = 9
            out_channels_conv1 = 32
            h_conv1, _ = custom_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
            h_conv1_act = lrelu(h_conv1,alpha)
            # [batch, N, 64]

        with tf.variable_scope('Level1'):
            # Conv2
            M_conv2 = 9
            out_channels_conv2 = 64
            h_conv2, _ = custom_conv2d(h_conv1_act, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv2_act = lrelu(h_conv2,alpha)
            # [batch, N/4, 128]

        with tf.variable_scope('Level2'):
            # Conv3
            M_conv3 = 9
            out_channels_conv3 = 128
            h_conv3, _ = custom_conv2d(h_conv2_act, adjs[0], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv3_act = lrelu(h_conv3,alpha)
            # [batch, N/16, 256]

            # --- Central features ---

            #DeConv3
            dconv3, _ = custom_conv2d(h_conv3_act, adjs[0], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
            dconv3_act = lrelu(dconv3,alpha)
            # [batch, N/16, 256]

        with tf.variable_scope('Level1'):
            upconv2, _ = custom_conv2d(dconv3_act, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            # [batch, N/4, 128]

            #DeConv2
            concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
            # [batch, N/4, 256]
            dconv2, _ = custom_conv2d(concat2, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            dconv2_act = lrelu(dconv2,alpha)
            # [batch, N/4, 128]

        with tf.variable_scope('Level0'):
            upconv1, _ = custom_conv2d(dconv2_act, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
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

    if architecture == 30:      # Like 29, with cycle conv
        alpha = 0.1
        coarsening_steps = 2
        _, _,in_channels = x.get_shape().as_list()
        # x = tf.slice(x,[0,0,0],[-1,-1,3])
        # pos0 = tf.slice(x,[0,0,in_channels-3],[-1,-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        with tf.variable_scope('Level0'):
            # Conv1
            M_conv1 = 9
            out_channels_conv1 = 32
            h_conv1 = cycle_conv2d(x, adjs[0], out_channels_conv1, M_conv1, translation_invariance=bTransInvariant, rotation_invariance=bRotInvariant)
            h_conv1_act = lrelu(h_conv1,alpha)
            # [batch, N, 64]

        with tf.variable_scope('Level1'):
            # Conv2
            M_conv2 = 9
            out_channels_conv2 = 64
            h_conv2 = cycle_conv2d(h_conv1_act, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
            h_conv2_act = lrelu(h_conv2,alpha)
            # [batch, N/4, 128]

        with tf.variable_scope('Level2'):
            with tf.variable_scope('conv', reuse=False):
                # Conv3
                M_conv3 = 9
                out_channels_conv3 = 128
                h_conv3 = cycle_conv2d(h_conv2_act, adjs[0], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
                h_conv3_act = lrelu(h_conv3,alpha)
                # [batch, N/16, 256]

            # --- Central features ---
            with tf.variable_scope('dconv', reuse=False):
                #DeConv3
                dconv3 = cycle_conv2d(h_conv3_act, adjs[0], out_channels_conv3, M_conv3,translation_invariance=bTransInvariant, rotation_invariance=False)
                dconv3_act = lrelu(dconv3,alpha)
                # [batch, N/16, 256]

        with tf.variable_scope('Level1'):
            with tf.variable_scope('dconv1', reuse=False):
                upconv2 = cycle_conv2d(dconv3_act, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
                # [batch, N/4, 128]

            #DeConv2
            concat2 = tf.concat([upconv2, h_conv2_act], axis=-1)
            # [batch, N/4, 256]
            with tf.variable_scope('dconv2', reuse=False):
                dconv2 = cycle_conv2d(concat2, adjs[0], out_channels_conv2, M_conv2,translation_invariance=bTransInvariant, rotation_invariance=False)
                dconv2_act = lrelu(dconv2,alpha)
                # [batch, N/4, 128]

        with tf.variable_scope('Level0'):
            with tf.variable_scope('dconv1', reuse=False):
                upconv1 = cycle_conv2d(dconv2_act, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
                # [batch, N, 64]

            concat1 = tf.concat([upconv1, h_conv1_act], axis=-1)
            # [batch, N, 128]
            with tf.variable_scope('dconv2', reuse=False):
                dconv1 = cycle_conv2d(concat1, adjs[0], out_channels_conv1, M_conv1,translation_invariance=bTransInvariant, rotation_invariance=False)
                dconv1_act = lrelu(dconv1,alpha)
                # [batch, N, 64]

            # Lin(1024)
            out_channels_fc1 = 1024
            h_fc1 = lrelu(custom_lin(dconv1_act, out_channels_fc1),alpha)
            
            # Lin(3)
            out_channels_reg = 3
            y_conv = custom_lin(h_fc1, out_channels_reg)
            return y_conv





# End of file