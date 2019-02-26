from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
from utils import *
#import h5py

random_seed=0
std_dev= 0.01 #0.05


# Levi-Civita tensor of dimension 3
LC_tensor = tf.constant([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]],dtype=tf.float32)

Id_tensor = tf.constant([0,0,1,0,1,0,1,0,0],shape=[3,3], dtype=tf.float32)

ref_axis = tf.constant([0,0,1],dtype=tf.float32)

def broadcast(tensor, shape):
    return tensor + tf.zeros(shape, dtype=tensor.dtype)

# CHECK
def weight_variable(shape):
        initial = tf.random_normal(shape, stddev=std_dev)
        #initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
        return tf.Variable(initial, name="weight")

# CHECK
def bias_variable(shape):
        initial = tf.random_normal(shape, stddev=std_dev)
        #initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
        return tf.Variable(initial, name="bias")

# CHECK
def assignment_variable(shape):
        initial = tf.random_normal(shape, stddev=std_dev)
        #initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
        return tf.Variable(initial, name="assignment")

# CHECK
def reusable_weight_variable(shape, name="weight"):
        initial = tf.random_normal_initializer(stddev=std_dev)
        #initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
        return tf.get_variable(name, shape=shape, initializer=initial)

# CHECK
def reusable_bias_variable(shape, name="bias"):
        initial = tf.random_normal_initializer(stddev=std_dev)
        #initial = tf.truncated_normal(shape, stddev=0.1, seed=random_seed)
        return tf.get_variable(name, shape=shape, initializer=initial)

# CHECK
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

# CHECK
def get_weight_assigments(x, adj, u, v, c):
    in_channels, num_points = x.get_shape().as_list()
    num_points, K = adj.get_shape().as_list()
    M, in_channels = u.get_shape().as_list()
    # [M, N]
    ux = tf.matmul(u, x)
    vx = tf.matmul(v, x)
    # [N, M]
    vx = tf.transpose(vx, [1, 0])
    # [N, K, M]
    patches = get_patches(vx, adj)
    # [K, M, N]
    patches = tf.transpose(patches, [1, 2, 0])
    # [K, M, N]
    patches = tf.add(ux, patches)
    # [K, N, M]
    patches = tf.transpose(patches, [0, 2, 1])
    patches = tf.add(patches, c)
    # [N, K, M]
    patches = tf.transpose(patches, [1, 0, 2])
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

# CHECK
def get_slices(x, adj):     #adj is one-indexed
        num_points, in_channels = x.get_shape().as_list()
        input_size, K = adj.get_shape().as_list()
        zeros = tf.zeros([1, in_channels], dtype=tf.float32)
        x = tf.concat([zeros, x], axis=0)
        slices = tf.gather(x, adj)
        # slices = tf.reshape(slices, [-1, K, in_channels])
        return slices

# CHECK
def get_patches(x, adj):
        patches = get_slices(x, adj)
        return patches

# CHECK
def custom_conv2d(x, adj, out_channels, M, translation_invariance=False, rotation_invariance=False):
        
        input_size, in_channels = x.get_shape().as_list()
        W0 = weight_variable([M, out_channels, in_channels])
        b = bias_variable([out_channels])
        u = assignment_variable([M, in_channels])
        c = assignment_variable([M])
        input_size, K = adj.get_shape().as_list()
        # Calculate neighbourhood size for each input - [input_size, neighbours]
        adj_size = tf.count_nonzero(adj, axis=-1)
        #deal with unconnected points: replace NaN with 0
        non_zeros = tf.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
        # [input_size, 1, 1]
        adj_size = tf.reshape(adj_size, [-1, 1, 1])
        

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

        # [in_channels, input_size]
        x = tf.transpose(x, [1, 0])
        W = tf.reshape(W0, [M*out_channels, in_channels])
        # Multiple w and x -> [M*out_channels, input_size]
        wx = tf.matmul(W, x)
        # Reshape and transpose wx into [input_size, M*out_channels]
        wx = tf.transpose(wx, [1, 0])
        # Get patches from wx - [input_size, K(neighbours-here input_size), M*out_channels]
        patches = get_patches(wx, adj)
        # [input_size, K, M]

        if (translation_invariance == False) and (rotation_invariance == False):
            q = get_weight_assigments(x, adj, u, v, c)
            # Element wise multiplication of q and patches for each input -- [input_size, K, M, out]
        else:
            pass

        patches = tf.reshape(patches, [-1, K, M, out_channels])
        # [out, input_size, K, M]
        patches = tf.transpose(patches, [3, 0, 1, 2])
        patches = tf.multiply(q, patches)
        patches = tf.transpose(patches, [1, 2, 3, 0])
        # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [input_size, M, out]
        patches = tf.reduce_sum(patches, axis=1)
        patches = tf.multiply(adj_size, patches)
        # Add add elements for all m
        patches = tf.reduce_sum(patches, axis=1)
        # [input_size, out]
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


# CHECK
def custom_conv2d_pos_for_assignment(x, adj, out_channels, M, translation_invariance=False, rotation_invariance=False):
        
        input_size, in_channels_ass = x.get_shape().as_list()
        in_channels_weights = in_channels_ass - 3
        #in_channels_ass = 6
        xn = tf.slice(x,[0,0],[-1,in_channels_weights])    # take normals only
        W0 = weight_variable([M, out_channels, in_channels_weights])
        b = bias_variable([out_channels])
        u = assignment_variable([M, in_channels_ass])
        c = assignment_variable([M])
        input_size, K = adj.get_shape().as_list()
        # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
        adj_size = tf.count_nonzero(adj, axis=-1)
        #deal with unconnected points: replace NaN with 0
        non_zeros = tf.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
        # [input_size, 1, 1]
        adj_size = tf.reshape(adj_size, [-1, 1, 1])
        

        # Make new assignement, that is translation invariant wrt position, but not normals
        vn = assignment_variable([M, in_channels_weights])
        up = tf.slice(u,[0,in_channels_weights],[-1,-1])
        vp = -up
        v = tf.concat([vn,vp],axis=-1)


        # [in_channels, input_size]
        x = tf.transpose(x, [1, 0])
        xn = tf.transpose(xn, [1, 0])
        W = tf.reshape(W0, [M*out_channels, in_channels_weights])
        # Multiple w and x -> [M*out_channels, input_size]
        wx = tf.matmul(W, x)
        # Reshape and transpose wx into [input_size, M*out_channels]
        wx = tf.transpose(wx, [1, 0])
        # Get patches from wx - [input_size, K(neighbours-here input_size), M*out_channels]
        patches = get_patches(wx, adj)
        # [batch_size, input_size, K, M]

        q = get_weight_assigments(x, adj, u, v, c)
        # Element wise multiplication of q and patches for each input -- [input_size, K, M, out]

        patches = tf.reshape(patches, [-1, K, M, out_channels])
        # [out, input_size, K, M]
        patches = tf.transpose(patches, [3, 0, 1, 2])
        patches = tf.multiply(q, patches)
        patches = tf.transpose(patches, [1, 2, 3, 0])
        # Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [input_size, M, out]
        patches = tf.reduce_sum(patches, axis=2)
        patches = tf.multiply(adj_size, patches)
        # Add add elements for all m
        patches = tf.reduce_sum(patches, axis=2)
        # [input_size, out]
        patches = patches + b
        return patches, W0

        


# CHECK
def custom_lin(input, out_channels):
        input_size, in_channels = input.get_shape().as_list()

        W = weight_variable([in_channels, out_channels])
        b = bias_variable([out_channels])
        return tf.matmul(input, W) + b
        # return tf.map_fn(lambda x: tf.matmul(x, W), input) + b



# CHECK
def custom_binary_tree_pooling(x, steps=1, pooltype='max'):

    input_size, channels = x.get_shape().as_list()
    print("pooling...")
    print("input size = "+str(input_size))
    print("channels = "+str(channels))

    # Pairs of nodes should already be grouped together
    if pooltype=='max':
        x = tf.reshape(x,[-1,int(math.pow(2,steps)),channels])
        outputs = tf.reduce_max(x,axis=1)
    elif pooltype=='avg':
        x = tf.reshape(x,[-1,int(math.pow(2,steps)),channels])
        outputs = tf.reduce_mean(x,axis=1)
    elif pooltype=='avg_ignore_zeros':
        px = x
        for step in range(steps):
            px = tf.reshape(px,[-1,2,channels])
            print("px shape = "+str(px.shape))
            line0 = tf.slice(px,[0,0,0],[-1,1,-1])
            line1 = tf.slice(px,[0,1,0],[-1,-1,-1])
            print("line0 shape = "+str(line0.shape))

            z0 = tf.equal(line0,0)
            z1 = tf.equal(line1,0)

            z0 = tf.reduce_all(z0,axis=-1,keep_dims=True)
            z1 = tf.reduce_all(z1,axis=-1,keep_dims=True)
            print("z0 shape = "+str(z0.shape))
            z0 = tf.tile(z0,[1,1,channels])
            z1 = tf.tile(z1,[1,1,channels])
            print("z0 shape = "+str(z0.shape))
            cline0 = tf.where(z0,line1,line0)
            cline1 = tf.where(z1,line0,line1)
            print("cline0 shape = "+str(cline0.shape))
            cx = tf.concat([cline0,cline1],axis=1)
            print("cx shape = "+str(cx.shape))
            px = tf.reduce_mean(cx,axis=1)
            print("px shape = "+str(px.shape))
        outputs = px
    return outputs


# CHECK
def custom_upsampling(x, steps=1):
    input_size, channels = x.get_shape().as_list()

    x = tf.expand_dims(x,axis=1)
    outputs = tf.tile(x,[1,int(math.pow(2,steps)),1])
    outputs = tf.reshape(outputs,[-1,channels])

    return outputs

# CHECK
def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



# CHECK
def get_model_gender_class(x, adjs, architecture, keep_prob):
    bTransInvariant = False
    bRotInvariant = False

    if architecture == 0:      # copied from architecture 14 above, one extra coarsening layer
        alpha = 0.1
        coarsening_steps = 3
        _,in_channels = x.get_shape().as_list()

        print("in_channels = "+str(in_channels))

        pos0 = tf.slice(x,[0,in_channels-3],[-1,-1])
        pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        pos3 = custom_binary_tree_pooling(pos2, steps=coarsening_steps, pooltype='avg_ignore_zeros')


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
        fcIn = tf.reshape(dconv4_act,(1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

        
        # Lin(3)
        out_channels_reg = 1
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.sigmoid(finalFeature)

        return finalProba

    if architecture == 1:      # like 0, w/o position for assignment
        alpha = 0.0 
        coarsening_steps = 3
        _,in_channels = x.get_shape().as_list()

        print("in_channels = "+str(in_channels))

        # pos0 = tf.slice(x,[0,in_channels-3],[-1,-1])
        # pos1 = custom_binary_tree_pooling(pos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos2 = custom_binary_tree_pooling(pos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
        # pos3 = custom_binary_tree_pooling(pos2, steps=coarsening_steps, pooltype='avg_ignore_zeros')


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
        fcIn = tf.reshape(dconv4_act,(1,-1))

        out_channels_fc1 = 1024
        h_fc1 = lrelu(custom_lin(fcIn, out_channels_fc1), alpha)

        
        # Lin(3)
        out_channels_reg = 1
        finalFeature = custom_lin(h_fc1, out_channels_reg)
        # Logistic function
        finalProba = tf.sigmoid(finalFeature)

        return finalProba