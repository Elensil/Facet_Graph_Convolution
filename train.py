from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
#import h5py
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
try:
    import scipy.io
except:
    print("scipy not found")
from model import *
from utils import *
from tensorflow.python import debug as tf_debug
import random
from lib.coarsening import *
from settings import *
from trainingSet import *

TF_VERSION = int(tf.__version__[0])
if TF_VERSION==2:
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

def inferNetOld(inputMesh):
    in_points = inputMesh.vertices
    f_normals = inputMesh.in_list
    f_adj = inputMesh.adj_list
    edge_map = inputMesh.edge_map
    v_e_map = inputMesh.v_e_map
    num_wofake_nodes = inputMesh.num_faces
    patch_indices = inputMesh.patch_indices
    old_to_new_permutations = inputMesh.permutations
    with tf.Graph().as_default():

        sess = tf.InteractiveSession()
        if(DEBUG):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)


        BATCH_SIZE=f_normals[0].shape[0]
        NUM_POINTS=in_points.shape[1]
        print("v_e_map shape = ",v_e_map.shape)
        MAX_EDGES = v_e_map.shape[2]
        NUM_EDGES = edge_map.shape[1]
        K_faces = f_adj[0][0].shape[2]
        NUM_IN_CHANNELS = f_normals[0].shape[2]
        patchNumber = len(f_normals)

        xp_ = tf.placeholder('float32', shape=(BATCH_SIZE, NUM_POINTS,3),name='xp_')

        fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')

        fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
        fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
        fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

        if len(f_adj[0])>3:
            fadj3 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj3')            

        e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_EDGES,4], name='e_map_')
        ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_POINTS,MAX_EDGES], name='ve_map_')
        keep_prob = tf.placeholder(tf.float32)
        
        if len(f_adj[0])>3:
            fadjs = [fadj0,fadj1,fadj2, fadj3]
        elif len(f_adj[0])==1:
            fadjs = [fadj0]
        else:
            fadjs = [fadj0,fadj1,fadj2]
        
        with tf.variable_scope("model"):
            n_conv = get_model_reg_multi_scale(fn_, fadjs, keep_prob, multiScale=False)

        n_conv = normalizeTensor(n_conv)

        squeezed_n_conv = tf.squeeze(n_conv)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("ERROR! Neural network not found! Aborting mission.")
            return

        # points shape should now be [NUM_POINTS, 3]
        
        print("patch number = %i"%patchNumber)
        if patchNumber>1:

            # Get faces number
            num_faces = 0
            for patchInd in range(patchNumber):
                num_faces = max(num_faces,np.max(patch_indices[patchInd])+1)
            predicted_normals = np.zeros([num_faces,3])
            
        for i in range(patchNumber):
            print("Patch "+str(i+1)+" / "+str(len(f_normals)))
            random_R = rand_rotation_matrix()

            num_p = f_normals[i].shape[1]
            tens_random_R = np.reshape(random_R,(1,1,3,3))
            tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

            my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0]}

            if len(f_adj[0])>1:
                my_feed_dict[fadj1]=f_adj[i][1]
                my_feed_dict[fadj2]=f_adj[i][2]
            if len(f_adj[0])>3:
                my_feed_dict[fadj3]=f_adj[i][3]
            outN = sess.run(squeezed_n_conv,feed_dict=my_feed_dict)

            if len(f_adj[0])>1:

                temp_perm = old_to_new_permutations[i]
                outN = outN[temp_perm]
                outN = outN[0:num_wofake_nodes[i]]

            if patchNumber==1:
                predicted_normals = outN
            else:
                predicted_normals[patch_indices[i]] = predicted_normals[patch_indices[i]] + outN
                
        #Update vertices position
        new_normals = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='fn_')
        refined_x = update_position2(xp_, new_normals, e_map_, ve_map_, iter_num=60, max_edges = MAX_EDGES)
        # refined_x, x_update = update_position_with_depth(xp_, new_normals, e_map_, ve_map_, depth_dir, iter_num=200)
        points = tf.squeeze(refined_x)
        points_update = points


        predicted_normals = normalize(predicted_normals)

        update_feed_dict = {xp_:in_points, new_normals: [predicted_normals], e_map_: edge_map, ve_map_: v_e_map}
        outPoints, x_disp = sess.run([points, points_update],feed_dict=update_feed_dict)
        sess.close()

        x_disp = normalize(x_disp)

        return outPoints, predicted_normals



def inferNet(inputMesh):
    in_points = inputMesh.v_list
    faces = inputMesh.faces_list
    f_normals = inputMesh.in_list
    f_adj = inputMesh.adj_list
    v_faces = inputMesh.v_faces_list
    new_to_old_v_list = inputMesh.vOldInd_list
    new_to_old_f_list = inputMesh.fOldInd_list
    num_points = inputMesh.vNum
    num_faces = inputMesh.fNum
    adjPerm_list = inputMesh.permutations
    real_nodes_num_list = inputMesh.num_faces
    with tf.Graph().as_default():
        random_seed = 0
        np.random.seed(random_seed)

        sess = tf.InteractiveSession()
        if(DEBUG):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)


        BATCH_SIZE=f_normals[0].shape[0]
        K_faces = f_adj[0][0].shape[2]
        K_vertices = v_faces[0].shape[2]
        NUM_IN_CHANNELS = f_normals[0].shape[2]

        xp_ = tf.placeholder('float32', shape=(BATCH_SIZE, None,3),name='xp_')

        fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')

        fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
        fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
        fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

        faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 3], name='faces_')

        v_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_vertices], name='v_faces_')
        keep_prob = tf.placeholder(tf.float32)
        
        fadjs = [fadj0,fadj1,fadj2]

        # --- Starting iterative process ---
        #rotTens = getRotationToAxis(fn_)
        with tf.variable_scope("model"):
            n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_, fadjs, keep_prob, multiScale=True)
            # n_conv1 = custom_binary_tree_pooling(n_conv0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
            # n_conv2 = custom_binary_tree_pooling(n_conv1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')

        n_conv0 = normalizeTensor(n_conv0)
        n_conv1 = normalizeTensor(n_conv1)
        n_conv2 = normalizeTensor(n_conv2)
        n_conv_list = [n_conv0, n_conv1, n_conv2]

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("ERROR! Neural network not found! Aborting mission.")
            return

        # points shape should now be [NUM_POINTS, 3]
        

        #Update vertices position
        new_normals0 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='new_normals0')
        new_normals1 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='new_normals1')
        new_normals2 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='new_normals2')

        # new_normals1 = custom_binary_tree_pooling(new_normals0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        # new_normals1 = normalizeTensor(new_normals1)
        # new_normals2 = custom_binary_tree_pooling(new_normals1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        # new_normals2 = normalizeTensor(new_normals2)
        # new_normals3 = custom_binary_tree_pooling(new_normals2, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        # new_normals3 = normalizeTensor(new_normals3)
        # new_normals4 = custom_binary_tree_pooling(new_normals3, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        # new_normals4 = normalizeTensor(new_normals4)

        upN1 = custom_upsampling(new_normals1,COARSENING_STEPS)
        upN2 = custom_upsampling(new_normals2,COARSENING_STEPS*2)
        # upN3 = custom_upsampling(new_normals3,COARSENING_STEPS*3)
        # upN4 = custom_upsampling(new_normals4,COARSENING_STEPS*4)
        new_normals = [new_normals0, new_normals1, new_normals2]
        
        normalised_disp_fine = new_normals0
        normalised_disp_mid = normalizeTensor(upN1)
        normalised_disp_coarse = normalizeTensor(upN2)
        # normalised_disp_mid = upN1
        # normalised_disp_coarse = upN2
        # normalised_disp_mid = new_normals1
        # normalised_disp_coarse = new_normals2

        # normalised_disp_coarse2 = normalizeTensor(upN3)
        # normalised_disp_coarse3 = normalizeTensor(upN4)


        pos0 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='pos0')
        pos1 = custom_binary_tree_pooling(pos0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        pos2 = custom_binary_tree_pooling(pos1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        
        refined_x, dx_list = update_position_MS(xp_, new_normals, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num_list=[80,20,20])

        refined_x = refined_x #+ dx_list[1] #+ dx_list[2]

        refined_x_mid = refined_x - dx_list[2]
        refined_x_coarse = refined_x_mid - dx_list[1]

        points = tf.reshape(refined_x,[-1,3])
        points_mid = tf.reshape(refined_x_mid,[-1,3])
        points_coarse = tf.reshape(refined_x_coarse,[-1,3])

        finalOutPoints = np.zeros((num_points,3),dtype=np.float32)
        finalOutPointsMid = np.zeros((num_points,3),dtype=np.float32)
        finalOutPointsCoarse = np.zeros((num_points,3),dtype=np.float32)
        pointsWeights = np.zeros((num_points,3),dtype=np.float32)

        finalFineNormals = np.zeros((num_faces,3),dtype=np.float32)
        finalMidNormals = np.zeros((num_faces,3),dtype=np.float32)
        finalCoarseNormals = np.zeros((num_faces,3),dtype=np.float32)
        finalCoarseNormals2 = np.zeros((num_faces,3),dtype=np.float32)
        finalCoarseNormals3 = np.zeros((num_faces,3),dtype=np.float32)

        finalFinePos = np.zeros((num_faces,3),dtype=np.float32)
        finalMidPos = np.zeros((num_faces,3),dtype=np.float32)
        finalCoarsePos = np.zeros((num_faces,3),dtype=np.float32)

        for i in range(len(f_normals)):
            print("Patch "+str(i+1)+" / "+str(len(f_normals)))
            my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2], 
                            keep_prob:1.0}
            # outN0, outN1, outN2 = sess.run([tf.squeeze(n_conv0), tf.squeeze(n_conv1), tf.squeeze(n_conv2)],feed_dict=my_feed_dict)
            print("Running normals...")
            outN0, outN1, outN2 = sess.run([n_conv0, n_conv1, n_conv2],feed_dict=my_feed_dict)
            # outN0 = sess.run(n_conv0,feed_dict=my_feed_dict)
            print("Normals: check")
            # outN = f_normals[i][0]

            fnum0 = f_adj[i][0].shape[1]
            fnum1 = f_adj[i][1].shape[1]
            fnum2 = f_adj[i][2].shape[1]

            # outN0 = f_normals[0][:,:,:3]
            outP0 = f_normals[0][:,:,3:]

            update_feed_dict = {xp_:in_points[i], new_normals0: outN0, pos0: outP0,
                                faces_: faces[i], v_faces_: v_faces[i]}

            update_feed_dict = {xp_:in_points[i], new_normals0: outN0, new_normals1: outN1, new_normals2: outN2, pos0: outP0,
                                faces_: faces[i], v_faces_: v_faces[i]}

            print("Running points...")

            outPoints, outPointsMid, outPointsCoarse, fineNormals, midNormals, coarseNormals, finePos, midPos, coarsePos = sess.run([points, points_mid, points_coarse, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse, pos0, pos1, pos2],feed_dict=update_feed_dict)

            midPos = finePos
            coarsePos = finePos

            print("Points: check")
            print("Updating mesh...")
            if len(f_normals)>1:
                finalOutPoints[new_to_old_v_list[i]] += outPoints
                finalOutPointsMid[new_to_old_v_list[i]] += outPointsMid
                finalOutPointsCoarse[new_to_old_v_list[i]] += outPointsCoarse


                pointsWeights[new_to_old_v_list[i]]+=1


                finePosP = np.squeeze(finePos)
                midPosP = np.squeeze(midPos)
                coarsePosP = np.squeeze(coarsePos)


                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
                midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
                coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
                coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]


                finalFineNormals[new_to_old_f_list[i]] = fineNormalsP
                finalMidNormals[new_to_old_f_list[i]] = midNormalsP
                finalCoarseNormals[new_to_old_f_list[i]] = coarseNormalsP


                finalFinePos = finePosP
                finalMidPos = midPosP
                finalCoarsePos = coarsePosP


            else:
                finalOutPoints = outPoints
                finalOutPointsMid = outPointsMid
                finalOutPointsCoarse = outPointsCoarse
                pointsWeights +=1


                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
                midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
                coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
                coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]

                finePosP = np.squeeze(finePos)[adjPerm_list[i]]
                finePosP = finePosP[:real_nodes_num_list[i],:]
                
                midPosP = np.squeeze(midPos)[adjPerm_list[i]]
                midPosP = midPosP[:real_nodes_num_list[i],:]
                
                coarsePosP = np.squeeze(coarsePos)[adjPerm_list[i]]
                coarsePosP = coarsePosP[:real_nodes_num_list[i],:]
                
                finalFinePos = finePosP
                finalMidPos = midPosP
                finalCoarsePos = coarsePosP

                finalFineNormals = fineNormalsP
                finalMidNormals = midNormalsP
                finalCoarseNormals = coarseNormalsP
            print("Mesh update: check")
        sess.close()

        finalOutPoints = np.true_divide(finalOutPoints,np.maximum(pointsWeights,1))
        finalOutPointsMid = np.true_divide(finalOutPointsMid,np.maximum(pointsWeights,1))
        finalOutPointsCoarse = np.true_divide(finalOutPointsCoarse,np.maximum(pointsWeights,1))

        return finalOutPoints, finalOutPointsMid, finalOutPointsCoarse, finalFineNormals, finalMidNormals, finalCoarseNormals, finalFinePos, finalMidPos, finalCoarsePos



def trainNet(trainSet, validSet):
    
    f_normals_list = trainSet.in_list
    GTfn_list = trainSet.gt_list
    f_adj_list = trainSet.adj_list
    valid_f_normals_list = validSet.in_list
    valid_GTfn_list = validSet.gt_list
    valid_f_adj_list = validSet.adj_list


    if TF_VERSION==2:
        tf.disable_eager_execution()
    # sess = tf.InteractiveSession(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=False))
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    if(DEBUG):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(NETWORK_PATH):
            os.makedirs(NETWORK_PATH)

    """
    Load dataset 
    x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
    adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                              K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
    """
    BATCH_SIZE=f_normals_list[0].shape[0]
    BATCH_SIZE=1
    K_faces = f_adj_list[0][0].shape[2]
    NUM_IN_CHANNELS = f_normals_list[0].shape[2]
    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')
    
    fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    costSamplesNum = 4000

    if len(f_adj_list[0])>3:
        fadj3 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj3')


    tfn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='tfn_')

    sample_ind = tf.placeholder(tf.int32, shape=[costSamplesNum], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    #rotTens = getRotationToAxis(fn_)
    print("NUM_IN_CHANNELS = "+str(NUM_IN_CHANNELS))
    bAddRot=True
    if bAddRot:

        tfn_rot = tf.reshape(tfn_,[BATCH_SIZE,-1,3,1])
        tfn_rot = tf.matmul(rot_mat,tfn_rot)
        tfn_rot = tf.reshape(tfn_rot,[BATCH_SIZE,-1,3])

        #Add random rotation
        if (NUM_IN_CHANNELS%3)==0:
            fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,int(NUM_IN_CHANNELS/3),3])    # 2 because of normal + position
            fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
            
            fn_rot = tf.matmul(rot_mat,fn_rot)

            fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
            fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,NUM_IN_CHANNELS])
            
        elif NUM_IN_CHANNELS==7:    #normals + border + pos
            fn_n = fn_[:,:,:3]
            fn_b = tf.expand_dims(fn_[:,:,3],axis=-1)
            fn_p = fn_[:,:,4:]

            fnn_rot = tf.expand_dims(fn_n,axis=-1)
            fnn_rot = tf.matmul(rot_mat,fnn_rot)
            fnn_rot = tf.reshape(fnn_rot, [BATCH_SIZE, -1, 3])
            fnp_rot = tf.expand_dims(fn_p,axis=-1)
            fnp_rot = tf.matmul(rot_mat,fnp_rot)
            fnp_rot = tf.reshape(fnp_rot, [BATCH_SIZE, -1, 3])

            fn_rot = tf.concat([fnn_rot, fn_b, fnp_rot], axis=-1)

        elif NUM_IN_CHANNELS==8:    #normals + area + border + pos
            fn_n = fn_[:,:,:3]
            fn_b = fn_[:,:,3:5]
            fn_p = fn_[:,:,5:]

            fnn_rot = tf.expand_dims(fn_n,axis=-1)
            fnn_rot = tf.matmul(rot_mat,fnn_rot)
            fnn_rot = tf.reshape(fnn_rot, [BATCH_SIZE, -1, 3])
            fnp_rot = tf.expand_dims(fn_p,axis=-1)
            fnp_rot = tf.matmul(rot_mat,fnp_rot)
            fnp_rot = tf.reshape(fnp_rot, [BATCH_SIZE, -1, 3])

            fn_rot = tf.concat([fnn_rot, fn_b, fnp_rot], axis=-1)

    else:
        fn_rot = fn_
        tfn_rot = tfn_

    if len(f_adj_list[0])>3:
        fadjs = [fadj0,fadj1, fadj2, fadj3]
    elif len(f_adj_list[0])==1:
        fadjs = [fadj0]
    else:
        fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        n_conv = get_model_reg_multi_scale(fn_rot, fadjs, keep_prob, multiScale=False)



    # n_conv = normalizeTensor(n_conv)
    # n_conv = tf.expand_dims(n_conv,axis=-1)
    # n_conv = tf.matmul(tf.transpose(rotTens,[0,1,3,2]),n_conv)
    # n_conv = tf.reshape(n_conv,[BATCH_SIZE,-1,3])
    # n_conv = tf.slice(fn_,[0,0,0],[-1,-1,3])+n_conv
    # print("WARNING!!!! Removed normalization of network output for training!!!")
    n_conv = normalizeTensor(n_conv)

    isNanNConv = tf.reduce_any(tf.is_nan(n_conv), name="isNanNConv")
    isFullNanNConv = tf.reduce_all(tf.is_nan(n_conv), name="isNanNConv")
    with tf.device(DEVICE):

        samp_n = tf.transpose(n_conv,[1,0,2])
        samp_n = tf.gather(samp_n,sample_ind)
        samp_n = tf.transpose(samp_n,[1,0,2])

        samp_gtn = tf.transpose(tfn_rot,[1,0,2])
        samp_gtn = tf.gather(samp_gtn,sample_ind)
        samp_gtn = tf.transpose(samp_gtn,[1,0,2])
        # customLoss = faceNormalsLoss(n_conv, tfn_rot)
        customLoss = faceNormalsLoss(samp_n, samp_gtn)
        # print("WARNING!! Charbonnier loss!")
        # customLoss = charbonnierFaceNormalsLoss(samp_n, samp_gtn)
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
    if ckpt and ckpt.model_checkpoint_path:
        splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
        if splitCkpt[0] == NET_NAME:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #Extract from checkpoint filename
            globalStep = int(splitCkpt[1])
    

    # Training

    train_loss=0
    train_samp=0

    hasNan = False

    evalStepNum=50

    with tf.device(DEVICE):
        lossArray = np.zeros([int(NUM_ITERATIONS/evalStepNum),2])
        last_loss = 0
        for iter in range(NUM_ITERATIONS):

            if (iter%SAVEITER == 0) and (iter>0):
                saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+iter)
                print("Ongoing training, net path = "+NETWORK_PATH)
                if sess.run(isFullNanNConv, feed_dict=train_fd):
                    break

            # Get random sample from training dictionary
            batch_num = random.randint(0,len(f_normals_list)-1)

            num_p = f_normals_list[batch_num].shape[1]
            random_ind = np.random.randint(num_p,size=costSamplesNum)

            random_R = rand_rotation_matrix()
            tens_random_R = np.reshape(random_R,(1,1,3,3))
            tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))


            train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], tfn_: GTfn_list[batch_num], rot_mat:tens_random_R2,
                            sample_ind: random_ind, keep_prob:1}

            if len(f_adj_list[0])>1:
                train_fd[fadj1]=f_adj_list[batch_num][1]
                train_fd[fadj2]=f_adj_list[batch_num][2]
            if len(f_adj_list[0])>3:
                train_fd[fadj3]=f_adj_list[batch_num][3]
            
            train_loss += sess.run(customLoss,feed_dict=train_fd)
            train_samp+=1
            # Show smoothed training loss
            if (iter%evalStepNum == 0):
                train_loss = train_loss/train_samp
                print("Iteration %d, training loss %g"%(iter, train_loss))

                lossArray[int(iter/evalStepNum),0]=train_loss
                train_loss=0
                train_samp=0

            # Compute validation loss
            if (iter%(evalStepNum*2) ==0):
                valid_loss = 0
                valid_samp = len(valid_f_normals_list)
                valid_random_ind = np.arange(costSamplesNum)
                for vbm in range(valid_samp):
                    
                    num_p = valid_f_normals_list[vbm].shape[1]
                    
                    tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

                    valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], tfn_: valid_GTfn_list[vbm], rot_mat:tens_random_R2,
                            sample_ind: valid_random_ind, keep_prob:1}

                    if len(f_adj_list[0])>1:
                        valid_fd[fadj1]=valid_f_adj_list[vbm][1]
                        valid_fd[fadj2]=valid_f_adj_list[vbm][2]

                    if len(f_adj_list[0])>3:
                        valid_fd[fadj3]=valid_f_adj_list[vbm][3]

                    valid_loss += sess.run(customLoss,feed_dict=valid_fd)
                valid_loss/=valid_samp
                print("Iteration %d, validation loss %g"%(iter, valid_loss))
                lossArray[int(iter/evalStepNum),1]=valid_loss
                if iter>0:
                    lossArray[int(iter/evalStepNum)-1,1] = (valid_loss+last_loss)/2
                    last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            if sess.run(isNanNConv,feed_dict=train_fd):
                hasNan = True
                print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(f_normals_list)))
                print("patch size: "+str(f_normals_list[batch_num].shape))
            
    
    saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = NETWORK_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()


# def trainAccuracyNet(in_points_list, GT_points_list, faces_list, f_normals_list, f_adj_list, v_faces_list, valid_in_points_list, valid_GT_points_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list):
def trainAccuracyNet(trainSet, validSet):
     
    in_points_list = trainSet.v_list
    GT_points_list = trainSet.gtv_list
    faces_list = trainSet.faces_list
    f_normals_list = trainSet.in_list
    # GTf_normals_list = trainSet.gt_list
    f_adj_list = trainSet.adj_list
    v_faces_list = trainSet.v_faces_list
    valid_in_points_list = validSet.v_list
    valid_GT_points_list = validSet.gtv_list
    valid_faces_list = validSet.faces_list
    valid_f_normals_list = validSet.in_list
    # valid_GTf_normals_list = validSet.gt_list
    valid_f_adj_list = validSet.adj_list
    valid_v_faces_list = validSet.v_faces_list

    SAMP_NUM = 500
    # keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(DEBUG):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(NETWORK_PATH):
            os.makedirs(NETWORK_PATH)


    """
    Load dataset 
    x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
    adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                              K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
    """
    
    dropout_prob = 0.8
    BATCH_SIZE=f_normals_list[0].shape[0]
    BATCH_SIZE=1
    K_faces = f_adj_list[0][0].shape[2]
    K_vertices = v_faces_list[0].shape[2]
    NUM_IN_CHANNELS = f_normals_list[0].shape[2]
    NUM_POINTS=in_points_list[0].shape[1]
    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')
    
    vp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='vp_')
    gtvp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='gtvp_')

    fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    # Needed for vertices update
    # e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,4], name='e_map_')
    # ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,MAX_EDGES], name='ve_map_')
    faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 3], name='faces_')

    v_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_vertices], name='v_faces_')

    sample_ind0 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind0')
    sample_ind1 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind1')


    keep_prob = tf.placeholder(tf.float32)
    
    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    rot_mat_vert = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of vertices rather than number of faces.
    rot_mat_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of GT vertices rather than number of faces.
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    bAddRot=True
    if bAddRot:

        vp_rot = tf.reshape(vp_,[BATCH_SIZE,-1,3,1])
        gtvp_rot = tf.reshape(gtvp_,[BATCH_SIZE,-1,3,1])
        vp_rot = tf.matmul(rot_mat_vert,vp_rot)
        gtvp_rot = tf.matmul(rot_mat_gt,gtvp_rot)
        vp_rot = tf.reshape(vp_rot,[BATCH_SIZE,-1,3])
        gtvp_rot = tf.reshape(gtvp_rot,[BATCH_SIZE,-1,3])
        #Add random rotation
        if (NUM_IN_CHANNELS%3)==0:
            fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,int(NUM_IN_CHANNELS/3),3])    # 2 because of normal + position
            fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
            
            fn_rot = tf.matmul(rot_mat,fn_rot)

            fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
            fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,NUM_IN_CHANNELS])
            
        elif NUM_IN_CHANNELS==7:    #normals + border + pos
            fn_n = fn_[:,:,:3]
            fn_b = tf.expand_dims(fn_[:,:,3],axis=-1)
            fn_p = fn_[:,:,4:]

            fnn_rot = tf.expand_dims(fn_n,axis=-1)
            fnn_rot = tf.matmul(rot_mat,fnn_rot)
            fnn_rot = tf.reshape(fnn_rot, [BATCH_SIZE, -1, 3])
            fnp_rot = tf.expand_dims(fn_p,axis=-1)
            fnp_rot = tf.matmul(rot_mat,fnp_rot)
            fnp_rot = tf.reshape(fnp_rot, [BATCH_SIZE, -1, 3])

            fn_rot = tf.concat([fnn_rot, fn_b, fnp_rot], axis=-1)

        elif NUM_IN_CHANNELS==8:    #normals + area + border + pos
            fn_n = fn_[:,:,:3]
            fn_b = fn_[:,:,3:5]
            fn_p = fn_[:,:,5:]

            fnn_rot = tf.expand_dims(fn_n,axis=-1)
            fnn_rot = tf.matmul(rot_mat,fnn_rot)
            fnn_rot = tf.reshape(fnn_rot, [BATCH_SIZE, -1, 3])
            fnp_rot = tf.expand_dims(fn_p,axis=-1)
            fnp_rot = tf.matmul(rot_mat,fnp_rot)
            fnp_rot = tf.reshape(fnp_rot, [BATCH_SIZE, -1, 3])

            fn_rot = tf.concat([fnn_rot, fn_b, fnp_rot], axis=-1)


    else:
        fn_rot = fn_
        vp_rot = vp_
        gtvp_rot = gtvp_





    fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_rot, fadjs, keep_prob, multiScale=True)

    n_conv0 = normalizeTensor(n_conv0)
    # n_conv1 = normalizeTensor(n_conv1)
    # n_conv2 = normalizeTensor(n_conv2)

    n_conv_list = [n_conv0, n_conv1, n_conv2]
    
    refined_x, _ = update_position_MS(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num_list=[80,20,20])

    
    samp_x = refined_x
    
    with tf.device(DEVICE):
        # customLoss = accuracyLoss(refined_x, gtvp_rot, sample_ind0)
        # customLoss = fullLoss(refined_x, gtvp_rot, sample_ind0, sample_ind1) + connectivityRegularizer(refined_x,faces_,v_faces_, sample_ind0)
        customLoss = fullLoss(refined_x, gtvp_rot, sample_ind0, sample_ind1)
        # customLoss = sampledAccuracyLoss(samp_x, gtvp_rot)
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
    if ckpt and ckpt.model_checkpoint_path:
        splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
        if splitCkpt[0] == NET_NAME:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #Extract from checkpoint filename
            globalStep = int(splitCkpt[1])
    

    # Training

    train_loss=0
    train_samp=0

    hasNan = False

    # with tf.device(DEVICE):
    lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
    last_loss = 0
    lossArrayIter = 0
    for iter in range(NUM_ITERATIONS):


        # Get random sample from training dictionary
        batch_num = random.randint(0,len(f_normals_list)-1)
        num_vgt = GT_points_list[batch_num].shape[1]
        num_vnoisy = in_points_list[batch_num].shape[1]

        while (num_vgt==0):
            print("WOLOLOLOLOLO " +str(batch_num))
            batch_num = random.randint(0,len(f_normals_list)-1)
            num_vgt = GT_points_list[batch_num].shape[1]


        num_p = f_normals_list[batch_num].shape[1]
        num_v = in_points_list[batch_num].shape[1]
        num_vgt = GT_points_list[batch_num].shape[1]
        random_ind0 = np.random.randint(num_v,size=SAMP_NUM)
        random_ind1 = np.random.randint(num_vgt,size=SAMP_NUM)


        random_R = rand_rotation_matrix()
        tens_random_R = np.reshape(random_R,(1,1,3,3))
        tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
        tens_random_Rv = np.tile(tens_random_R,(BATCH_SIZE,num_v,1,1))
        tens_random_Rgt = np.tile(tens_random_R,(BATCH_SIZE,num_vgt,1,1))

        train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
                        fadj2: f_adj_list[batch_num][2], vp_: in_points_list[batch_num], gtvp_: GT_points_list[batch_num],
                        faces_: faces_list[batch_num], v_faces_: v_faces_list[batch_num], 
                        rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
                        sample_ind0: random_ind0, sample_ind1: random_ind1, keep_prob:dropout_prob}

        train_loss_cur = customLoss.eval(feed_dict=train_fd)

        train_loss += train_loss_cur
        train_samp+=1
        # Show smoothed training loss
        if (iter%10 == 0)and(iter>-1):
            train_loss = train_loss/train_samp


            print("Iteration %d, training loss %g"%(iter, train_loss))

            lossArray[int(lossArrayIter/10)-1,0]=train_loss
            train_loss=0
            train_samp=0

        # Compute validation loss
        if ((iter%20 ==0)and(iter>0)):
            valid_loss = 0
            valid_samp = len(valid_f_normals_list)
            for vbm in range(valid_samp):
                num_p = valid_f_normals_list[vbm].shape[1]
                num_v = valid_in_points_list[vbm].shape[1]
                num_vgt = valid_GT_points_list[vbm].shape[1]
                valid_random_ind0 = np.random.randint(num_v,size=SAMP_NUM)
                valid_random_ind1 = np.random.randint(num_vgt,size=SAMP_NUM)
                tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
                tens_random_Rv = np.tile(tens_random_R,(BATCH_SIZE,num_v,1,1))
                tens_random_Rgt = np.tile(tens_random_R,(BATCH_SIZE,num_vgt,1,1))



                valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                        fadj2: valid_f_adj_list[vbm][2], vp_: valid_in_points_list[vbm], gtvp_: valid_GT_points_list[vbm],
                        faces_:valid_faces_list[vbm], v_faces_:valid_v_faces_list[vbm],
                        rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
                        sample_ind0: valid_random_ind0, sample_ind1: valid_random_ind1, keep_prob:1.0}

                valid_loss_cur = customLoss.eval(feed_dict=valid_fd)
                valid_loss += valid_loss_cur
            valid_loss/=valid_samp
            print("Iteration %d, validation loss %g"%(iter, valid_loss))
            lossArray[int(lossArrayIter/10)-1,1]=valid_loss
            if iter>0:
                lossArray[int(lossArrayIter/10)-2,1] = (valid_loss+last_loss)/2
                last_loss=valid_loss

        sess.run(train_step,feed_dict=train_fd)
        # if sess.run(isNanNConv,feed_dict=train_fd):
        #     hasNan = True
        #     print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(f_normals_list)))
        #     print("patch size: "+str(f_normals_list[batch_num].shape))
        if ((iter%500 == 0)and(iter>0)):
            saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+iter)
            # if sess.run(isFullNanNConv, feed_dict=train_fd):
            #     break
            csv_filename = NETWORK_PATH+NET_NAME+".csv"
            f = open(csv_filename,'ab')
            np.savetxt(f,lossArray, delimiter=",")
            f.close()
            lossArray = np.zeros([int(50),2]) 
            lossArrayIter=0

        lossArrayIter+=1
    
    saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = NETWORK_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()




def trainDoubleLossNet(trainSet, validSet):
    in_points_list = trainSet.v_list
    GT_points_list = trainSet.gtv_list
    faces_list = trainSet.faces_list
    f_normals_list = trainSet.in_list
    GTf_normals_list = trainSet.gt_list
    f_adj_list = trainSet.adj_list
    v_faces_list = trainSet.v_faces_list
    valid_in_points_list = validSet.v_list
    valid_GT_points_list = validSet.gtv_list
    valid_faces_list = validSet.faces_list
    valid_f_normals_list = validSet.in_list
    valid_GTf_normals_list = validSet.gt_list
    valid_f_adj_list = validSet.adj_list
    valid_v_faces_list = validSet.v_faces_list

    SAMP_NUM = 500
    # keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(DEBUG):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(NETWORK_PATH):
            os.makedirs(NETWORK_PATH)


    """
    Load dataset 
    x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
    adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                              K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
    """
    
    dropout_prob = 0.8
    BATCH_SIZE=f_normals_list[0].shape[0]
    BATCH_SIZE=1
    K_faces = f_adj_list[0][0].shape[2]
    K_vertices = v_faces_list[0].shape[2]
    NUM_IN_CHANNELS = f_normals_list[0].shape[2]
    NUM_POINTS=in_points_list[0].shape[1]
    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')
    gtfn_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='gtfn_')
    
    vp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='vp_')
    gtvp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='gtvp_')


    fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    # Needed for vertices update
    # e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,4], name='e_map_')
    # ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,MAX_EDGES], name='ve_map_')
    faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 3], name='faces_')

    v_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_vertices], name='v_faces_')

    sample_ind0 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind0')
    sample_ind1 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind1')


    keep_prob = tf.placeholder(tf.float32)
    
    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    rot_mat_vert = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of vertices rather than number of faces.
    rot_mat_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of GT vertices rather than number of faces.
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    # #Add random rotation
    # fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,2,3])    # 2 because of normal + position
    # fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
    
    # vp_rot = tf.reshape(vp_,[BATCH_SIZE,-1,3,1])
    # gtvp_rot = tf.reshape(gtvp_,[BATCH_SIZE,-1,3,1])
    
    # if keep_rot_inv:
    #     fn_rot = tf.matmul(rot_mat,fn_rot)
    #     vp_rot = tf.matmul(rot_mat_vert,vp_rot)
    #     gtvp_rot = tf.matmul(rot_mat_gt,gtvp_rot)
    # else:
    #     print("WARNING: hard-coded rot inv removal")

    # fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
    # fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,6])

    # vp_rot = tf.reshape(vp_rot,[BATCH_SIZE,-1,3])
    # gtvp_rot = tf.reshape(gtvp_rot,[BATCH_SIZE,-1,3])


    bAddRot=True
    if bAddRot:

        vp_rot = tf.reshape(vp_,[BATCH_SIZE,-1,3,1])
        gtvp_rot = tf.reshape(gtvp_,[BATCH_SIZE,-1,3,1])
        vp_rot = tf.matmul(rot_mat_vert,vp_rot)
        gtvp_rot = tf.matmul(rot_mat_gt,gtvp_rot)
        vp_rot = tf.reshape(vp_rot,[BATCH_SIZE,-1,3])
        gtvp_rot = tf.reshape(gtvp_rot,[BATCH_SIZE,-1,3])

        gtfn_rot = tf.reshape(gtfn_, [BATCH_SIZE,-1,3,1])
        gtfn_rot = tf.matmul(rot_mat,gtfn_rot)
        gtfn_rot = tf.reshape(gtfn_rot, [BATCH_SIZE,-1,3])

        #Add random rotation
        if (NUM_IN_CHANNELS%3)==0:
            fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,int(NUM_IN_CHANNELS/3),3])    # 2 because of normal + position
            fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
            
            fn_rot = tf.matmul(rot_mat,fn_rot)

            fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
            fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,NUM_IN_CHANNELS])
            
        elif NUM_IN_CHANNELS==7:    #normals + border + pos
            fn_n = fn_[:,:,:3]
            fn_b = tf.expand_dims(fn_[:,:,3],axis=-1)
            fn_p = fn_[:,:,4:]

            fnn_rot = tf.expand_dims(fn_n,axis=-1)
            fnn_rot = tf.matmul(rot_mat,fnn_rot)
            fnn_rot = tf.reshape(fnn_rot, [BATCH_SIZE, -1, 3])
            fnp_rot = tf.expand_dims(fn_p,axis=-1)
            fnp_rot = tf.matmul(rot_mat,fnp_rot)
            fnp_rot = tf.reshape(fnp_rot, [BATCH_SIZE, -1, 3])

            fn_rot = tf.concat([fnn_rot, fn_b, fnp_rot], axis=-1)

        elif NUM_IN_CHANNELS==8:    #normals + area + border + pos
            fn_n = fn_[:,:,:3]
            fn_b = fn_[:,:,3:5]
            fn_p = fn_[:,:,5:]

            fnn_rot = tf.expand_dims(fn_n,axis=-1)
            fnn_rot = tf.matmul(rot_mat,fnn_rot)
            fnn_rot = tf.reshape(fnn_rot, [BATCH_SIZE, -1, 3])
            fnp_rot = tf.expand_dims(fn_p,axis=-1)
            fnp_rot = tf.matmul(rot_mat,fnp_rot)
            fnp_rot = tf.reshape(fnp_rot, [BATCH_SIZE, -1, 3])

            fn_rot = tf.concat([fnn_rot, fn_b, fnp_rot], axis=-1)


    else:
        fn_rot = fn_
        vp_rot = vp_
        gtvp_rot = gtvp_
        gtfn_rot = gtfn_




    fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_rot, fadjs, keep_prob, multiScale=True)
        # n_conv0 = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)


    n_conv0 = normalizeTensor(n_conv0)
    n_conv1 = normalizeTensor(n_conv1)
    n_conv2 = normalizeTensor(n_conv2)

    n_conv_list = [n_conv0, n_conv1, n_conv2]
    # n_conv_list = [n_conv0]
    # isNanNConv = tf.reduce_any(tf.is_nan(n_conv), name="isNanNConv")
    # isFullNanNConv = tf.reduce_all(tf.is_nan(n_conv), name="isNanNConv")

    # refined_x, _ = update_position_MS(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num_list=[80])
    refined_x, _ = update_position_MS(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num_list=[80,20,20])

    
    # samp_x = tf.transpose(refined_x,[1,0,2])
    # samp_x = tf.gather(samp_x,sample_ind0)
    # samp_x = tf.transpose(samp_x,[1,0,2])
    samp_x = refined_x
    
    with tf.device(DEVICE):
        # customLoss = accuracyLoss(refined_x, gtvp_rot, sample_ind0)
        # customLoss = fullLoss(refined_x, gtvp_rot, sample_ind0, sample_ind1) + connectivityRegularizer(refined_x,faces_,v_faces_, sample_ind0)
        pointsLoss = fullLoss(refined_x, gtvp_rot, sample_ind0, sample_ind1)
        normalsLoss = faceNormalsLoss(n_conv0, gtfn_rot)
        customLoss = pointsLoss + normalsLoss
        # customLoss = normalsLoss
        # customLoss = sampledAccuracyLoss(samp_x, gtvp_rot)
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
    if ckpt and ckpt.model_checkpoint_path:
        splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
        if splitCkpt[0] == NET_NAME:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #Extract from checkpoint filename
            globalStep = int(splitCkpt[1])
    

    # Training

    train_loss=0
    train_samp=0

    hasNan = False

    # with tf.device(DEVICE):
    # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
    lossArray = np.zeros([int(50),2])                  # 200 = 2000/10: save csv file every 2000 iter
    last_loss = 0
    lossArrayIter = 0
    for iter in range(NUM_ITERATIONS):

        if (iter%SAVEITER == 0) and (iter>0):
            saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+iter)
            print("Ongoing training, net path = "+NETWORK_PATH)

            csv_filename = NETWORK_PATH+NET_NAME+".csv"
            f = open(csv_filename,'ab')
            np.savetxt(f,lossArray, delimiter=",")
            f.close()
            lossArray = np.zeros([int(50),2]) 
            lossArrayIter=0

        # Get random sample from training dictionary
        batch_num = random.randint(0,len(f_normals_list)-1)
        # print("Selecting patch "+str(batch_num)+" on "+str(len(f_normals_list)))
        num_vgt = GT_points_list[batch_num].shape[1]
        num_vnoisy = in_points_list[batch_num].shape[1]

        while (num_vgt==0):
            print("WOLOLOLOLOLO " +str(batch_num))
            batch_num = random.randint(0,len(f_normals_list)-1)
            num_vgt = GT_points_list[batch_num].shape[1]

        num_p = f_normals_list[batch_num].shape[1]
        num_v = in_points_list[batch_num].shape[1]
        num_vgt = GT_points_list[batch_num].shape[1]
        random_ind0 = np.random.randint(num_v,size=SAMP_NUM)
        random_ind1 = np.random.randint(num_vgt,size=SAMP_NUM)


        random_R = rand_rotation_matrix()
        tens_random_R = np.reshape(random_R,(1,1,3,3))
        tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
        tens_random_Rv = np.tile(tens_random_R,(BATCH_SIZE,num_v,1,1))
        tens_random_Rgt = np.tile(tens_random_R,(BATCH_SIZE,num_vgt,1,1))

        # batch_in_points = np.transpose(in_points_list[batch_num],[1,0,2])
        # batch_in_points = batch_in_points[random_ind]
        # batch_in_points = np.transpose(batch_in_points,[1,0,2])

        train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
                        fadj2: f_adj_list[batch_num][2], vp_: in_points_list[batch_num], gtvp_: GT_points_list[batch_num],
                        faces_: faces_list[batch_num], v_faces_: v_faces_list[batch_num], 
                        rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
                        gtfn_: GTf_normals_list[batch_num],
                        sample_ind0: random_ind0, sample_ind1: random_ind1, keep_prob:dropout_prob}
        # train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
        #                 fadj2: f_adj_list[batch_num][2], vp_: batch_in_points, gtvp_: GT_points_list[batch_num],
        #                 faces_: faces_list[batch_num], v_faces_: v_faces_list[batch_num], 
        #                 rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
        #                 sample_ind: random_ind, keep_prob:dropout_prob}


        train_loss_cur = customLoss.eval(feed_dict=train_fd)

        train_loss += train_loss_cur
        train_samp+=1
        # Show smoothed training loss
        if (iter%10 == 0)and(iter>-1):
            train_loss = train_loss/train_samp


            print("Iteration %d, training loss %g"%(iter, train_loss))

            lossArray[int(lossArrayIter/10)-1,0]=train_loss
            train_loss=0
            train_samp=0

        # Compute validation loss
        if ((iter%20 ==0)and(iter>0)):
            valid_loss = 0
            valid_normalsLoss = 0
            valid_pointsLoss = 0
            valid_samp = len(valid_f_normals_list)
            for vbm in range(valid_samp):
                num_p = valid_f_normals_list[vbm].shape[1]
                num_v = valid_in_points_list[vbm].shape[1]
                num_vgt = valid_GT_points_list[vbm].shape[1]
                valid_random_ind0 = np.random.randint(num_v,size=SAMP_NUM)
                valid_random_ind1 = np.random.randint(num_vgt,size=SAMP_NUM)
                tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
                tens_random_Rv = np.tile(tens_random_R,(BATCH_SIZE,num_v,1,1))
                tens_random_Rgt = np.tile(tens_random_R,(BATCH_SIZE,num_vgt,1,1))



                valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                        fadj2: valid_f_adj_list[vbm][2], vp_: valid_in_points_list[vbm], gtvp_: valid_GT_points_list[vbm],
                        faces_:valid_faces_list[vbm], v_faces_:valid_v_faces_list[vbm],
                        rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
                        gtfn_: valid_GTf_normals_list[vbm],
                        sample_ind0: valid_random_ind0, sample_ind1: valid_random_ind1, keep_prob:1.0}

                # valid_loss_cur = customLoss.eval(feed_dict=valid_fd)
                valid_loss_cur, valid_normalsLoss_cur, valid_poitnsLoss_cur = sess.run([customLoss, normalsLoss, pointsLoss], feed_dict=valid_fd)
                # print("valid sample "+str(vbm)+": loss = "+str(valid_loss_cur))
                valid_loss += valid_loss_cur
                valid_normalsLoss += valid_normalsLoss_cur
                valid_pointsLoss += valid_poitnsLoss_cur
            valid_loss/=valid_samp
            valid_normalsLoss/=valid_samp
            valid_pointsLoss/=valid_samp
            print("Iteration %d, validation loss = %g (points %g, normals %g)"%(iter, valid_loss, valid_pointsLoss, valid_normalsLoss))
            lossArray[int(lossArrayIter/10)-1,1]=valid_loss
            if iter>0:
                lossArray[int(lossArrayIter/10)-2,1] = (valid_loss+last_loss)/2
                last_loss=valid_loss

        sess.run(train_step,feed_dict=train_fd)
        # if sess.run(isNanNConv,feed_dict=train_fd):
        #     hasNan = True
        #     print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(f_normals_list)))
        #     print("patch size: "+str(f_normals_list[batch_num].shape))
        if ((iter%500 == 0)and(iter>0)):
            saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+iter)
            # if sess.run(isFullNanNConv, feed_dict=train_fd):
            #     break
            csv_filename = NETWORK_PATH+NET_NAME+".csv"
            f = open(csv_filename,'ab')
            np.savetxt(f,lossArray, delimiter=",")
            f.close()
            lossArray = np.zeros([int(50),2]) 
            lossArrayIter=0

        lossArrayIter+=1
    
    saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = NETWORK_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()




def faceNormalsLoss(fn,gt_fn):

    #version 1
    n_dt = tensorDotProduct(fn,gt_fn)
    # [1, fnum]
    #loss = tf.acos(n_dt-1e-5)    # So that it stays differentiable close to 1
    close_to_one = 0.9999999
    loss = tf.acos(tf.minimum(tf.maximum(n_dt,-close_to_one),close_to_one))    # So that it stays differentiable close to 1 and -1
    gtfn_abs_sum = tf.reduce_sum(tf.abs(gt_fn),axis=2)
    fakenodes = tf.less_equal(gtfn_abs_sum,10e-4)
    #fakenodes = tf.reduce_all(fakenodes,axis=-1)

    zeroVec = tf.zeros_like(loss)
    oneVec = tf.ones_like(loss)
    realnodes = tf.where(fakenodes,zeroVec,oneVec)
    loss = 180*loss/math.pi
    # loss = 1 - n_d

    #Set loss to zero for fake nodes
    loss = tf.where(fakenodes,zeroVec,loss)
    loss = tf.reduce_sum(loss)/tf.reduce_sum(realnodes)
    #loss = tf.reduce_mean(loss)
    return loss


def charbonnierFaceNormalsLoss(fn, gt_fn):

    # Squared Error part
    n_dt = tensorDotProduct(fn,gt_fn)
    # [1, fnum]
    #loss = tf.acos(n_dt-1e-5)    # So that it stays differentiable close to 1
    close_to_one = 0.999999999
    loss = tf.acos(tf.minimum(tf.maximum(n_dt,-close_to_one),close_to_one))    # So that it stays differentiable close to 1 and -1
    gtfn_abs_sum = tf.reduce_sum(tf.abs(gt_fn),axis=2)
    fakenodes = tf.less_equal(gtfn_abs_sum,10e-4)
    #fakenodes = tf.reduce_all(fakenodes,axis=-1)

    zeroVec = tf.zeros_like(loss)
    oneVec = tf.ones_like(loss)
    realnodes = tf.where(fakenodes,zeroVec,oneVec)
    squaredDiff = tf.square(loss)

    #Set loss to zero for fake nodes
    squaredDiff = tf.where(fakenodes,zeroVec,squaredDiff)

    # Charbonnier part
    epsilon = 10e-4
    summedSquaredDiff = tf.reduce_sum(squaredDiff,axis=-1)
    squaredEpsilonTensor = tf.constant(epsilon*epsilon,shape=summedSquaredDiff.shape)
    withEps = summedSquaredDiff + squaredEpsilonTensor
    loss = tf.sqrt(withEps)
    # loss = tf.reduce_mean(loss)
    loss = tf.reduce_sum(loss)/tf.reduce_sum(realnodes)
    return loss





# Loss defined as the average distance from points of P0 to point set P1
def accuracyLoss(P0, P1, sample_ind):

    accuracyThreshold = 5      # Completely empirical
    with tf.variable_scope('accuracyLoss'):
    # P0 shape: [batch, numP0, 3]
    # P1 shape: [batch, numP1, 3]

        # Take only a few selected points
        P0 = tf.transpose(P0,[1,0,2])
        P0 = tf.gather(P0,sample_ind)
        P0 = tf.transpose(P0,[1,0,2])

        numP0 = P0.shape[1]
        numP1 = P1.shape[1]

        eP0 = tf.expand_dims(P0,axis=2)
        eP1 = tf.expand_dims(P1,axis=1)

        diff = eP0 - eP1
        # [batch, numP0, numP1, 3]

        dist = tf.norm(diff, axis=-1, name = 'norm')
        # [batch ,numP0, numP1]

        precision = tf.reduce_min(dist,axis=2, name = 'min_point_set')
        # [batch, numP0]
        completeness = tf.reduce_min(dist,axis=1, name = 'completeness')
        # [batch, numP1]

        keptPoints = tf.less_equal(precision,accuracyThreshold)

        zeroVec = tf.zeros_like(precision)

        precision = tf.where(keptPoints,precision,zeroVec)

        avg_precision = 1000*(tf.reduce_mean(precision,name='avg_precision')+tf.reduce_mean(completeness, name='avg_completeness'))

    return avg_precision


# Loss defined as the average distance from points of P0 to point set P1
def fullLoss(P0, P1, sample_ind0, sample_ind1):

    accuracyThreshold = 5000      # Completely empirical
    compThreshold = 5000
    with tf.variable_scope('accuracyLoss'):
    # P0 shape: [batch, numP0, 3]
    # P1 shape: [batch, numP1, 3]

        # Take only a few selected points
        sP0 = tf.transpose(P0,[1,0,2])
        sP0 = tf.gather(sP0,sample_ind0)
        sP0 = tf.transpose(sP0,[1,0,2])

        sP1 = tf.transpose(P1,[1,0,2])
        sP1 = tf.gather(sP1,sample_ind1)
        sP1 = tf.transpose(sP1,[1,0,2])

        numsP0 = sP0.shape[1]
        numP0 = P0.shape[1]
        numP1 = P1.shape[1]
        numsP1 = sP1.shape[1]

        eP0 = tf.expand_dims(P0,axis=2)
        eP1 = tf.expand_dims(P1,axis=1)
        esP0 = tf.expand_dims(sP0,axis=2)
        esP1 = tf.expand_dims(sP1,axis=1)

        diff0 = esP0 - eP1
        diff1 = eP0 - esP1
        # [batch, numP0, numP1, 3]

        dist0 = tf.norm(diff0, axis=-1, name = 'norm')
        # [batch ,numsP0, numP1]
        dist1 = tf.norm(diff1, axis=-1, name = 'norm')
        # [batch ,numP0, numsP1]

        precision = tf.reduce_min(dist0,axis=2, name = 'min_point_set')
        # [batch, numP0]
        completeness = tf.reduce_min(dist1,axis=1, name = 'completeness')
        # [batch, numP1]

        keptPoints = tf.less_equal(precision,accuracyThreshold)
        zeroVec = tf.zeros_like(precision)
        precision = tf.where(keptPoints,precision,zeroVec)

        keptPointsC = tf.less_equal(completeness,compThreshold)
        zeroVecC = tf.zeros_like(completeness)
        completeness = tf.where(keptPointsC,completeness,zeroVecC)

        avg_precision = 1000*(tf.reduce_mean(precision,name='avg_precision')+tf.reduce_mean(completeness, name='avg_completeness'))

    return avg_precision


# Loss defined as the average distance from points of P0 to point set P1
def sampledAccuracyLoss(P0, P1):

    accuracyThreshold = 5      # Completely empirical
    with tf.variable_scope('accuracyLoss'):
    # P0 shape: [batch, numP0, 3]
    # P1 shape: [batch, numP1, 3]

        numP0 = P0.shape[1]
        numP1 = P1.shape[1]

        # eP0 = tf.expand_dims(P0,axis=2)
        # eP1 = tf.expand_dims(P1,axis=1)

        eP0 = tf.reshape(P0,[1,-1,1,3])
        eP1 = tf.reshape(P1,[1,1,-1,3])

        diff = eP0 - eP1
        # [batch, numP0, numP1, 3]

        dist = tf.norm(diff, axis=-1, name = 'norm')
        # [batch ,numP0, numP1]

        accu = tf.reduce_min(dist,axis=2, name = 'min_point_set')
        # [batch, numP0]
        completeness = tf.reduce_min(dist,axis=1, name = 'completeness')

        keptPoints = tf.less_equal(accu,accuracyThreshold)

        zeroVec = tf.zeros_like(accu)

        accu = tf.where(keptPoints,accu,zeroVec)

        avg_precision = 1000 * (tf.reduce_mean(accu,name='avg_precision')+tf.reduce_mean(completeness, name='avg_completeness'))

    return avg_precision


# Original update algorithm from Taubin (Linear anisotropic mesh filtering)
# Copied from function above, which was my own adaptation of Taubin's algorithm with vertices normals 
def update_position2(x, face_normals, edge_map, v_edges, iter_num=20, max_edges=20):

    lmbd = 1/18

    batch_size, num_points, space_dims = x.get_shape().as_list()
    _, num_edges, _ = edge_map.get_shape().as_list()

    # edge_map is a list of edges of the form [v1, v2, f1, f2]
    # shape = (batch_size, edge_num, 4)
    # v_edges is a list of edges indices for each vertex
    # shape = (batch_size, num_points, max_edges (50))

    v_edges=v_edges+1                                                       # start indexing from 1. Transform unused slots (-1) to 0
    

    # Offset both 'faces' columns: we switch from 0-indexing to 1-indexing
    e_offset = tf.constant([[[0,0,1,1]]],dtype = tf.int32)
    edge_map = edge_map + e_offset

    # Then, add zero-line (order is important, so that faces in first line stay 0), since we offset v_edges
    pad_line = tf.zeros([batch_size,1,4],dtype=tf.int32)
    edge_map = tf.concat([pad_line,edge_map], axis=1)   # Add zero-line accordingly

    # Add zero-line to face normals as well, since we offset last columns of edge_map
    face_normals = tf.concat([tf.zeros([batch_size,1,3]),face_normals],axis=1)

    v_edges = tf.squeeze(v_edges)
    edge_map = tf.transpose(edge_map,[1,0,2])
    n_edges = tf.gather(edge_map,v_edges)
    edge_map = tf.transpose(edge_map,[1,0,2])
    n_edges = tf.transpose(n_edges,[2,0,1,3])
    # shape = (batch_size, num_points, max_edges, 4)

    n_edges = tf.squeeze(n_edges)

    fn_slice = tf.slice(n_edges, [0,0,2],[-1,-1,-1])

    face_normals = tf.transpose(face_normals,[1,0,2])
    n_f_normals = tf.gather(face_normals,fn_slice)
    face_normals = tf.transpose(face_normals,[1,0,2])
    n_f_normals = tf.transpose(n_f_normals,[3,0,1,2,4])
    # shape = (batch_size, num_points, max_edges, 2, 3)

    n_f_normals = tf.tile(n_f_normals,[1,1,1,2,1])

    for it in range(iter_num):
        
        v_slice = tf.slice(n_edges, [0,0,0],[-1,-1,2])

        x = tf.transpose(x,[1,0,2])
        n_v_pairs = tf.gather(x,v_slice)
        x = tf.transpose(x,[1,0,2])
        n_v_pairs = tf.transpose(n_v_pairs,[3,0,1,2,4])
        # shape = (batch_size, num_points, max_edges, 2, 3)

        # We need all (xj-xi) for each xi, but we do not know the order of vertices for each edge.
        # Thus, rather than subtracting both, we subtract x, and compute both (xj-xi) and (xi-xi)
        # The second case will yield 0, and the sum should be as expected
        # Expand and reshape x to the right dimension for the subtraction
        exp_x = tf.expand_dims(x,axis=2)
        exp_x = tf.expand_dims(exp_x,axis=2)
        exp_x = tf.tile(exp_x, [1,1,max_edges,2,1])

        n_edges_vec = tf.subtract(n_v_pairs,exp_x)
        # shape = (batch_size, num_points, max_edges, 2, 3)

        # Double the input (tile and reshape), to multiply with both adjacent faces
        n_edges_vec = tf.tile(n_edges_vec, [1,1,1,1,2])

        n_edges_vec = tf.reshape(n_edges_vec, [batch_size, -1, max_edges, 4, 3])


        v_dp = tensorDotProduct(n_edges_vec,n_f_normals)        # since normals should be 0 for unused edges, resulting dot product should be zero. No need to deal with it explicitly
        # shape = (batch_size, num_points, max_edges, 4)

        v_dp = tf.tile(tf.expand_dims(v_dp, axis=-1),[1,1,1,1,3])
        # shape = (batch_size, num_points, max_edges, 4, 3)

        pos_update = tf.multiply(n_f_normals,v_dp)
        # shape = (batch_size, num_points, max_edges, 4, 3)

        pos_update = tf.reduce_sum(pos_update,axis=3)
        # shape = (batch_size, num_points, max_edges, 3)
        pos_update = tf.reduce_sum(pos_update,axis=2)
        # shape = (batch_size, num_points, 3)

        x_update = lmbd * pos_update

        x = tf.add(x,x_update)
        
    return x


# Same as above, but displacement is only done along a given direction depth_dir ([1,3]) 
def update_position_with_depth(x, face_normals, edge_map, v_edges, depth_dir, iter_num=20):

    lmbd = 1/18

    batch_size, num_points, space_dims = x.get_shape().as_list()
    max_edges = 50
    _, num_edges, _ = edge_map.get_shape().as_list()

    # edge_map is a list of edges of the form [v1, v2, f1, f2]
    # shape = (batch_size, edge_num, 4)
    # v_edges is a list of edges indices for each vertex
    # shape = (batch_size, num_points, max_edges (50))

    v_edges=v_edges+1                                                       # start indexing from 1. Transform unused slots (-1) to 0
    

    # Offset both 'faces' columns: we switch from 0-indexing to 1-indexing
    e_offset = tf.constant([[[0,0,1,1]]],dtype = tf.int32)
    edge_map = edge_map + e_offset

    # Then, add zero-line (order is important, so that faces in first line stay 0), since we offset v_edges
    pad_line = tf.zeros([batch_size,1,4],dtype=tf.int32)
    edge_map = tf.concat([pad_line,edge_map], axis=1)   # Add zero-line accordingly

    # Add zero-line to face normals as well, since we offset last columns of edge_map
    face_normals = tf.concat([tf.zeros([batch_size,1,3]),face_normals],axis=1)

    v_edges = tf.squeeze(v_edges)
    edge_map = tf.transpose(edge_map,[1,0,2])
    n_edges = tf.gather(edge_map,v_edges)
    edge_map = tf.transpose(edge_map,[1,0,2])
    n_edges = tf.transpose(n_edges,[2,0,1,3])
    # shape = (batch_size, num_points, max_edges, 4)

    n_edges = tf.squeeze(n_edges)

    fn_slice = tf.slice(n_edges, [0,0,2],[-1,-1,-1])

    face_normals = tf.transpose(face_normals,[1,0,2])
    n_f_normals = tf.gather(face_normals,fn_slice)
    face_normals = tf.transpose(face_normals,[1,0,2])
    n_f_normals = tf.transpose(n_f_normals,[3,0,1,2,4])
    # shape = (batch_size, num_points, max_edges, 2, 3)

    n_f_normals = tf.tile(n_f_normals,[1,1,1,2,1])

    v_slice = tf.slice(n_edges, [0,0,0],[-1,-1,2])


    depth_dir_tiled = tf.reshape(depth_dir, [1,-1,1,1,3])

    x_init = x
    for it in range(iter_num):
        
        
        x = tf.transpose(x,[1,0,2])
        n_v_pairs = tf.gather(x,v_slice)
        x = tf.transpose(x,[1,0,2])
        n_v_pairs = tf.transpose(n_v_pairs,[3,0,1,2,4])
        # shape = (batch_size, num_points, max_edges, 2, 3)

        # We need all (xj-xi) for each xi, but we do not know the order of vertices for each edge.
        # Thus, rather than subtracting both, we subtract x, and compute both (xj-xi) and (xi-xi)
        # The second case will yield 0, and the sum should be as expected
        # Expand and reshape x to the right dimension for the subtraction
        exp_x = tf.expand_dims(x,axis=2)
        exp_x = tf.expand_dims(exp_x,axis=2)
        exp_x = tf.tile(exp_x, [1,1,max_edges,2,1])

        n_edges_vec = tf.subtract(n_v_pairs,exp_x)
        # shape = (batch_size, num_points, max_edges, 2, 3)

        # Double the input (tile and reshape), to multiply with both adjacent faces
        n_edges_vec = tf.tile(n_edges_vec, [1,1,1,1,2])

        n_edges_vec = tf.reshape(n_edges_vec, [batch_size, -1, max_edges, 4, 3])


        v_dp = tensorDotProduct(n_edges_vec,n_f_normals)        # since normals should be 0 for unused edges, resulting dot product should be zero. No need to deal with it explicitly
        # shape = (batch_size, num_points, max_edges, 4)

        v_dp = tf.tile(tf.expand_dims(v_dp, axis=-1),[1,1,1,1,3])
        # shape = (batch_size, num_points, max_edges, 4, 3)

        pos_update = tf.multiply(n_f_normals,v_dp)
        # shape = (batch_size, num_points, max_edges, 4, 3)

        pos_update_proj = tensorDotProduct(pos_update, depth_dir_tiled)

        pos_update_proj = tf.tile(tf.expand_dims(pos_update_proj, axis=-1),[1,1,1,1,3])

        pos_update_proj = tf.multiply(pos_update_proj, depth_dir_tiled)


        pos_update_proj = tf.reduce_sum(pos_update_proj,axis=3)
        # shape = (batch_size, num_points, max_edges, 3)
        pos_update_proj = tf.reduce_sum(pos_update_proj,axis=2)
        # shape = (batch_size, num_points, 3)

        x_update = lmbd * pos_update_proj

        x = tf.add(x,x_update)

        
    return x, (x-x_init)


def update_position_MS(x, face_normals_list, faces, v_faces0, coarsening_steps, iter_num_list=[80,20,20]):

    # batch_size, num_points, space_dims = x.get_shape().as_list()
    x = tf.reshape(x,[-1,3])
    _, _, K = v_faces0.get_shape().as_list()

    scale_num = len(face_normals_list)   # Always 3 so far

    minus1Tens = tf.zeros_like(v_faces0,dtype=tf.int32)
    minus1Tens = minus1Tens - 1
    real_v_faces = tf.not_equal(v_faces0,minus1Tens)
    v_numf = tf.reduce_sum(tf.where(real_v_faces, tf.ones_like(v_faces0,dtype=tf.float32),tf.zeros_like(v_faces0,dtype=tf.float32)),axis=-1)
    # v_numf = v_numf_list[cur_scale]
    lmbd = tf.reciprocal(v_numf)
    lmbd = tf.reshape(lmbd, [-1,1])
    lmbd = tf.tile(lmbd,[1,3])

    dx_list = []
    for s in range(scale_num):
    # for cur_scale in range(1):
        cur_scale = scale_num-1-s

        # # print("WARNING! Hard-coded mid-and-fine scale vertex update")
        # # if cur_scale>1:
        # #     continue
        # print("WARNING! Hard-coded fine scale vertex update")
        # if cur_scale>0:
        #     continue

        face_n = face_normals_list[cur_scale]
        
        # Get rid of batch dim
        face_n = tf.reshape(face_n,[-1,3])

        # v_faces0 gives the indices of adjacent faces at the finest level of the graph.
        # To get adjacent nodes at the next coarser level, we divide by 2^coarsening_steps
        # Note that nodes will appear several times, meaning the contribution of nodes to each vertex will be weighted depending on how 'close' they are to the vertex
        
        coarsening_tens = tf.zeros_like(v_faces0, dtype=tf.int32) + int(math.pow(math.pow(2,coarsening_steps),cur_scale))

        v_faces = tf.div(v_faces0,coarsening_tens)

        # get rid of batch dim
        v_faces = tf.squeeze(v_faces)

        
        # Add fake face with null normal and switch to 1-indexing for v_faces
        v_faces = v_faces+1
        face_n = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),face_n),axis=0)

        v_fn = tf.gather(face_n, v_faces)
        # [vnum, v_faces_num, 3]
        x_init = x
        # if cur_scale==2:
        #     iter_num=iter_num
        #     # iter_num= 40
        # else:
        #     iter_num=iter_num
        #     iter_num = 20
        iter_num = iter_num_list[s]
        for it in range(iter_num):
            # print("Scale "+str(cur_scale)+", iter "+str(it))
            
            new_fpos = updateFacesCenter(x,faces,coarsening_steps)
            face_pos = new_fpos[cur_scale]
            face_pos = tf.reshape(face_pos,[-1,3])

            face_pos = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),face_pos),axis=0)
            v_c = tf.gather(face_pos,v_faces)
            # [vnum, v_faces_num, 3]

            # xm = tf.reshape(x,[-1,1,3])
            xm = tf.expand_dims(x,axis=1)
            xm = tf.tile(xm,[1,K,1])

            e = v_c - xm

            n_w = tensorDotProduct(v_fn, e)
            # [vnum, v_faces_num]
            # Since face_n should be null for 'non-faces' in v_faces, dot product should be null, so there is no need to deal with this explicitly

            n_w = tf.tile(tf.expand_dims(n_w,axis=-1),[1,1,3])
            # [vnum, v_faces_num, 3]

            update = tf.multiply(n_w,v_fn)
            # [vnum, v_faces_num, 3]
            update = tf.reduce_sum(update, axis=1)
            # [vnum, 3]        

            x_update = tf.multiply(lmbd,update)
            # x_update = (1/18)* update

            x = tf.add(x,x_update)

        # iter_num*=2
        dx_list.append(x-x_init)
    x = tf.expand_dims(x,axis=0)
    return x, dx_list


def updateFacesCenter(vertices, faces, coarsening_steps):

    batch_size, fnum, space_dims = faces.get_shape().as_list()

    # vertices shape: [batch, vnum, 3]
    # faces shape: [batch, fnum, 3]

    # get rid of batch dimension (useless... should do this everywhere)
    vertices = tf.reshape(vertices,[-1,3])
    faces = tf.reshape(faces,[-1,3])

    # Add fake vertex for fake face nodes
    faces = faces+1     #switch to 1-indexing (we assume fake faces have vertices equal to -1)

    fake_vertex = tf.constant([[0,0,0]],dtype=tf.float32)
    vertices = tf.concat((fake_vertex,vertices),axis=0)



    fpos0 = tf.gather(vertices,faces)
    # [fnum, 3, 3]
    fpos0 = tf.reduce_mean(fpos0,axis=1)
    # [fnum, 3]

    fpos0 = tf.expand_dims(fpos0,axis=0)
    # We have position of finest graph nodes
    fpos1 = custom_binary_tree_pooling(fpos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
    fpos2 = custom_binary_tree_pooling(fpos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')


    return [fpos0, fpos1, fpos2]


# This is a piece of legacy code copied here to be stored for the time being.
# Not read/controlled/tested yet
def validateData():
    pass
    # --- Validate data ---

    # # input "raw" training data
    # examplesNum = len(f_normals_list)

    # for p in range(examplesNum):
    #     myN = gtf_normals_list[p][0]
    #     myN = normalize(myN)
    #     gtf_normals_list[p] = myN[np.newaxis,:,:]

    # for p in range(examplesNum):
    #     myN = gtf_normals_list[p][0]
    #     angColorGT = (myN+1)/2
    #     angColorNoisy = (f_normals_list[p][0,:,:3]+1)/2
    #     myV = v_pos_list[p][0]
    #     myF = faces_list[p][0]
    #     print("myV shape = ",myV.shape)
    #     print("myF shape = ",myF.shape)
    #     print("angColorGT shape = ",angColorGT.shape)
    #     print("myF sample: ",myF[:4])
    #     myF = myF.astype(np.int32)
    #     newV, newF = getColoredMesh(myV, myF, angColorGT)
    #     newVnoisy, newFnoisy = getColoredMesh(myV, myF, angColorNoisy)
    #     denoizedFile = "gtnormals_%i.obj"%p
    #     noisyFile = "noisynormals_%i.obj"%p
    #     write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
    #     write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFile)

    # Corrected GT data

    # examplesNum = len(f_normals_list)
    # for p in range(examplesNum):

    #     # p = 6
    #     samp = 1641
    #     myN = gtf_normals_list[p][0]
    #     # print("wut length = ",len(f_adj_list[p]))
    #     myAdj = f_adj_list[p][0][0]
    #     # print("myAdj shape = ",myAdj.shape)
    #     # print("myN shape = ",myN.shape)
    #     filteredN = filterFlippedFaces(myN, myAdj)

    #     print("wtf samp = ",filteredN[samp])
    #     adjN = colorFacesByAdjacency(myN, myAdj)
    #     angColorGT = (filteredN+1)/2
    #     adjColor = (adjN+1)/2

    #     print("wtf2 samp = ",angColorGT[samp])

    #     myV = v_pos_list[p][0]
    #     myF = faces_list[p][0]
    #     # print("myV shape = ",myV.shape)
    #     # print("myF shape = ",myF.shape)
    #     # print("angColorGT shape = ",angColorGT.shape)
    #     # print("myF sample: ",myF[:4])
    #     myF = myF.astype(np.int32)
    #     newV, newF = getColoredMesh(myV, myF, angColorGT)
    #     newVAdj, newFAdj = getColoredMesh(myV, myF, adjColor)
        
    #     denoizedFile = "filtered_gtnormals_%i.obj"%p
    #     adjFile = "adjnormals_%i.obj"%p
    #     write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
    #     write_mesh(newVAdj, newFAdj, RESULTS_PATH+adjFile)
        
    # Valid data
    # examplesNum = len(valid_f_normals_list)
    # for p in range(examplesNum):
    #     angColorGT = (valid_gtf_normals_list[p][0]+1)/2
    #     angColorNoisy = (valid_f_normals_list[p][0,:,:3]+1)/2
    #     myV = valid_v_pos_list[p][0]
    #     myF = valid_faces_list[p][0]
    #     print("myV shape = ",myV.shape)
    #     print("myF shape = ",myF.shape)
    #     print("angColorGT shape = ",angColorGT.shape)
    #     print("myF sample: ",myF[:4])
    #     myF = myF.astype(np.int32)
    #     newV, newF = getColoredMesh(myV, myF, angColorGT)
    #     newVnoisy, newFnoisy = getColoredMesh(myV, myF, angColorNoisy)
    #     denoizedFile = "valid_gtnormals_%i.obj"%p
    #     noisyFile = "valid_noisynormals_%i.obj"%p
    #     write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
    #     write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFile)

def train(withVerts=False):
    binDumpPath = BINARY_DUMP_PATH
    if withVerts:
        tsPickleName = 'trainingSetWithVertices.pkl'
        vsPickleName = 'validSetWithVertices.pkl'
    else:
        tsPickleName = 'trainingSet.pkl'
        vsPickleName = 'validSet.pkl'

    # Load data
    with open(binDumpPath+tsPickleName, 'rb') as fp:
        myTS = pickle.load(fp)
    with open(binDumpPath+vsPickleName, 'rb') as fp:
        myVS = pickle.load(fp)
    
    if withVerts:
        # Train network with accuracy loss on point sets (rather than normals angular loss)
        # trainDoubleLossNet(myTS, myVS)
        trainAccuracyNet(myTS, myVS)
    else:

        # myTS.correctGTFlippedFaces()
        # myVS.correctGTFlippedFaces()
        trainNet(myTS,myVS)


def mainFunction():

    

    maxSize = MAX_PATCH_SIZE
    patchSize = MAX_PATCH_SIZE

    # Coarsening parameters
    # coarseningLvlNum = 3
    coarseningStepNum = COARSENING_STEPS
    coarseningLvlNum = COARSENING_LVLS


    binDumpPath = BINARY_DUMP_PATH


    train(withVerts=INCLUDE_VERTICES)
    print("Training Complete: network saved to " + NETWORK_PATH)
   

if __name__ == "__main__":
    
    print("Tensorflow version = ", tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--num_iterations', type=int, default=20000)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--net_name', type=str, default='net')
    parser.add_argument('--running_mode', type=int, default=0)
    parser.add_argument('--coarsening_steps', type=int, default=2)

    FLAGS = parser.parse_args()

    # Override default results path if specified as command parameter
    if not FLAGS.results_path is None:
        RESULTS_PATH = FLAGS.results_path
        if not RESULTS_PATH[-1]=='/':
            RESULTS_PATH = RESULTS_PATH + "/"
    if not FLAGS.network_path is None:
        NETWORK_PATH = FLAGS.network_path
        if not NETWORK_PATH[-1]=='/':
            NETWORK_PATH = NETWORK_PATH + "/"


    NUM_ITERATIONS = FLAGS.num_iterations
    DEVICE = FLAGS.device
    NET_NAME = FLAGS.net_name
    RUNNING_MODE = FLAGS.running_mode
    DEBUG = FLAGS.debug
    
    # Experimental value on synthetic dataset:
    AVG_EDGE_LENGTH = 0.005959746586165783


    mainFunction()


