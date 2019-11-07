from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
#import h5py
import argparse
import os
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


# def inferNet(in_points, faces, f_normals, f_adj, v_faces, new_to_old_v_list, new_to_old_f_list, num_points, num_faces, adjPerm_list, real_nodes_num_list):

    #     with tf.Graph().as_default():
    #         random_seed = 0
    #         np.random.seed(random_seed)

    #         sess = tf.InteractiveSession()
    #         if(FLAGS.debug):    #launches debugger at every sess.run() call
    #             sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #             sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    #         if not os.path.exists(RESULTS_PATH):
    #                 os.makedirs(RESULTS_PATH)

    #         """
    #         Load dataset
    #         x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
    #         adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
    #                                                   K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
    #         """

    #         BATCH_SIZE=f_normals[0].shape[0]
    #         K_faces = f_adj[0][0].shape[2]
    #         K_vertices = v_faces[0].shape[2]
    #         NUM_IN_CHANNELS = f_normals[0].shape[2]

    #         xp_ = tf.placeholder('float32', shape=(BATCH_SIZE, None,3),name='xp_')

    #         fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')

    #         fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    #         fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    #         fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    #         faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 3], name='faces_')

    #         v_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_vertices], name='v_faces_')
    #         keep_prob = tf.placeholder(tf.float32)
            
    #         fadjs = [fadj0,fadj1,fadj2]

    #         # --- Starting iterative process ---
    #         #rotTens = getRotationToAxis(fn_)
    #         with tf.variable_scope("model"):
    #             n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
    #             # n_conv0 = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
    #             # n_conv1 = n_conv0
    #             # n_conv2 = n_conv0
            
    #             # n_conv1 = custom_binary_tree_pooling(n_conv0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
    #             # n_conv2 = custom_binary_tree_pooling(n_conv1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')

    #         # n_conv0 = fn_
    #         n_conv0 = normalizeTensor(n_conv0)
    #         n_conv1 = normalizeTensor(n_conv1)
    #         n_conv2 = normalizeTensor(n_conv2)
    #         n_conv_list = [n_conv0, n_conv1, n_conv2]

    #         # refined_x = update_position_MS(xp_, new_normals, faces_, v_faces_, coarsening_steps=3)


    #         saver = tf.train.Saver()

    #         sess.run(tf.global_variables_initializer())

    #         ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
    #         if ckpt and ckpt.model_checkpoint_path:
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #         else:
    #             print("ERROR! Neural network not found! Aborting mission.")
    #             return

    #         # points shape should now be [NUM_POINTS, 3]
            

    #         #Update vertices position
    #         new_normals0 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='new_normals0')
    #         new_normals1 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='new_normals1')
    #         new_normals2 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='new_normals2')

    #         # new_normals1 = custom_binary_tree_pooling(new_normals0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
    #         # new_normals1 = normalizeTensor(new_normals1)
    #         # new_normals2 = custom_binary_tree_pooling(new_normals1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
    #         # new_normals2 = normalizeTensor(new_normals2)
    #         # new_normals3 = custom_binary_tree_pooling(new_normals2, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
    #         # new_normals3 = normalizeTensor(new_normals3)
    #         # new_normals4 = custom_binary_tree_pooling(new_normals3, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
    #         # new_normals4 = normalizeTensor(new_normals4)

    #         upN1 = custom_upsampling(new_normals1,COARSENING_STEPS)
    #         upN2 = custom_upsampling(new_normals2,COARSENING_STEPS*2)
    #         # upN3 = custom_upsampling(new_normals3,COARSENING_STEPS*3)
    #         # upN4 = custom_upsampling(new_normals4,COARSENING_STEPS*4)
    #         new_normals = [new_normals0, new_normals1, new_normals2]
            
    #         normalised_disp_fine = new_normals0
    #         normalised_disp_mid = normalizeTensor(upN1)
    #         normalised_disp_coarse = normalizeTensor(upN2)
    #         # normalised_disp_mid = upN1
    #         # normalised_disp_coarse = upN2
    #         # normalised_disp_mid = new_normals1
    #         # normalised_disp_coarse = new_normals2

    #         # normalised_disp_coarse2 = normalizeTensor(upN3)
    #         # normalised_disp_coarse3 = normalizeTensor(upN4)


    #         pos0 = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='pos0')
    #         pos1 = custom_binary_tree_pooling(pos0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
    #         pos2 = custom_binary_tree_pooling(pos1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
            
    #         # refined_x, dx_list = update_position_disp(xp_, new_normals, faces_, v_faces_, coarsening_steps=COARSENING_STEPS)
    #         # refined_x, dx_list = update_position_MS_damp(xp_, new_normals, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num=600)
    #         refined_x, dx_list = update_position_MS(xp_, new_normals, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num_list=[80,20,20])

    #         refined_x = refined_x #+ dx_list[1] #+ dx_list[2]

    #         refined_x_mid = refined_x - dx_list[2]
    #         refined_x_coarse = refined_x_mid - dx_list[1]

    #         points = tf.reshape(refined_x,[-1,3])
    #         points_mid = tf.reshape(refined_x_mid,[-1,3])
    #         points_coarse = tf.reshape(refined_x_coarse,[-1,3])
    #         # points = tf.squeeze(refined_x)

    #         finalOutPoints = np.zeros((num_points,3),dtype=np.float32)
    #         finalOutPointsMid = np.zeros((num_points,3),dtype=np.float32)
    #         finalOutPointsCoarse = np.zeros((num_points,3),dtype=np.float32)
    #         pointsWeights = np.zeros((num_points,3),dtype=np.float32)

    #         finalFineNormals = np.zeros((num_faces,3),dtype=np.float32)
    #         finalMidNormals = np.zeros((num_faces,3),dtype=np.float32)
    #         finalCoarseNormals = np.zeros((num_faces,3),dtype=np.float32)
    #         finalCoarseNormals2 = np.zeros((num_faces,3),dtype=np.float32)
    #         finalCoarseNormals3 = np.zeros((num_faces,3),dtype=np.float32)

    #         finalFinePos = np.zeros((num_faces,3),dtype=np.float32)
    #         finalMidPos = np.zeros((num_faces,3),dtype=np.float32)
    #         finalCoarsePos = np.zeros((num_faces,3),dtype=np.float32)

    #         for i in range(len(f_normals)):
    #             print("Patch "+str(i+1)+" / "+str(len(f_normals)))
    #             my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2], 
    #                             keep_prob:1.0}
    #             # outN0, outN1, outN2 = sess.run([tf.squeeze(n_conv0), tf.squeeze(n_conv1), tf.squeeze(n_conv2)],feed_dict=my_feed_dict)
    #             print("Running normals...")
    #             outN0, outN1, outN2 = sess.run([n_conv0, n_conv1, n_conv2],feed_dict=my_feed_dict)
    #             # outN0 = sess.run(n_conv0,feed_dict=my_feed_dict)
    #             print("Normals: check")
    #             # outN = f_normals[i][0]

    #             fnum0 = f_adj[i][0].shape[1]
    #             fnum1 = f_adj[i][1].shape[1]
    #             fnum2 = f_adj[i][2].shape[1]

    #             # outN0 = f_normals[0][:,:,:3]
    #             outP0 = f_normals[0][:,:,3:]
    #             # outN0 = np.tile(np.array([[[0,0,1]]]),[1,f_normals[0].shape[1],1])

                
                

    #             update_feed_dict = {xp_:in_points[i], new_normals0: outN0, pos0: outP0,
    #                                 faces_: faces[i], v_faces_: v_faces[i]}
    #             # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
    #             #                     faces_: faces[i], v_faces_: v_faces[i]}
    #             # testNorm = f_normals[i][:,:,:3]/100
    #             update_feed_dict = {xp_:in_points[i], new_normals0: outN0, new_normals1: outN1, new_normals2: outN2, pos0: outP0,
    #                                 faces_: faces[i], v_faces_: v_faces[i]}
    #             # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
    #             #                     faces_: faces[i], v_faces_: v_faces[i]}

    #             print("Running points...")

                

    #             # outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, new_normals0, upN1, upN2],feed_dict=update_feed_dict)
    #             # outPoints, fineNormals, midNormals, coarseNormals, coarseNormals2, coarseNormals3 = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse, normalised_disp_coarse2, normalised_disp_coarse3],feed_dict=update_feed_dict)
    #             # outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse],feed_dict=update_feed_dict)

    #             # outPoints, fineNormals, midNormals, coarseNormals, finePos, midPos, coarsePos = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse, pos0, pos1, pos2],feed_dict=update_feed_dict)
    #             outPoints, outPointsMid, outPointsCoarse, fineNormals, midNormals, coarseNormals, finePos, midPos, coarsePos = sess.run([points, points_mid, points_coarse, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse, pos0, pos1, pos2],feed_dict=update_feed_dict)

    #             midPos = finePos
    #             coarsePos = finePos

    #             print("Points: check")
    #             print("Updating mesh...")
    #             if len(f_normals)>1:
    #                 finalOutPoints[new_to_old_v_list[i]] += outPoints
    #                 finalOutPointsMid[new_to_old_v_list[i]] += outPointsMid
    #                 finalOutPointsCoarse[new_to_old_v_list[i]] += outPointsCoarse


    #                 pointsWeights[new_to_old_v_list[i]]+=1

    #                 # fineNormalsP = np.squeeze(fineNormals)
    #                 # midNormalsP = np.squeeze(midNormals)
    #                 # coarseNormalsP = np.squeeze(coarseNormals)

    #                 finePosP = np.squeeze(finePos)
    #                 midPosP = np.squeeze(midPos)
    #                 coarsePosP = np.squeeze(coarsePos)

    #                 # finePosP = np.squeeze(finePos)[adjPerm_list[i]]
    #                 # midPosP = np.squeeze(midPos)[adjPerm_list[i]]
    #                 # coarsePosP = np.squeeze(coarsePos)[adjPerm_list[i]]

    #                 # finePosP = finePosP[:real_nodes_num_list[i],:]
    #                 # midPosP = midPosP[:real_nodes_num_list[i],:]
    #                 # coarsePosP = coarsePosP[:real_nodes_num_list[i],:]

    #                 fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
    #                 fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
    #                 midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
    #                 midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
    #                 coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
    #                 coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]


    #                 # coarseNormalsP2 = np.squeeze(coarseNormals2)[adjPerm_list[i]]
    #                 # coarseNormalsP2 = coarseNormalsP2[:real_nodes_num_list[i],:]
    #                 # coarseNormalsP3 = np.squeeze(coarseNormals3)[adjPerm_list[i]]
    #                 # coarseNormalsP3 = coarseNormalsP3[:real_nodes_num_list[i],:]

    #                 finalFineNormals[new_to_old_f_list[i]] = fineNormalsP
    #                 finalMidNormals[new_to_old_f_list[i]] = midNormalsP
    #                 finalCoarseNormals[new_to_old_f_list[i]] = coarseNormalsP

    #                 # finalFineNormals = fineNormalsP
    #                 # finalMidNormals = midNormalsP
    #                 # finalCoarseNormals = coarseNormalsP

    #                 finalFinePos = finePosP
    #                 finalMidPos = midPosP
    #                 finalCoarsePos = coarsePosP

    #                 # finalFinePos[new_to_old_f_list[i]] = finePosP
    #                 # finalMidPos[new_to_old_f_list[i]] = midPosP
    #                 # finalCoarsePos[new_to_old_f_list[i]] = coarsePosP

    #             else:
    #                 finalOutPoints = outPoints
    #                 finalOutPointsMid = outPointsMid
    #                 finalOutPointsCoarse = outPointsCoarse
    #                 pointsWeights +=1

    #                 # fineNormalsP = np.squeeze(fineNormals)
    #                 # midNormalsP = np.squeeze(midNormals)
    #                 # coarseNormalsP = np.squeeze(coarseNormals)

    #                 fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
    #                 fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
    #                 midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
    #                 midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
    #                 coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
    #                 coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]

    #                 # coarseNormalsP2 = np.squeeze(coarseNormals2)[adjPerm_list[i]]
    #                 # coarseNormalsP2 = coarseNormalsP2[:real_nodes_num_list[i],:]
    #                 # coarseNormalsP3 = np.squeeze(coarseNormals3)[adjPerm_list[i]]
    #                 # coarseNormalsP3 = coarseNormalsP3[:real_nodes_num_list[i],:]

    #                 # finePosP = np.squeeze(finePos)
    #                 # midPosP = np.squeeze(midPos)
    #                 # coarsePosP = np.squeeze(coarsePos)

    #                 finePosP = np.squeeze(finePos)[adjPerm_list[i]]
    #                 finePosP = finePosP[:real_nodes_num_list[i],:]
                    
    #                 midPosP = np.squeeze(midPos)[adjPerm_list[i]]
    #                 midPosP = midPosP[:real_nodes_num_list[i],:]
                    
    #                 coarsePosP = np.squeeze(coarsePos)[adjPerm_list[i]]
    #                 coarsePosP = coarsePosP[:real_nodes_num_list[i],:]
                    
    #                 finalFinePos = finePosP
    #                 finalMidPos = midPosP
    #                 finalCoarsePos = coarsePosP

    #                 finalFineNormals = fineNormalsP
    #                 finalMidNormals = midNormalsP
    #                 finalCoarseNormals = coarseNormalsP
    #                 # finalCoarseNormals2 = coarseNormalsP2
    #                 # finalCoarseNormals3 = coarseNormalsP3
    #             print("Mesh update: check")
    #         sess.close()

    #         finalOutPoints = np.true_divide(finalOutPoints,np.maximum(pointsWeights,1))
    #         finalOutPointsMid = np.true_divide(finalOutPointsMid,np.maximum(pointsWeights,1))
    #         finalOutPointsCoarse = np.true_divide(finalOutPointsCoarse,np.maximum(pointsWeights,1))

    #         # return finalOutPoints, finalFineNormals, finalMidNormals, finalCoarseNormals, finalCoarseNormals, finalCoarseNormals, finalFinePos, finalMidPos, finalCoarsePos
    #         return finalOutPoints, finalOutPointsMid, finalOutPointsCoarse, finalFineNormals, finalMidNormals, finalCoarseNormals, finalFinePos, finalMidPos, finalCoarsePos





def inferNet(inputColor_list, pos_list, adjMat_list):
    outColor_list = []
    with tf.Graph().as_default():
        # random_seed = 0
        # np.random.seed(random_seed)

        sess = tf.InteractiveSession()
        if(FLAGS.debug):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)


        # --- Graph Construction ---

        # -- Input nodes --
        BATCH_SIZE=1
        NUM_IN_CHANNELS = inputColor_list[0].shape[2]
        # NUM_CAM = inputColor_list[0].shape[2]
        NUM_CAM = 20
        TOTAL_CAM = int(NUM_IN_CHANNELS/3)
        # NODES_NUM = inputColor.shape[1]
        K_faces = adjMat_list[0].shape[2]

        # Data reorganising for cam-wise conv network:
        for patch in range(len(inputColor_list)):
            samp = inputColor_list[patch]
            inputColor_list[patch] = np.reshape(samp,[BATCH_SIZE,-1,TOTAL_CAM,3])


        # inColors_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='inColors_')
        inColors_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, TOTAL_CAM,3], name='inColors_')

        adjMat_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='adjMat_')
        inPos_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='inPos_')
        # sample_ind = tf.placeholder(tf.int32, shape=[10000], name='sample_ind')
        keep_prob = tf.placeholder(tf.float32)
        # rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
        batch = tf.Variable(0, trainable=False)




        # -- Network --
        # fullInput_ = tf.concat([inPos_,inColors_],axis=-1)
        with tf.variable_scope("model"):
            # outColor_ = getSRModel(fullInput_, adjMat_, ARCHITECTURE, keep_prob)
            outColor_ = allCamNet(inPos_, inColors_, adjMat_, NUM_CAM, ARCHITECTURE, keep_prob)

        # -- Get model --
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("ERROR! Neural network not found! Aborting mission.")
            return        

        for i in range(len(inputColor_list)):

            my_feed_dict = {inColors_: inputColor_list[i], adjMat_: adjMat_list[i], inPos_: pos_list[i], keep_prob:1}

            outColor = sess.run(outColor_,feed_dict=my_feed_dict)
            outColor_list.append(outColor)

            testOutColor = outColor
            maxOutColor = np.amax(testOutColor)
            minOutColor = np.amin(testOutColor)
            avgOutColor = np.mean(testOutColor)
            # sliceOutColor = testOutColor[0,:5,:]
            print("Output check-up:")
            print("(min, max) color: (%f,%f), avg value = %f"%(minOutColor,maxOutColor,avgOutColor))
            testOutColor = inputColor_list[i]
            maxOutColor = np.amax(testOutColor)
            minOutColor = np.amin(testOutColor)
            avgOutColor = np.mean(testOutColor)
            print("Input check-up:")
            print("(min, max) color: (%f,%f), avg value = %f"%(minOutColor,maxOutColor,avgOutColor))

            # print("Example slice: ",sliceOutColor)


        sess.close()

        

        return outColor_list


def trainNet(input_list, GT_list, adj_list, valid_input_list, valid_GT_list, valid_adj_list):
    
    # random_seed = 0
    # np.random.seed(random_seed)

    sess = tf.InteractiveSession(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=False))
    # sess = tf.InteractiveSession()
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(NETWORK_PATH):
            os.makedirs(NETWORK_PATH)


    """
    Load dataset 
    x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
    adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                              K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
    """
    BATCH_SIZE=input_list[0].shape[0]
    BATCH_SIZE=1
    K_faces = adj_list[0].shape[2]
    NUM_IN_CHANNELS = input_list[0].shape[2]
    NUM_CAM = 20
    TOTAL_CAM = int(NUM_IN_CHANNELS/3)


    # Data reorganising for cam-wise conv network:
    for patch in range(len(input_list)):
        samp = input_list[patch]
        input_list[patch] = np.reshape(samp,[BATCH_SIZE,-1,TOTAL_CAM,3])

    for patch in range(len(valid_input_list)):
        samp = valid_input_list[patch]
        valid_input_list[patch] = np.reshape(samp,[BATCH_SIZE,-1,TOTAL_CAM,3])


    
    # training data
    # inColors_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='inColors_')
    inColors_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, TOTAL_CAM,3], name='inColors_')
    
    adjMat_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='adjMat_')


    gtColor_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='gtColor_')
    inPos_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='inPos_')

    samples_num = 1000
    sample_ind = tf.placeholder(tf.int32, shape=[samples_num], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    bAddRot=True
    if bAddRot:

        inPos_rot = tf.reshape(inPos_,[BATCH_SIZE,-1,3,1])
        inPos_rot = tf.matmul(rot_mat,inPos_rot)
        inPos_rot = tf.reshape(inPos_rot,[BATCH_SIZE,-1,3])

    else:
        inPos_rot = inPos_

    # fullInput_ = tf.concat([inPos_rot,inColors_],axis=-1)

    with tf.variable_scope("model"):
        # outColor_ = getSRModel(fullInput_, adjMat_, ARCHITECTURE, keep_prob)
        outColor_ = allCamNet(inPos_rot, inColors_, adjMat_, NUM_CAM, ARCHITECTURE, keep_prob)




    isNanNConv = tf.reduce_any(tf.is_nan(outColor_), name="isNanNConv")
    isFullNanNConv = tf.reduce_all(tf.is_nan(outColor_), name="isFullNanNConv")

    fakenodes=tf.equal(adjMat_[:,:,1],0)

    with tf.device(DEVICE):

        samp_n = tf.transpose(outColor_,[1,0,2])
        samp_n = tf.gather(samp_n,sample_ind)
        samp_n = tf.transpose(samp_n,[1,0,2])

        samp_gtn = tf.transpose(gtColor_,[1,0,2])
        samp_gtn = tf.gather(samp_gtn,sample_ind)
        samp_gtn = tf.transpose(samp_gtn,[1,0,2])

        samp_fakenodes = tf.transpose(fakenodes,[1,0])
        samp_fakenodes = tf.gather(samp_fakenodes,sample_ind)
        samp_fakenodes = tf.transpose(samp_fakenodes,[1,0])
        
        # customLoss = mseLoss(samp_n, samp_gtn)
        # customLoss = maeLoss(samp_n, samp_gtn,samp_fakenodes)
        customLoss = maeLoss(samp_n, samp_gtn)
        # customLoss = binaryMSELoss(outColor_,gtColor_,adjMat_)
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
    forbidden_examples = []

    # evalStepNum=50
    # SAVEITER = 5000

    totalSteps = 0
    evalStepNum=50
    SAVEITER = 50
    # NUM_EPOCHS = 2000

    examples_num = len(input_list)
    valid_examples_num = len(valid_input_list)
    print("valid_examples_num = ",valid_examples_num)
    print("examples_num = ",examples_num)
    
    lossArray = np.zeros([int(NUM_EPOCHS),2])

    with tf.device(DEVICE):

        # Epochs loop
        for epoch in range(NUM_EPOCHS):
            print("epoch ",epoch)
            epoch_order = np.random.permutation(examples_num)
            cam_order = np.random.permutation(TOTAL_CAM)
            isComplete=False
            it = 0
            train_loss=0
            
            # Training iteration loop
            while not isComplete:
                startInd = BATCH_SIZE*it
                # endInd = BATCH_SIZE*(it+1)
                # if endInd>=examples_num:
                #     isComplete=True     # We've reached the end of the dataset
                #     batchSlice = epoch_order[startInd:]
                #     if batchSlice.shape[0]<BATCH_SIZE:  
                #         batchSlice = np.concatenate((batchSlice,epoch_order[:BATCH_SIZE-batchSlice.shape[0]]))
                #         print("shape inRays[batchSlice] = ",inRays[batchSlice].shape)
                # else:
                #     batchSlice = epoch_order[startInd:endInd]
                if startInd+1>=examples_num:
                    isComplete=True
                batchSlice = epoch_order[startInd]

                num_p = input_list[batchSlice].shape[1]
                random_ind = np.random.randint(num_p,size=samples_num)
                random_R = rand_rotation_matrix()
                tens_random_R = np.reshape(random_R,(1,1,3,3))
                tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
                # print("GT shpae = ",GT_list[batchSlice].shape)
                # print("input_list shpae = ",input_list[batchSlice].shape)
                # print("adj_list shpae = ",adj_list[batchSlice].shape)
                temp_GT = GT_list[batchSlice][:,:,3:]
                temp_Pos = GT_list[batchSlice][:,:,:3]
                # print("temp_in shape = ",input_list[batchSlice].shape)
                temp_in = np.transpose(input_list[batchSlice],[2,0,1,3])
                # print("temp_in shape = ",temp_in.shape)
                np.random.shuffle(temp_in)
                # print("temp_in shape = ",temp_in.shape)
                temp_in = np.transpose(temp_in,[1,2,0,3])
                # print("temp_in shape = ",temp_in.shape)
                train_fd = {inColors_: temp_in, gtColor_: temp_GT, adjMat_: adj_list[batchSlice], inPos_: temp_Pos, rot_mat:tens_random_R2, sample_ind: random_ind, keep_prob:1}

                # out0, in0 = sess.run([outColor_, fullInput_],feed_dict=train_fd)
                # # print("in0 slice = ",in0[0,:5,:12])
                # print("out0 slice = ",out0[0,:5,:])

                # goodInput = input_list[batchSlice]
                # mixedInput = np.copy(goodInput)
                # mixedInput[:,:,:3] = goodInput[:,:,6:9]
                # mixedInput[:,:,6:9] = goodInput[:,:,:3]
                # train_fd1 = {inColors_: mixedInput, gtColor_: temp_GT, adjMat_: adj_list[batchSlice], inPos_: temp_Pos, rot_mat:tens_random_R2, sample_ind: random_ind, keep_prob:1}
                # out1, in1 = sess.run([outColor_, fullInput_],feed_dict=train_fd1)
                # # print("in1 slice = ",in1[0,:5,:12])
                # print("out1 slice = ",out1[0,:5,:])

                # mixedInput = np.copy(goodInput)
                # mixedInput[:,:,:3] = goodInput[:,:,9:12]
                # # mixedInput[:,:,6:9] = goodInput[:,:,:3]
                # train_fd2 = {inColors_: mixedInput, gtColor_: temp_GT, adjMat_: adj_list[batchSlice], inPos_: temp_Pos, rot_mat:tens_random_R2, sample_ind: random_ind, keep_prob:1}
                # out2, in2 = sess.run([outColor_, fullInput_],feed_dict=train_fd2)
                # # print("in2 slice = ",in2[0,:5,:12])
                # print("out2 slice = ",out2[0,:5,:])

                sess.run(train_step,feed_dict=train_fd)
            
                train_loss += customLoss.eval(feed_dict=train_fd)
                it+=1
                totalSteps+=1


            testOutColor = sess.run(outColor_,feed_dict=train_fd)
            maxOutColor = np.amax(testOutColor)
            minOutColor = np.amin(testOutColor)
            avgOutColor = np.mean(testOutColor)
            sliceOutColor = testOutColor[0,:5,:]
            # print("Output check-up: (min, max) color: (%f,%f), avg value = %f"%(minOutColor,maxOutColor,avgOutColor))
            # print("Example slice: ",sliceOutColor)

            testGTColor = sess.run(gtColor_,feed_dict=train_fd)
            maxGTColor = np.amax(testGTColor)
            minGTColor = np.amin(testGTColor)
            avgGTColor = np.mean(testGTColor)
            sliceGTColor = testGTColor[0,:5,:]
            # print("GT check-up: (min, max) color: (%f,%f), avg value = %f"%(minGTColor,maxGTColor,avgGTColor))
            # print("Example slice: ",sliceGTColor)
            testInColor = sess.run(inColors_,feed_dict=train_fd)
            maxInColor = np.amax(testInColor)
            minInColor = np.amin(testInColor)
            avgInColor = np.mean(testInColor)
            sliceInColor = testInColor[0,:5,12:]
            # print("Input check-up: (min, max) color: (%f,%f), avg value = %f"%(minInColor,maxInColor,avgInColor))
            # print("Example slice: ",sliceInColor)

            lossArray[epoch,0] = train_loss/it

            if (epoch%SAVEITER == 0)and(epoch>0):
                saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+epoch)
                print("Ongoing training: architecture "+str(ARCHITECTURE)+", net path = "+NETWORK_PATH)
                # if sess.run(isFullNanNConv, feed_dict=train_fd):
                #     break

            # Validation:

            it = 0
            valid_loss=0
            validIsComplete = False
            valid_perm = np.arange(valid_examples_num)
            while not validIsComplete:
                startInd = BATCH_SIZE*it
                # endInd = BATCH_SIZE*(it+1)
                # if endInd>=valid_examples_num:
                #     validIsComplete=True
                #     validSlice = valid_perm[startInd:]
                #     if validSlice.shape[0]<BATCH_SIZE:  
                #         validSlice = np.concatenate((validSlice,valid_perm[:BATCH_SIZE-validSlice.shape[0]]))
                # else:
                #     validSlice = valid_perm[startInd:endInd]
                if startInd+1>=valid_examples_num:
                    validIsComplete=True
                validSlice = valid_perm[startInd]
                # print("validSlice = ",validSlice)
                num_p = valid_input_list[validSlice].shape[1]
                random_ind = np.random.randint(num_p,size=samples_num)
                random_R = rand_rotation_matrix()
                tens_random_R = np.reshape(random_R,(1,1,3,3))
                tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
                temp_GT = valid_GT_list[validSlice][:,:,3:]
                temp_Pos = valid_GT_list[validSlice][:,:,:3]
                valid_fd = {inColors_: valid_input_list[validSlice], gtColor_: temp_GT, adjMat_: valid_adj_list[validSlice], inPos_: temp_Pos, rot_mat:tens_random_R2, sample_ind: random_ind, keep_prob:1}

                valid_loss += customLoss.eval(feed_dict=valid_fd)
                it+=1


            lossArray[epoch,1] = valid_loss/it

            print("Validation loss = ",valid_loss/it)

                    # lossArray = np.zeros([int(NUM_ITERATIONS/evalStepNum),2])
                    # last_loss = 0
                    # for iter in range(NUM_ITERATIONS):

                    #     # Get random sample from training dictionary
                    #     batch_num = random.randint(0,len(f_normals_list)-1)

                    #     while batch_num in forbidden_examples:
                    #         batch_num = random.randint(0,len(f_normals_list)-1)
                    #     num_p = f_normals_list[batch_num].shape[1]
                    #     # print("num_p = ",num_p)
                    #     random_ind = np.random.randint(num_p,size=10000)
                    #     # random_ind = np.random.randint(1,size=10000)

                    #     random_R = rand_rotation_matrix()
                    #     tens_random_R = np.reshape(random_R,(1,1,3,3))
                    #     tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

                    #     train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
                    #                     fadj2: f_adj_list[batch_num][2], tfn_: GTfn_list[batch_num], rot_mat:tens_random_R2,
                    #                     sample_ind: random_ind, keep_prob:1}

                    #     # print("OK?")
                    #     if len(f_adj_list[0])>3:
                    #         train_fd[fadj3]=f_adj_list[batch_num][3]
                    #     #i = train_shuffle[iter%(len(train_data))]
                    #     #in_points = train_data[i]

                    #     #sess.run(customLoss,feed_dict=train_fd)

                    #     # print("OK")
                    #     train_loss += customLoss.eval(feed_dict=train_fd)
                    #     train_samp+=1
                    #     # print("Still OK!")
                    #     # Show smoothed training loss
                    #     if (iter%evalStepNum == 0):
                    #         train_loss = train_loss/train_samp

                    #         print("Iteration %d, training loss %g"%(iter, train_loss))

                    #         lossArray[int(iter/evalStepNum),0]=train_loss
                    #         train_loss=0
                    #         train_samp=0

                    #     # Compute validation loss
                    #     if (iter%(evalStepNum*2) ==0):
                    #         valid_loss = 0
                    #         valid_samp = len(valid_f_normals_list)

                    #         valid_random_ind = np.random.randint(num_p,size=10000)
                    #         # valid_random_ind = np.random.randint(1,size=10000)
                    #         for vbm in range(valid_samp):

                    #             num_p = valid_f_normals_list[vbm].shape[1]
                    #             tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

                    #             valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                    #                     fadj2: valid_f_adj_list[vbm][2], tfn_: valid_GTfn_list[vbm], rot_mat:tens_random_R2,
                    #                     sample_ind: valid_random_ind, keep_prob:1}

                    #             if len(f_adj_list[0])>3:
                    #                 valid_fd[fadj3]=valid_f_adj_list[vbm][3]

                    #             valid_loss += customLoss.eval(feed_dict=valid_fd)
                    #         valid_loss/=valid_samp
                    #         print("Iteration %d, validation loss %g"%(iter, valid_loss))
                    #         lossArray[int(iter/evalStepNum),1]=valid_loss
                    #         if iter>0:
                    #             lossArray[int(iter/evalStepNum)-1,1] = (valid_loss+last_loss)/2
                    #             last_loss=valid_loss

                    #     sess.run(train_step,feed_dict=train_fd)

                    #     if sess.run(isNanNConv,feed_dict=train_fd):
                    #         hasNan = True
                    #         print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(f_normals_list)))
                    #         print("patch size: "+str(f_normals_list[batch_num].shape))
                    #     if (iter%SAVEITER == 0):
                    #         saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
                    #         print("Ongoing training: architecture "+str(ARCHITECTURE)+", net path = "+RESULTS_PATH)
                    #         if sess.run(isFullNanNConv, feed_dict=train_fd):
                    #             break
    
    saver.save(sess, NETWORK_PATH+NET_NAME,global_step=globalStep+NUM_EPOCHS)

    sess.close()
    csv_filename = NETWORK_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()


def mseLoss(prediction, gt):
    loss = tf.square(tf.subtract(gt,prediction))
    loss = tf.reduce_mean(loss)
    return loss

# def maeLoss(prediction,gt,fakenodes):
#     loss = tf.abs(tf.subtract(gt,prediction))
#     loss = tf.reduce_sum(loss,axis=-1)

#     zeroVec = tf.zeros_like(loss)
#     oneVec = tf.ones_like(loss)
#     realnodes = tf.where(fakenodes,zeroVec,oneVec)

#     loss = tf.where(fakenodes,zeroVec,loss)
#     loss = tf.reduce_sum(loss)/tf.reduce_sum(realnodes)

#     # loss = tf.reduce_mean(loss)
#     return loss

def maeLoss(prediction,gt):
    loss = tf.abs(tf.subtract(gt,prediction))
    loss = tf.reduce_mean(loss)
    return loss


def binaryMSELoss(prediction,gt,adjMat):

    predPatches = get_patches(prediction,adjMat)
    # [batch, N, K, ch]
    gtPatches = get_patches(gt,adjMat)
    predDiff = predPatches-tf.expand_dims(prediction,axis=2)
    gtDiff = gtPatches-tf.expand_dims(gt,axis=2)

    loss = tf.square(tf.subtract(gtDiff,predDiff))
    loss = tf.reduce_mean(loss)
    return loss


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




# Repickle all files in folder with protocol 2 for python 2
def repickleFolder(sourceFolder, destFolder):

    for filename in os.listdir(sourceFolder):
        with open(sourceFolder+"/"+filename, 'rb') as fp:
            pickleRick = pickle.load(fp)
        with open(destFolder+"/"+filename, 'wb') as fp:
                pickle.dump(pickleRick, fp, protocol=2)
        print("pickle "+filename+": check")


def getFolderAvgEdgeLength(folderPath, normalize=False):
    tot_weight=0
    tot_el=0
    for filename in os.listdir(folderPath):

        print("loading "+filename)
        V0,_,_, faces0, _ = load_mesh(folderPath, filename, 0, False)

        cur_el, cur_tot = getAverageEdgeLength(V0, faces0, normalize)

        tot_weight += cur_tot
        tot_el += cur_el*cur_tot

    tot_el /= tot_weight
    # print("edge length = "+str(tot_el))
    return tot_el



def pickleFND(folderPath, binDumpPath):
    sigma_r_list = [0.1,0.2,0.35,0.5,-1]
    # sigma_r_list = [-1]
    sigma_s_list = [AVG_EDGE_LENGTH,2*AVG_EDGE_LENGTH]

    for filename in os.listdir(folderPath):

        # if filename.startswith("dragon"):
        #     continue

        if not os.path.isfile(binDumpPath+filename+'FND'):
            print("loading "+filename)
            V0,_,_, faces0, _ = load_mesh(folderPath, filename, 0, False)

            my_el, _ = getAverageEdgeLength(V0, faces0)
            print("avg edge length = "+str(my_el))
            cur_sigma_s_list = [my_el, 2*my_el]
            # Compute normals
            f_normals0 = computeFacesNormals(V0, faces0)
            # Get faces position
            f_pos0 = getTrianglesBarycenter(V0, faces0)

            f_area0 = getTrianglesArea(V0,faces0)

            
            f_FND = FND(f_pos0, f_normals0, f_area0, cur_sigma_s_list, sigma_r_list)
            
            with open(binDumpPath+filename+'FND', 'wb') as fp:
                pickle.dump(f_FND, fp)



def checkFND(filePath, filename):

    V0,_,_, faces_noisy, _ = load_mesh(filePath, filename, 0, False)

    binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/Synthetic/BinaryDump/FND_el_c4/"
    with open(binDumpPath+filename+'FND', 'rb') as fp:
        fnd = pickle.load(fp)

    for fil in range(10):

        cur_n = fnd[:,3*fil:3*fil+3]

        colormap = (cur_n+1)/2
        newV, newF = getColoredMesh(V0, faces_noisy, colormap)
        
        write_mesh(newV, newF, RESULTS_PATH+filename[:-4]+"_FND_"+str(fil)+".obj")



def getDepthDirection():

    noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
    gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

    noisyFile = "boy_01_noisy.obj"
    gtFile = "boy_01.obj"

    # Load GT
    GT0,_,_,_,_ = load_mesh(gtFolder, gtFile, 0, False)
    V0,_,_,_,_ = load_mesh(noisyFolder, noisyFile, 0, False)

    print("V0 shape = "+str(V0.shape))

    VDiff = V0-GT0

    VDiffN = normalize(VDiff)

    print("Test: "+str(VDiffN[:10,:]))

    VDiffN_p = VDiffN[VDiffN[:,0]>0]

    std_p = np.std(VDiffN_p, axis=0)
    mean_p = np.mean(VDiffN_p, axis=0)

    dev_p = ((VDiffN_p[:,0]-mean_p[0])**2)
    std_p = np.std(VDiffN_p, axis=0)
    max_std = np.argmax(dev_p, axis=0)
    print("mean VDiffN_p = "+str(mean_p))
    print("std VDiffN_p = "+str(std_p))

    print("max dev: "+str(VDiffN_p[max_std,:]))

    norm_VDiff = np.linalg.norm(VDiff,axis=-1)

    arg_max_diff = np.argmax(norm_VDiff)

    print("Max diff = "+str(VDiff[arg_max_diff,:]))
    print("dir: "+str(VDiffN[arg_max_diff,:]))

    VDiff_p = VDiff[VDiff[:,0]>0]

    print("mean norm_VDiff = "+str(np.mean(norm_VDiff)))
    print("max norm_VDiff = "+str(np.max(norm_VDiff)))

    my_th = 0.000001*np.mean(norm_VDiff) + 0.999999*np.max(norm_VDiff)

    testDiffN = VDiffN[norm_VDiff>my_th]

    print("testDiffN shape = "+str(testDiffN.shape))

    print("testDiffN = "+str(testDiffN))

    testDiffN_p = testDiffN[testDiffN[:,0]>0]

    print("mean testDiffN_p = "+str(np.mean(testDiffN_p,axis=0)))

    testDiffN_n = testDiffN[testDiffN[:,0]<0]

    print("mean testDiffN_n = "+str(np.mean(testDiffN_n,axis=0)))

    return


def mainFunction():


    pickleLoad = True
    pickleSave = True

    K_faces = 23

    maxSize = 5000 #35000
    patchSize = 5000 #15000
    maxPatch = 50

    training_meshes_num = [0]
    valid_meshes_num = [0]

    empiricMax = 30.0

    # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/Synthetic/BinaryDump/normals_c4/"

    running_mode = RUNNING_MODE
    ###################################################################################################
    #   0 - Training on all meshes in a given folder
    #   1 - Run checkpoint on a given mesh as input
    #   2 - Run checkpoint on all meshes in a folder. Compute angular diff and Haus dist
    #   3 - Test mode
    ###################################################################################################


    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMesh(inputFilePath,filename, in_list, gt_list, adj_list, mesh_count_list):
        patch_indices = []

        with open(inputFilePath+"/"+pickleGtMatFormat%filename, 'rb') as fp:
            gtMat = pickle.load(fp)

        nodesPos = gtMat[:,:3]
        nodesColor = gtMat[:,3:]

        with open(inputFilePath+"/"+pickleAdjMatFormat%filename, 'rb') as fp:
            adjMat = pickle.load(fp)
        adjMat = adjMat+1

        with open(inputFilePath+"/"+pickleInputFormat%filename, 'rb') as fp:
            nodesInput = pickle.load(fp)

        nodesInput = nodesInput.astype(float)
        nodesInput = nodesInput/255.0

        # Get patches if mesh is too big
        nodesNum = nodesPos.shape[0]

        # Remove useless nodes
        rangeInd = np.arange(nodesNum)
        goodInd = rangeInd[adjMat[:,1]>0]

        print("nodesInput size = ",nodesInput.shape)
        nodesInput = nodesInput[goodInd]
        gtMat = gtMat[goodInd]

        goodIndAdj = np.concatenate([[0],goodInd+1],axis=0)

        print("max adjMat = ",np.amax(adjMat))
        print("adjMat size = ",adjMat.shape[0])
        print("nodesNum = ",nodesNum)
        print("goodIndAdj size = ",goodIndAdj.shape)
        print("max goodIndAj = ",np.amax(goodIndAdj))
        invGoodInd = inv_perm(goodIndAdj)
        newAdj = invGoodInd[adjMat]
        newAdj = newAdj[goodInd]

        adjMat = newAdj

        if BOOL_N2:
            adjMat = getN2Adj(adjMat,40)

        # Get patches if mesh is too big
        nodesNum = nodesInput.shape[0]

        nodesCheck = np.zeros(nodesNum)
        nodesRange = np.arange(nodesNum)
        if nodesNum>maxSize:
            patchNum = 0
            while(np.any(nodesCheck==0) and patchNum<maxPatch):
                toBeProcessed = nodesRange[nodesCheck==0]
                nodeSeed = np.random.randint(toBeProcessed.shape[0])
                nodeSeed = toBeProcessed[nodeSeed]

                patchAdjMat, nodesOldInd = getGraphPatch(adjMat, patchSize, nodeSeed)

                nodesCheck[nodesOldInd]+=1

                patchGT = gtMat[nodesOldInd]
                patchInput = nodesInput[nodesOldInd]

                old_N = patchGT.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue

                ##### Save number of triangles and patch new_to_old permutation #####
                patch_indices.append(nodesOldInd)
                #####################################################################

                

                # Expand dimensions
                patchInput = np.expand_dims(patchInput, axis=0)
                patchAdjMat = np.expand_dims(patchAdjMat, axis=0)
                patchGT = np.expand_dims(patchGT, axis=0)

                in_list.append(patchInput)
                adj_list.append(patchAdjMat)
                gt_list.append(patchGT)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                processedNodesNum = np.sum(nodesCheck>0)
                print("%d%% nodes processed (%i out of %i)"%(100*processedNodesNum/nodesNum, processedNodesNum, nodesNum))
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small graph case

            ##### Save number of triangles and patch new_to_old permutation #####
            patch_indices.append([])
            #####################################################################

            # Expand dimensions
            nodesInput = np.expand_dims(nodesInput, axis=0)
            adjMat = np.expand_dims(adjMat, axis=0)
            gtMat = np.expand_dims(gtMat, axis=0)

            in_list.append(nodesInput)
            adj_list.append(adjMat)
            gt_list.append(gtMat)
        
            # print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return patch_indices, nodesNum

    
    

    # Train network
    if running_mode == 0:

        in_list = []
        adj_list = []
        GT_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_GTfn_list = []


        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"


        addMesh(dataFolder, dataBaseName, in_list, GT_list, adj_list, training_meshes_num)

        # in_list = np.concatenate(in_list,axis=0)
        # GT_list = np.concatenate(GT_list,axis=0)
        # adj_list = np.concatenate(adj_list,axis=0)

        valid_in_list = in_list[-5:]
        valid_GT_list = GT_list[-5:]
        valid_adj_list = adj_list[-5:]

        in_list = in_list[:-5]
        GT_list = GT_list[:-5]
        adj_list = adj_list[:-5]

        # maxColor = np.amax(in_list[0])
        # print("max color = ",maxColor)
        # print("in_list type = ",in_list[0].dtype)

        # maxColorGT = np.amax(GT_list[0][:,:,3:])
        # print("max GT color = ",maxColorGT)
        # maxPosGT = np.amax(GT_list[0][:,:,:3])
        # print("max pos = ",maxPosGT)
        # print("GT_list type = ",GT_list[0].dtype)

        # Normalize data
        for i in range(len(in_list)):
            in_list[i] = 2*in_list[i]-1
            GTcolor = GT_list[i][:,:,3:]
            GTcolor = 2*GTcolor-1
            GT_list[i][:,:,3:]=GTcolor

        print("size in_list = ",len(in_list))
        print("size GT_list = ",len(GT_list))
        print("size adj_list = ",len(adj_list))
        trainNet(in_list, GT_list, adj_list, valid_in_list, valid_GT_list, valid_adj_list)

            # # Simple inference, no GT mesh involved
            # elif running_mode == 1:
                
            #     maxSize = 100000
            #     patchSize = 100000

            #     # noisyFolder = "/morpheo-nas2/vincent/DTU_Robot_Image_Dataset/Surface/furu/"


            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/test_bits/"
            #     noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/test/"
            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/Kinovis/"
            #     noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/tola/test_bits/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid_cleaned/"
            #     # noisyFolder = "/morpheo-nas2/marmando/smpl/"
            #     # noisyFolder = "/morpheo-nas2/marmando/densepose/Test/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/results/1stStep/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/kickVH/results/1stStep/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/realData/vincentSlip/decimVH/"
            #     # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/realData/vincentSlip/results/1stStep/"
            #     # noisyFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/registrations/"
            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"

            #     # Get GT mesh
            #     for noisyFile in os.listdir(noisyFolder):

                    
            #         if (not noisyFile.endswith(".obj")):
            #             continue
            #         print("noisyFile: "+noisyFile)
            #         if (noisyFile.startswith("arma")):
            #             continue
            #         mesh_count = [0]


            #         denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"


            #         noisyFilesList = [noisyFile]
            #         denoizedFilesList = [denoizedFile]

            #         for fileNum in range(len(denoizedFilesList)):
                        
            #             # randToDel = np.random.rand(1)
            #             # if randToDel>0.1:
            #             #     continue
            #             denoizedFile = denoizedFilesList[fileNum]
            #             noisyFile = noisyFilesList[fileNum]
            #             # noisyFileWInferredColor = noisyFile[:-4]+"_inferred_normals.obj"

            #             noisyFileWInferredColor0 = noisyFile[:-4]+"_fine_normals_s.obj"
            #             noisyFileWInferredColor1 = noisyFile[:-4]+"_mid_normals_s.obj"
            #             noisyFileWInferredColor2 = noisyFile[:-4]+"_coarse_normals_s.obj"
            #             noisyFileWInferredColor3 = noisyFile[:-4]+"_coarse_normals2_s.obj"
            #             noisyFileWInferredColor4 = noisyFile[:-4]+"_coarse_normals3_s.obj"

            #             noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
            #             denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"
            #             faceMeshFile = noisyFile[:-4]+"_face_mesh.obj"
            #             faceMeshFile1 = noisyFile[:-4]+"_face_mesh1.obj"
            #             faceMeshFile2 = noisyFile[:-4]+"_face_mesh2.obj"

            #             if not os.path.isfile(RESULTS_PATH+denoizedFile):
            #             # if True:

            #                 f_normals_list = []
            #                 GTfn_list = []
            #                 f_adj_list = []
            #                 faces_list = []
            #                 v_faces_list = []
            #                 v_list = []
            #                 gtv_list = []


            #                 V0,_,_, faces_noisy, _ = load_mesh(noisyFolder, noisyFile, 0, False)
            #                 f_normals0 = computeFacesNormals(V0, faces_noisy)

            #                 print("Adding mesh "+noisyFile+"...")
            #                 t0 = time.clock()
            #                 # faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, noisyFolder, noisyFile, faces_list, f_normals_list, GTfn_list, f_adj_list, mesh_count)
            #                 vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list, = addMeshWithVertices(noisyFolder, noisyFile, noisyFolder, noisyFile, v_list, gtv_list, faces_list, f_normals_list, f_adj_list, v_faces_list, mesh_count)
            #                 print("mesh added ("+str(1000*(time.clock()-t0))+"ms)")
            #                 # Now recover vertices positions and create Edge maps

            #                 # with open("/morpheo-nas2/marmando/ShapeRegression/BinaryDump/adj3lvl.pkl", 'wb') as fp:
            #                 #     pickle.dump(f_adj_list[0], fp)
            #                 # with open("/morpheo-nas2/marmando/ShapeRegression/BinaryDump/perm3lvl.pkl", 'wb') as fp:
            #                 #     pickle.dump(inv_perm(adjPerm_list[0]), fp) 

            #                 # return
            #                 V0 = np.expand_dims(V0, axis=0)

            #                 # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            #                 # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
                            

            #                 faces_noisy = np.array(faces_noisy).astype(np.int32)
            #                 faces = np.expand_dims(faces_noisy,axis=0)

            #                 # v_faces = getVerticesFaces(np.squeeze(faces_list[0]), 15, V0.shape[1])
            #                 # v_faces = np.expand_dims(v_faces,axis=0)

            #                 print("Inference ...")
            #                 t0 = time.clock()
            #                 #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
            #                 # upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, faces_num, patch_indices, permutations,facesNum)
            #                 # upV0, upN0, upN1, upN2, upN3, upN4, upP0, upP1, upP2 = inferNet(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
            #                 upV0, upV0mid, upV0coarse, upN0, upN1, upN2, upP0, upP1, upP2 = inferNet(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
                            
            #                 # upV0, upN0 = inferNet6D(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
            #                 print("Inference complete ("+str(1000*(time.clock()-t0))+"ms)")

            #                 # write_mesh(np.concatenate((upV0,np.zeros_like(upV0)),axis=-1), faces[0,:,:], RESULTS_PATH+denoizedFile)
            #                 write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)
            #                 write_mesh(upV0mid, faces[0,:,:], RESULTS_PATH+noisyFile[:-4]+"_d_mid.obj")
            #                 write_mesh(upV0coarse, faces[0,:,:], RESULTS_PATH+noisyFile[:-4]+"_d_coarse.obj")
            #                 # write_mesh(V0, faces[0,:,:], RESULTS_PATH+noisyFile[:-4]+"_test.obj")

            #                 # testP = upP0
            #                 # testN = upN0
            #                 # # testP = f_normals_list[0][:,:,3:]
            #                 # # testP = np.squeeze(testP)
            #                 # testAdj = f_adj_list[0][0]
            #                 # testAdj = np.squeeze(testAdj)
            #                 # # testN = f_normals_list[0][:,:,:3]
            #                 # # testN = np.squeeze(testN)
            #                 # faceMeshV, faceMeshF = makeFacesMesh(testAdj,testP,testN)

            #                 # write_mesh(faceMeshV, faceMeshF, RESULTS_PATH+faceMeshFile)
                            
            #                 # testP1 = upP1
            #                 # testN1 = upN1
            #                 # testAdj1 = f_adj_list[0][1]
            #                 # testAdj1 = np.squeeze(testAdj1)
            #                 # faceMeshV, faceMeshF = makeFacesMesh(testAdj1,testP1,testN1)

            #                 # write_mesh(faceMeshV, faceMeshF, RESULTS_PATH+faceMeshFile1)

            #                 # testP2 = upP2
            #                 # testN2 = upN2
            #                 # testAdj2 = f_adj_list[0][2]
            #                 # testAdj2 = np.squeeze(testAdj2)
            #                 # faceMeshV, faceMeshF = makeFacesMesh(testAdj2,testP2,testN2)

            #                 # write_mesh(faceMeshV, faceMeshF, RESULTS_PATH+faceMeshFile2)

            #                 # continue



            #                 angColor0 = (upN0+1)/2
            #                 angColor1 = (upN1+1)/2
            #                 angColor2 = (upN2+1)/2
            #                 # angColor3 = (upN3+1)/2
            #                 # angColor4 = (upN4+1)/2

            #                 # f_normals0 = np.squeeze(f_normals_list[0])
            #                 # print("f_normals0 shape: "+str(f_normals0.shape))
            #                 # f_normals0 = f_normals0[:,:3]
            #                 # print("f_normals0 shape: "+str(f_normals0.shape))
            #                 angColorNoisy = (f_normals0+1)/2
                            
            #                 print("faces_noisy shape: "+str(faces_noisy.shape))

            #                 print("angColor0 shape: "+str(angColor0.shape))
            #                 print("angColor1 shape: "+str(angColor1.shape))
            #                 print("angColor2 shape: "+str(angColor2.shape))
            #                 print("V0 shape: "+str(V0.shape))
            #                 # newV, newF = getColoredMesh(upV0, faces_gt, angColor)
            #                 newVn0, newFn0 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor0)
            #                 newVn1, newFn1 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor1)
            #                 newVn2, newFn2 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor2)
            #                 # newVn3, newFn3 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor3)
            #                 # newVn4, newFn4 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor4)
                            

            #                 # write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
            #                 write_mesh(newVn0, newFn0, RESULTS_PATH+noisyFileWInferredColor0)
            #                 write_mesh(newVn1, newFn1, RESULTS_PATH+noisyFileWInferredColor1)
            #                 write_mesh(newVn2, newFn2, RESULTS_PATH+noisyFileWInferredColor2)
            #                 # write_mesh(newVn3, newFn3, RESULTS_PATH+noisyFileWInferredColor3)
            #                 # write_mesh(newVn4, newFn4, RESULTS_PATH+noisyFileWInferredColor4)

            #                 print("angColorNoisy shape: "+str(angColorNoisy.shape))
            #                 newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_noisy, angColorNoisy)
            #                 write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFileWColor)

            #                 # return

            # # master branch inference (old school, w/o multi-scale vertex update)
            # elif running_mode == 12:
                
            #     maxSize = 100000
            #     patchSize = 100000

            #     # noisyFolder = "/morpheo-nas2/vincent/DTU_Robot_Image_Dataset/Surface/furu/"


            #     noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/test_bits/"
            #     noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/test/"
            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/Kinovis/"
            #     noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid/"
            #     noisyFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/registrations/"


            #     # Get GT mesh
            #     for noisyFile in os.listdir(noisyFolder):


            #         if (not noisyFile.endswith(".obj")):
            #             continue
            #         mesh_count = [0]


            #         denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"


            #         noisyFilesList = [noisyFile]
            #         denoizedFilesList = [denoizedFile]

            #         for fileNum in range(len(denoizedFilesList)):
                        
            #             denoizedFile = denoizedFilesList[fileNum]
            #             noisyFile = noisyFilesList[fileNum]
            #             noisyFileWInferredColor = noisyFile[:-4]+"_inferred_normals.obj"
            #             noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
            #             denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"

            #             if not os.path.isfile(RESULTS_PATH+denoizedFile):
                            

            #                 f_normals_list = []
            #                 GTfn_list = []
            #                 f_adj_list = []

            #                 V0,_,_, faces_noisy, _ = load_mesh(noisyFolder, noisyFile, 0, False)
            #                 f_normals0 = computeFacesNormals(V0, faces_noisy)

            #                 print("Adding mesh "+noisyFile+"...")
            #                 t0 = time.clock()
            #                 faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, noisyFolder, noisyFile, f_normals_list, GTfn_list, f_adj_list, mesh_count)
            #                 print("mesh added ("+str(1000*(time.clock()-t0))+"ms)")
            #                 # Now recover vertices positions and create Edge maps

                            

            #                 facesNum = faces_noisy.shape[0]
            #                 V0 = np.expand_dims(V0, axis=0)

            #                 _, edge_map, v_e_map = getFacesAdj(faces_noisy)
            #                 f_adj = getFacesLargeAdj(faces_noisy,K_faces)
            #                 # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            #                 # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
                            

            #                 faces_noisy = np.array(faces_noisy).astype(np.int32)
            #                 faces = np.expand_dims(faces_noisy,axis=0)
            #                 edge_map = np.expand_dims(edge_map, axis=0)
            #                 v_e_map = np.expand_dims(v_e_map, axis=0)

            #                 print("Inference ...")
            #                 t0 = time.clock()
            #                 #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
            #                 upV0, upN0 = inferNetOld(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
            #                 print("Inference complete ("+str(1000*(time.clock()-t0))+"ms)")

            #                 write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)

            #                 angColor = (upN0+1)/2

            #                 angColorNoisy = (f_normals0+1)/2
                            
            #                 # newV, newF = getColoredMesh(upV0, faces_gt, angColor)
            #                 newVn, newFn = getColoredMesh(np.squeeze(V0), faces_noisy, angColor)
            #                 newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_noisy, angColorNoisy)

            #                 # write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
            #                 write_mesh(newVn, newFn, RESULTS_PATH+noisyFileWInferredColor)
            #                 write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFileWColor)

            # # Inference: Denoise set, save meshes (colored with heatmap), compute metrics
            # elif running_mode == 2:
                
            #     maxSize = 100000
            #     patchSize = 100000


            #     # Take the opportunity to generate array of metrics on reconstructions
            #     nameArray = []      # String array, to now which row is what
            #     resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
            #                         # [Max distance, Mean distance, Mean angle, std angle, face num]

            #     noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
            #     gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"

            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
            #     # # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
            #     # # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/Kinect_v1/Results/b22/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"
            #     # # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"

            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/test/noisy/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/test/original/"

            #     noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/noisy/"
            #     gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"

            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

            #     # noisyFolder = "/morpheo-nas/marmando/Data/Anja/look-6-2/Results/RVDs/Static/"
            #     # gtFolder = "/morpheo-nas/marmando/Data/Anja/look-6-2/Results/RVDs/Static/"

            #     # noisyFolder = "/morpheo-nas2/marmando/marmando_temp/kick540/"
            #     # gtFolder = "/morpheo-nas2/marmando/marmando_temp/kick540/"


            #     # results file name
            #     csv_filename = RESULTS_PATH+"results.csv"
                
                
            #     angDict = {}
            #     # Get GT mesh
            #     for gtFileName in os.listdir(gtFolder):

            #         nameArray = []
            #         resultsArray = []
            #         if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("aaaMerlion")):
            #             continue
            #         mesh_count = [0]

            #         # Get all 3 noisy meshes
            #         noisyFile0 = gtFileName[:-4]+"_noisy.obj"
            #         # noisyFile0 = gtFileName[:-4]+"_denoized.obj"
            #         denoizedFile0 = gtFileName[:-4]+"_denoized.obj"


            #         # noisyFile0 = gtFileName
            #         # denoizedFile0 = gtFileName[:-4]+"_denoized_synth.obj"
            #         # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            #         # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            #         # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
                    

            #         # noisyFile0 = gtFileName[:-4]+"_n1.obj"
            #         # noisyFile1 = gtFileName[:-4]+"_n2.obj"
            #         # noisyFile2 = gtFileName[:-4]+"_n3.obj"

            #         # denoizedFile0 = gtFileName[:-4]+"_denoized_gray_1.obj"
            #         # denoizedFile1 = gtFileName[:-4]+"_denoized_gray_2.obj"
            #         # denoizedFile2 = gtFileName[:-4]+"_denoized_gray_3.obj"

            #         noisyFilesList = [noisyFile0]
            #         denoizedFilesList = [denoizedFile0]

            #         # noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            #         # denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            #         isTreated=True
            #         for fileNum in range(len(denoizedFilesList)):
            #             if not os.path.isfile(RESULTS_PATH+denoizedFilesList[fileNum]):
            #                 isTreated=False
            #         if isTreated:
            #             continue
            #         # if (os.path.isfile(RESULTS_PATH+denoizedFile0)) and (os.path.isfile(RESULTS_PATH+denoizedFile1)) and (os.path.isfile(RESULTS_PATH+denoizedFile2)):
            #         #     continue

            #         # Load GT mesh
            #         GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            #         GTf_normals = computeFacesNormals(GT, faces_gt)

            #         facesNum = faces_gt.shape[0]
            #         # We only need to load faces once. Connectivity doesn't change for noisy meshes
            #         # Same for adjacency matrix

            #         _, edge_map, v_e_map = getFacesAdj(faces_gt)
            #         f_adj = getFacesLargeAdj(faces_gt,K_faces)
            #         # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            #         # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
                    

            #         faces_gt = np.array(faces_gt).astype(np.int32)
            #         faces = np.expand_dims(faces_gt,axis=0)
            #         edge_map = np.expand_dims(edge_map, axis=0)
            #         v_e_map = np.expand_dims(v_e_map, axis=0)

                    

                    
            #         for testRep in range(1):
            #             for fileNum in range(len(denoizedFilesList)):
                            
            #                 denoizedFile = denoizedFilesList[fileNum]
            #                 denoizedHeatmap = denoizedFile[:-4]+"_H.obj"
            #                 noisyFile = noisyFilesList[fileNum]

            #                 denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"
            #                 noisyFileWColor = noisyFile[:-4]+"_nC.obj"
            #                 noisyFileWGtN = noisyFile[:-4]+"_gtnC.obj"
            #                 gtFileWGtN = gtFileName[:-4]+"_GT.obj"

            #                 if not os.path.exists(noisyFolder+noisyFile):
            #                     continue

            #                 # if not noisyFile.startswith("boy_15"):
            #                 #     continue

            #                 # if True:
            #                 if not os.path.isfile(RESULTS_PATH+denoizedFile):
                            
                            
            #                     f_normals_list = []
            #                     GTfn_list = []
            #                     f_adj_list = []
            #                     v_list = []
            #                     gtv_list = []
            #                     faces_list = []
            #                     v_faces_list = []

            #                     print("Adding mesh "+noisyFile+"...")
            #                     t0 = time.clock()
            #                     # faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, gtFolder, gtFileName, f_normals_list, GTfn_list, f_adj_list, mesh_count)
            #                     vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list, = addMeshWithVertices(noisyFolder, noisyFile, gtFolder, gtFileName, v_list, gtv_list, faces_list, f_normals_list, f_adj_list, v_faces_list, mesh_count)
                                
            #                     print("mesh added ("+str(1000*(time.clock()-t0))+"ms)")
            #                     # Now recover vertices positions and create Edge maps
            #                     V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)


            #                     V0exp = np.expand_dims(V0, axis=0)

            #                     depth_diff = GT-V0
            #                     depth_dir = normalize(depth_diff)

            #                     # print("Inference ...")
            #                     t0 = time.clock()
            #                     #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
            #                     # upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)

            #                     upV0, upV0mid, upV0coarse, upN0, upN1, upN2, upP0, upP1, upP2 = inferNet(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
            #                     # upV0, upN0 = inferNetOld(V0exp, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)

            #                     # upV0, upN0 = inferNetOld(V0exp, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum, depth_dir)


            #                     print("Inference complete ("+str(1000*(time.clock()-t0))+"ms)")


            #                     # print("computing Hausdorff "+str(fileNum+1)+"...")
            #                     t0 = time.clock()
            #                     # haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
            #                     denseGT = getDensePC(GT, faces_gt, res=1)
                                
            #                     if gtFileName.startswith("aaarma") or gtFileName.startswith("Merlion"):
            #                         haus_dist0 = 0
            #                         avg_dist0 = 0
            #                     else:
            #                         haus_dist0, _, avg_dist0, _ = hausdorffOverSampled(upV0, GT, upV0, denseGT, accuracyOnly=True)
                                
            #                     # print("Hausdorff complete ("+str(1000*(time.clock()-t0))+"ms)")
            #                     # print("computing Angular diff "+str(fileNum+1)+"...")
            #                     t0 = time.clock()
            #                     angDistVec = angularDiffVec(upN0, GTf_normals)

            #                     borderF = getBorderFaces(faces_gt)

            #                     angDistIn = angDistVec[borderF==0]
            #                     angDistOut = angDistVec[borderF==1]

            #                     angDistIn0 = np.mean(angDistIn)
            #                     angStdIn0 = np.std(angDistIn)
            #                     angDistOut0 = np.mean(angDistOut)
            #                     angStdOut0 = np.std(angDistOut)
            #                     angDist0 = np.mean(angDistVec)
            #                     angStd0 = np.std(angDistVec)
            #                     # print("ang dist, std = (%f, %f)"%(angDist0, angStd0))

            #                     angDist0, angStd0 = angularDiff(upN0, GTf_normals)
            #                     # print("ang dist, std = (%f, %f)"%(angDist0, angStd0))
            #                     # print("Angular diff complete ("+str(1000*(time.clock()-t0))+"ms)")
            #                     # print("max angle: "+str(np.amax(angDistVec)))

            #                     angDict[noisyFile[:-4]] = angDistVec
            #                      # --- heatmap ---
            #                     angColor = angDistVec / empiricMax

            #                     angColor = 1 - angColor
            #                     angColor = np.maximum(angColor, np.zeros_like(angColor))

            #                     # print("getting colormap "+str(fileNum+1)+"...")
            #                     t0 = time.clock()
            #                     colormap = getHeatMapColor(1-angColor)
            #                     # print("colormap shape: "+str(colormap.shape))
            #                     newV, newF = getColoredMesh(upV0, faces_gt, colormap)
            #                     # print("colormap complete ("+str(1000*(time.clock()-t0))+"ms)")
            #                     #newV, newF = getHeatMapMesh(upV0, faces_gt, angColor)
            #                     # print("writing mesh...")
            #                     t0 = time.clock()
            #                     write_mesh(newV, newF, RESULTS_PATH+denoizedHeatmap)
            #                     print("mesh written ("+str(1000*(time.clock()-t0))+"ms)")
                                

            #                     write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)


            #                     finalNormals = computeFacesNormals(upV0, faces_gt)
            #                     f_normals0 = computeFacesNormals(V0, faces_gt)
            #                     angColor = (upN0+1)/2
            #                     angColorFinal = (finalNormals+1)/2
            #                     angColorNoisy = (f_normals0+1)/2
            #                     angColorGt = (GTf_normals+1)/2
                           
            #                     newV, newF = getColoredMesh(upV0, faces_gt, angColorFinal)
            #                     newVn, newFn = getColoredMesh(V0, faces_gt, angColor)
            #                     newVnoisy, newFnoisy = getColoredMesh(V0, faces_gt, angColorNoisy)
            #                     newVgt, newFgt = getColoredMesh(V0, faces_gt, angColorGt)

            #                     Vgt, Fgt = getColoredMesh(GT, faces_gt, angColorGt)

            #                     write_mesh(newV, newF, RESULTS_PATH+denoizedFileWColor)
            #                     write_mesh(newVn, newFn, RESULTS_PATH+noisyFileWColor)
            #                     write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFile)
            #                     write_mesh(newVgt, newFgt, RESULTS_PATH+noisyFileWGtN)
            #                     write_mesh(Vgt, Fgt, RESULTS_PATH+gtFileWGtN)
            #                     # angColor0 = (upN0+1)/2
            #                     # newV, newF = getColoredMesh(V0, faces_gt, angColor0)
            #                     # write_mesh(newV, newF, RESULTS_PATH+denoizedHeatmap)

            #                     # angColorRaw = (upN0Raw+1)/2
            #                     # newV, newF = getColoredMesh(V0, faces_gt, angColorRaw)
            #                     # write_mesh(newV, newF, RESULTS_PATH+denoizedFile[:-4]+"_nRaw.obj")

            #                     # Fill arrays
            #                     nameArray.append(denoizedFile)
            #                     resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum, angDistIn0, angStdIn0, angDistOut0, angStdOut0])

            #         if not nameArray:
            #             continue

            #         outputFile = open(csv_filename,'a')
            #         nameArray = np.array(nameArray)
            #         resultsArray = np.array(resultsArray,dtype=np.float32)

            #         tempArray = resultsArray.flatten()
            #         resStr = ["%.7f" % number for number in tempArray]
            #         resStr = np.reshape(resStr,resultsArray.shape)

            #         nameArray = np.expand_dims(nameArray, axis=-1)

            #         finalArray = np.concatenate((nameArray,resStr),axis=1)
            #         for row in range(finalArray.shape[0]):
            #             for col in range(finalArray.shape[1]):
            #                 outputFile.write(finalArray[row,col])
            #                 outputFile.write(' ')
            #             outputFile.write('\n')

            #         outputFile.close()
            #         scipy.io.savemat(RESULTS_PATH+"angDiffRaw.mat",mdict=angDict)

            # # Compute metrics and heatmaps on denoised meshes + GT
            # elif running_mode == 8:
            #     # Take the opportunity to generate array of metrics on reconstructions
            #     nameArray = []      # String array, to now which row is what
            #     resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
            #                         # [Max distance, Mean distance, Mean angle, std angle, face num]

            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"
            #     gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/test/original/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

            #     # results file name
            #     csv_filename = RESULTS_PATH+"results_heat.csv"
                
            #     angDict={}
            #     # Get GT mesh
            #     for gtFileName in os.listdir(gtFolder):

            #         nameArray = []
            #         resultsArray = []
            #         if (not gtFileName.endswith(".obj")):
            #             continue

            #         # denoizedFile0 = gtFileName[:-4]+"_n1-dtree3.obj"
            #         # denoizedFile1 = gtFileName[:-4]+"_n2-dtree3.obj"
            #         # denoizedFile2 = gtFileName[:-4]+"_n3-dtree3.obj"

            #         # denoizedFile0 = gtFileName[:-4]+"_noisy-dtree3.obj"
            #         denoizedFile0 = gtFileName[:-4]+"_denoized.obj"
            #         heatFile0 = gtFileName[:-4]+"_heatmap.obj"

            #         denoizedFile0 = gtFileName[:-4]+"_denoized_gray_1.obj"
            #         denoizedFile1 = gtFileName[:-4]+"_denoized_gray_2.obj"
            #         denoizedFile2 = gtFileName[:-4]+"_denoized_gray_3.obj"

            #         # denoizedFile0 = gtFileName[:-4]+"_n1_rescaled.obj"
            #         # denoizedFile1 = gtFileName[:-4]+"_n2_rescaled.obj"
            #         # denoizedFile2 = gtFileName[:-4]+"_n3_rescaled.obj"

            #         # denoizedFile0 = gtFileName[:-4]+"_n1_denoised_gray.obj"
            #         # denoizedFile1 = gtFileName[:-4]+"_n2_denoised_gray.obj"
            #         # denoizedFile2 = gtFileName[:-4]+"_n3_denoised_gray.obj"

            #         # heatFile0 = gtFileName[:-4]+"_dtree3_heatmap_1.obj"
            #         # heatFile1 = gtFileName[:-4]+"_dtree3_heatmap_2.obj"
            #         # heatFile2 = gtFileName[:-4]+"_dtree3_heatmap_3.obj"

            #         heatFile0 = gtFileName[:-4]+"_heatmap_1.obj"
            #         heatFile1 = gtFileName[:-4]+"_heatmap_2.obj"
            #         heatFile2 = gtFileName[:-4]+"_heatmap_3.obj"

            #         # if (os.path.isfile(RESULTS_PATH+heatFile0)) and (os.path.isfile(RESULTS_PATH+heatFile1)) and (os.path.isfile(RESULTS_PATH+heatFile2)):
            #         #     continue

            #         # Load GT mesh
            #         GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            #         GTf_normals = computeFacesNormals(GT, faces_gt)
            #         denseGT = getDensePC(GT, faces_gt, res=1)
            #         facesNum = faces_gt.shape[0]
            #         # We only need to load faces once. Connectivity doesn't change for noisy meshes
            #         # Same for adjacency matrix

            #         _, edge_map, v_e_map = getFacesAdj(faces_gt)
            #         f_adj = getFacesLargeAdj(faces_gt,K_faces)
            #         # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            #         # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
                    

            #         faces_gt = np.array(faces_gt).astype(np.int32)
            #         faces = np.expand_dims(faces_gt,axis=0)
            #         #faces = np.array(faces).astype(np.int32)
            #         #f_adj = np.expand_dims(f_adj, axis=0)
            #         #edge_map = np.expand_dims(edge_map, axis=0)
            #         v_e_map = np.expand_dims(v_e_map, axis=0)

            #         denoizedFilesList = [denoizedFile0]
            #         heatMapFilesList = [heatFile0]

            #         denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]
            #         heatMapFilesList = [heatFile0,heatFile1,heatFile2]

            #         for fileNum in range(len(denoizedFilesList)):
                        
            #             denoizedFile = denoizedFilesList[fileNum]
            #             heatFile = heatMapFilesList[fileNum]
                        
            #             if not os.path.isfile(RESULTS_PATH+heatFile):
                            
            #                 V0,_,_, _, _ = load_mesh(RESULTS_PATH, denoizedFile, 0, False)
            #                 f_normals0 = computeFacesNormals(V0, faces_gt)

            #                 print("computing Hausdorff "+ denoizedFile + " " + str(fileNum+1)+"...")
            #                 # haus_dist0, avg_dist0 = oneSidedHausdorff(V0, GT)

                            
            #                 haus_dist0, _, avg_dist0, _ = hausdorffOverSampled(V0, GT, V0, denseGT, accuracyOnly=True)

            #                 angDistVec = angularDiffVec(f_normals0, GTf_normals)

            #                 borderF = getBorderFaces(faces_gt)

            #                 angDistIn = angDistVec[borderF==0]
            #                 angDistOut = angDistVec[borderF==1]

            #                 angDistIn0 = np.mean(angDistIn)
            #                 angStdIn0 = np.std(angDistIn)
            #                 angDistOut0 = np.mean(angDistOut)
            #                 angStdOut0 = np.std(angDistOut)
            #                 angDist0 = np.mean(angDistVec)
            #                 angStd0 = np.std(angDistVec)
            #                 #print("ang dist, std = (%f, %f)"%(angDist0, angStd0))

            #                 angDist0, angStd0 = angularDiff(f_normals0, GTf_normals)
            #                 print("max angle: "+str(np.amax(angDistVec)))
            #                 angDict[denoizedFile[:-4]] = angDistVec
            #                 # --- Test heatmap ---
            #                 angColor = angDistVec / empiricMax
            #                 angColor = 1 - angColor
            #                 angColor = np.maximum(angColor, np.zeros_like(angColor))

            #                 colormap = getHeatMapColor(1-angColor)
            #                 newV, newF = getColoredMesh(V0, faces_gt, colormap)

            #                 # newV, newF = getHeatMapMesh(V0, faces_gt, angColor)

            #                 write_mesh(newV, newF, RESULTS_PATH+heatFile)
                            
            #                 # Fill arrays
            #                 nameArray.append(denoizedFile)
            #                 resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum, angDistIn0, angStdIn0, angDistOut0, angStdOut0])

            #         if not nameArray:
            #             continue
            #         outputFile = open(csv_filename,'a')
            #         nameArray = np.array(nameArray)
            #         resultsArray = np.array(resultsArray,dtype=np.float32)

            #         tempArray = resultsArray.flatten()
            #         resStr = ["%.7f" % number for number in tempArray]
            #         resStr = np.reshape(resStr,resultsArray.shape)

            #         nameArray = np.expand_dims(nameArray, axis=-1)

            #         finalArray = np.concatenate((nameArray,resStr),axis=1)
            #         for row in range(finalArray.shape[0]):
            #             for col in range(finalArray.shape[1]):
            #                 outputFile.write(finalArray[row,col])
            #                 outputFile.write(' ')
            #             outputFile.write('\n')

            #         outputFile.close()

            #         scipy.io.savemat(RESULTS_PATH+"angDiffFinal.mat",mdict=angDict)

            # # Color mesh by estimated normals
            # elif running_mode == 9:
                
            #     maxSize = 90000
            #     patchSize = 90000

            #     # Take the opportunity to generate array of metrics on reconstructions
            #     nameArray = []      # String array, to now which row is what
            #     resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
            #                         # [Max distance, Mean distance, Mean angle, std angle, face num]

            #     noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
            #     gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"

            #     noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
            #     gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

            #     # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
            #     # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

            #     noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/noisy/"
            #     gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"

            #     # results file name
            #     csv_filename = RESULTS_PATH+"results.csv"
                
            #     # Get GT mesh
            #     for gtFileName in os.listdir(gtFolder):

            #         nameArray = []
            #         resultsArray = []
            #         if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("aMerlion")) or (gtFileName.startswith("aarmadillo")) or (gtFileName.startswith("agargoyle")) or \
            #         (gtFileName.startswith("adragon")):
            #             continue

            #         mesh_count = [0]

            #         noisyFile0 = gtFileName[:-4]+"_noisy.obj"
            #         denoizedFile0 = gtFileName[:-4]+"_denoized.obj"
            #         noisyFileWColor0 = gtFileName[:-4]+"_nC.obj"

            #         # Get all 3 noisy meshes
            #         # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            #         # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            #         # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
                    
            #         # noisyFile0 = gtFileName[:-4]+"_n1.obj"
            #         # noisyFile1 = gtFileName[:-4]+"_n2.obj"
            #         # noisyFile2 = gtFileName[:-4]+"_n3.obj"

            #         # noisyFileWColor0 = gtFileName[:-4]+"_n1C.obj"
            #         # noisyFileWColor1 = gtFileName[:-4]+"_n2C.obj"
            #         # noisyFileWColor2 = gtFileName[:-4]+"_n3C.obj"

            #         # denoizedFile0 = gtFileName[:-4]+"_denoizedC_1.obj"
            #         # denoizedFile1 = gtFileName[:-4]+"_denoizedC_2.obj"
            #         # denoizedFile2 = gtFileName[:-4]+"_denoizedC_3.obj"


            #         # if (os.path.isfile(RESULTS_PATH+denoizedFile0)):
            #         if (os.path.isfile(RESULTS_PATH+denoizedFile0)) and (os.path.isfile(RESULTS_PATH+denoizedFile1)) and (os.path.isfile(RESULTS_PATH+denoizedFile2)):
            #             continue

            #         # Load GT mesh
            #         GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            #         GTf_normals = computeFacesNormals(GT, faces_gt)

            #         facesNum = faces_gt.shape[0]
            #         # We only need to load faces once. Connectivity doesn't change for noisy meshes
            #         # Same for adjacency matrix

            #         _, edge_map, v_e_map = getFacesAdj(faces_gt)
            #         f_adj = getFacesLargeAdj(faces_gt,K_faces)
                    

            #         faces_gt = np.array(faces_gt).astype(np.int32)
            #         faces = np.expand_dims(faces_gt,axis=0)
            #         #faces = np.array(faces).astype(np.int32)
            #         #f_adj = np.expand_dims(f_adj, axis=0)
            #         edge_map = np.expand_dims(edge_map, axis=0)
            #         v_e_map = np.expand_dims(v_e_map, axis=0)

                    
            #         noisyFilesList = [noisyFile0]
            #         noisyFilesWColorList = [noisyFileWColor0]
            #         denoizedFilesList = [denoizedFile0]

            #         # noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            #         # noisyFilesWColorList = [noisyFileWColor0,noisyFileWColor1,noisyFileWColor2]
            #         # denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            #         for fileNum in range(len(denoizedFilesList)):
                        
            #             denoizedFile = denoizedFilesList[fileNum]
            #             noisyFile = noisyFilesList[fileNum]
            #             noisyFileWColor = noisyFilesWColorList[fileNum]
                        
            #             if not os.path.isfile(RESULTS_PATH+denoizedFile):
                            

            #                 f_normals_list = []
            #                 GTfn_list = []
            #                 f_adj_list = []


            #                 faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, gtFolder, gtFileName, f_normals_list, GTfn_list, f_adj_list, mesh_count)

            #                 # Now recover vertices positions and create Edge maps
            #                 V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)
            #                 f_normals0 = computeFacesNormals(V0, faces_gt)
            #                 V0exp = np.expand_dims(V0, axis=0)

            #                 print("running ...")
            #                 #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                            

            #                 depth_diff = GT-V0
            #                 depth_dir = normalize(depth_diff)


            #                 # upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
            #                 upV0, upN0 = inferNetOld(V0exp, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum, depth_dir)
                            
                            
            #                 print("computing Hausdorff "+str(fileNum+1)+"...")
            #                 haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
            #                 angDistVec = angularDiffVec(upN0, GTf_normals)
            #                 angDist0, angStd0 = angularDiff(upN0, GTf_normals)
            #                 print("max angle: "+str(np.amax(angDistVec)))

            #                 finalNormals = computeFacesNormals(upV0, faces_gt)
            #                 angColor = (upN0+1)/2
            #                 angColorFinal = (finalNormals+1)/2
            #                 angColorNoisy = (f_normals0+1)/2
                       
            #                 newV, newF = getColoredMesh(upV0, faces_gt, angColorFinal)
            #                 newVn, newFn = getColoredMesh(V0, faces_gt, angColor)
            #                 newVnoisy, newFnoisy = getColoredMesh(V0, faces_gt, angColorNoisy)

            #                 write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
            #                 write_mesh(newVn, newFn, RESULTS_PATH+noisyFileWColor)
            #                 write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFile)

            #                 # Fill arrays
            #                 nameArray.append(denoizedFile)
            #                 resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum])

            #         outputFile = open(csv_filename,'a')
            #         nameArray = np.array(nameArray)
            #         resultsArray = np.array(resultsArray,dtype=np.float32)

            #         tempArray = resultsArray.flatten()
            #         resStr = ["%.7f" % number for number in tempArray]
            #         resStr = np.reshape(resStr,resultsArray.shape)

            #         nameArray = np.expand_dims(nameArray, axis=-1)

            #         finalArray = np.concatenate((nameArray,resStr),axis=1)
            #         for row in range(finalArray.shape[0]):
            #             for col in range(finalArray.shape[1]):
            #                 outputFile.write(finalArray[row,col])
            #                 outputFile.write(' ')
            #             outputFile.write('\n')

            #         outputFile.close()

            # # Test. Load and write pickled data
            # elif running_mode == 10:

            #     f_normals_list = []
            #     f_adj_list = []
            #     v_pos_list = []
            #     gtv_pos_list = []
            #     e_map_list = []
            #     v_emap_list = []

            #     valid_f_normals_list = []
            #     valid_f_adj_list = []
            #     valid_v_pos_list = []
            #     valid_gtv_pos_list = []
            #     valid_e_map_list = []
            #     valid_v_emap_list = []

            #     f_normals_list_temp = []
            #     f_adj_list_temp = []
            #     v_pos_list_temp = []
            #     gtv_pos_list_temp = []
            #     e_map_list_temp = []
            #     v_emap_list_temp = []

            #     inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train/"
            #     validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
            #     gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

            #     # Training
            #     pickleNum=4
            #     # while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
            #     with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
            #         f_normals_list_temp = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
            #         f_adj_list_temp = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'v_pos_list'+str(pickleNum), 'rb') as fp:
            #         v_pos_list_temp = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'rb') as fp:
            #         gtv_pos_list_temp = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'faces_list'+str(pickleNum), 'rb') as fp:
            #         faces_list_temp = pickle.load(fp, encoding='latin1')
            #     # with open(binDumpPath+'e_map_list'+str(pickleNum), 'rb') as fp:
            #     #     e_map_list_temp = pickle.load(fp, encoding='latin1')
            #     # with open(binDumpPath+'v_emap_list'+str(pickleNum), 'rb') as fp:
            #     #     v_emap_list_temp = pickle.load(fp, encoding='latin1')

            #     if pickleNum>=0:
            #         f_normals_list = f_normals_list_temp
            #         f_adj_list = f_adj_list_temp
            #         v_pos_list = v_pos_list_temp
            #         gtv_pos_list = gtv_pos_list_temp
            #         faces_list = faces_list_temp
            #         # e_map_list = e_map_list_temp
            #         # v_emap_list = v_emap_list_temp
            #     else:

            #         f_normals_list+=f_normals_list_temp
            #         f_adj_list+=f_adj_list_temp
            #         v_pos_list+=v_pos_list_temp
            #         gtv_pos_list+=gtv_pos_list_temp
            #         faces_list += faces_list_temp
            #         # e_map_list+=e_map_list_temp
            #         # v_emap_list+=v_emap_list_temp


            #     print("loaded training pickle "+str(pickleNum))
            #     pickleNum+=1


            #     # Validation
            #     with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
            #         valid_f_normals_list = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
            #         valid_f_adj_list = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'valid_v_pos_list', 'rb') as fp:
            #         valid_v_pos_list = pickle.load(fp, encoding='latin1')
            #     with open(binDumpPath+'valid_gtv_pos_list', 'rb') as fp:
            #         valid_gtv_pos_list = pickle.load(fp, encoding='latin1')
            #     # with open(binDumpPath+'valid_e_map_list', 'rb') as fp:
            #     #     valid_e_map_list = pickle.load(fp, encoding='latin1')
            #     # with open(binDumpPath+'valid_v_emap_list', 'rb') as fp:
            #     #     valid_v_emap_list = pickle.load(fp, encoding='latin1')
            #     testWriteFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Test/"
            #     for samp in range(len(v_pos_list)):
            #         V0 = np.squeeze(v_pos_list[samp])
            #         GT0 = np.squeeze(gtv_pos_list[samp])
            #         faces0 = np.squeeze(faces_list[samp])
                    
            #         write_xyz(V0, testWriteFolder+'points_'+str(samp)+'.xyz')
            #         write_xyz(GT0, testWriteFolder+'GT0_'+str(samp)+'.xyz')

            #         write_mesh(V0,faces0,testWriteFolder+'mesh_'+str(samp)+'.obj')
            #         # haus0 = oneSidedHausdorff(V0,GT0)
            #         # print("Haus0: "+str(haus0))
            #         # validV0 = np.squeeze(valid_v_pos_list[samp])
            #         # validGT0 = np.squeeze(valid_gtv_pos_list[samp])
            #         # write_xyz(validV0, testWriteFolder+'v_points_'+str(samp)+'.xyz')
            #         # write_xyz(validGT0, testWriteFolder+'v_GT_'+str(samp)+'.xyz')

            #     # vhaus0 = oneSidedHausdorff(validV0,validGT0)
            #     # print("Haus0: "+str(vhaus0))

            # # Load and pickle training data.
            # elif running_mode == 11:

            #     f_normals_list_temp = []
            #     f_adj_list_temp = []
            #     v_pos_list_temp = []
            #     gtv_pos_list_temp = []
            #     e_map_list_temp = []
            #     v_emap_list_temp = []

            #     valid_f_normals_list = []
            #     valid_f_adj_list = []
            #     valid_v_pos_list = []
            #     valid_gtv_pos_list = []
            #     valid_e_map_list = []
            #     valid_v_emap_list = []

            #     inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train2/"
            #     validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
            #     gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

            #     pickleNum=0
            #     # Training set
            #     for filename in os.listdir(inputFilePath):
            #         #print("training_meshes_num start_iter " + str(training_meshes_num))
            #         if training_meshes_num[0]>1000:
            #             break
            #         #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
            #         #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
            #         if (filename.endswith(".obj")):

            #             print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")

            #             # For DTU
            #             fileNumStr = filename[4:7]
            #             gtfilename = 'stl'+fileNumStr+'_total.obj'

            #             addMeshWithVertices(inputFilePath, filename, gtFilePath, gtfilename, v_pos_list_temp, gtv_pos_list_temp, f_normals_list_temp, f_adj_list_temp, e_map_list_temp, v_emap_list_temp, training_meshes_num)

            #             # Save batches of meshes/patches (for training only)
            #             if training_meshes_num[0]>30:
            #                 # Training
            #                 with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
            #                     pickle.dump(f_normals_list_temp, fp)
            #                 with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
            #                     pickle.dump(f_adj_list_temp, fp)
            #                 with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
            #                     pickle.dump(v_pos_list_temp, fp)
            #                 with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
            #                     pickle.dump(gtv_pos_list_temp, fp)
            #                 with open(binDumpPath+'e_map_list'+str(pickleNum), 'wb') as fp:
            #                     pickle.dump(e_map_list_temp, fp)
            #                 with open(binDumpPath+'v_emap_list'+str(pickleNum), 'wb') as fp:
            #                     pickle.dump(v_emap_list_temp, fp)

            #                 pickleNum+=1
            #                 f_normals_list_temp = []
            #                 f_adj_list_temp = []
            #                 v_pos_list_temp = []
            #                 gtv_pos_list_temp = []
            #                 e_map_list_temp = []
            #                 v_emap_list_temp = []
            #                 training_meshes_num[0] = 0

            #     if training_meshes_num[0]>0:
            #         # Training
            #         with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
            #             pickle.dump(f_normals_list_temp, fp)
            #         with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
            #             pickle.dump(f_adj_list_temp, fp)
            #         with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
            #             pickle.dump(v_pos_list_temp, fp)
            #         with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
            #             pickle.dump(gtv_pos_list_temp, fp)
            #         with open(binDumpPath+'e_map_list'+str(pickleNum), 'wb') as fp:
            #             pickle.dump(e_map_list_temp, fp)
            #         with open(binDumpPath+'v_emap_list'+str(pickleNum), 'wb') as fp:
            #             pickle.dump(v_emap_list_temp, fp)

            #     # # Validation set
            #     # for filename in os.listdir(validFilePath):
            #     #     if (filename.endswith(".obj")):

            #     #         # For DTU
            #     #         fileNumStr = filename[4:7]
            #     #         gtfilename = 'stl'+fileNumStr+'_total.obj'

            #     #         addMeshWithVertices(validFilePath, filename, gtFilePath, gtfilename, valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list, valid_meshes_num)
                        
            #     # # Validation
            #     # with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
            #     #     pickle.dump(valid_f_normals_list, fp)
            #     # with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
            #     #     pickle.dump(valid_f_adj_list, fp)
            #     # with open(binDumpPath+'valid_v_pos_list', 'wb') as fp:
            #     #     pickle.dump(valid_v_pos_list, fp)
            #     # with open(binDumpPath+'valid_gtv_pos_list', 'wb') as fp:
            #     #     pickle.dump(valid_gtv_pos_list, fp)
            #     # with open(binDumpPath+'valid_e_map_list', 'wb') as fp:
            #     #     pickle.dump(valid_e_map_list, fp)
            #     # with open(binDumpPath+'valid_v_emap_list', 'wb') as fp:
            #     #     pickle.dump(valid_v_emap_list, fp)


            # # Train network
            # if running_mode == 16:

                # gtnameoffset = 10
                # f_normals_list = []
                # f_adj_list = []
                # GTfn_list = []
                # GTfdisp_list = []

                # f_normals_list_temp = []
                # f_adj_list_temp = []
                # GTfn_list_temp = []
                # GTfdisp_list_temp = []

                # valid_f_normals_list = []
                # valid_f_adj_list = []
                # valid_GTfn_list = []
                # valid_GTfdisp_list = []

                # toDelFaces_list = []

                # # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
                # # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
                # # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

                # # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/noisy/"
                # # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
                # # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"

                # # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/noisy/"
                # # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/valid/"
                # # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/original/"


                # # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/noisy/"
                # # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/valid/"
                # # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/original/"
                
                # # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
                # # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid/"
                # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
                # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid_cleaned/"
                # # gtFilePath = "/morpheo-nas2/marmando/MPI-FAUST/training/rescaled/"
                # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Ground_Truth_Meshes/"

                # #print("training_meshes_num 0 " + str(training_meshes_num))
                # if pickleLoad:

                #     # Training
                #     pickleNum=0
                #     while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
                #         with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
                #             f_normals_list_temp = pickle.load(fp, encoding='latin1')
                #         with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
                #             f_adj_list_temp = pickle.load(fp, encoding='latin1')
                #         with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'rb') as fp:
                #             GTfdisp_list_temp = pickle.load(fp, encoding='latin1')
                #         with open(binDumpPath+'GTfn_list'+str(pickleNum), 'rb') as fp:
                #             GTfn_list_temp = pickle.load(fp, encoding='latin1')

                #         if pickleNum==0:
                #             f_normals_list = f_normals_list_temp
                #             f_adj_list = f_adj_list_temp
                #             GTfn_list = GTfn_list_temp
                #             GTfdisp_list = GTfdisp_list_temp
                #         else:
                #             f_normals_list+=f_normals_list_temp
                #             f_adj_list+=f_adj_list_temp
                #             GTfn_list+=GTfn_list_temp
                #             GTfdisp_list+=GTfdisp_list_temp

                #         print("loaded training pickle "+str(pickleNum))
                #         pickleNum+=1

                #     # Validation
                #     with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
                #         valid_f_normals_list = pickle.load(fp)
                #     with open(binDumpPath+'valid_GTfn_list', 'rb') as fp:
                #         valid_GTfn_list = pickle.load(fp)
                #     with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
                #         valid_f_adj_list = pickle.load(fp)
                #     with open(binDumpPath+'valid_GTfdisp_list', 'rb') as fp:
                #         valid_GTfdisp_list = pickle.load(fp)


                # else:
                #     pickleNum=400
                #     # Training set
                #     for filename in os.listdir(inputFilePath):
                #         #print("training_meshes_num start_iter " + str(training_meshes_num))
                #         if training_meshes_num[0]>300:
                #             break
                #         #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
                #         #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
                #         if (filename.endswith(".obj")):


                #             #For FAUST
                #             fileNumStr = filename[5:8]
                #             fileNum = int(fileNumStr)

                #             if fileNum<100:
                #                 continue

                #             gtfilename = 'gt'+fileNumStr+'.obj'
                #             # if int(fileNumStr)>2:
                #             #     continue
                #             print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")
                #             # gtfilename = filename[:-gtnameoffset]+".obj"
                #             addMeshWGTDispAndNormals(inputFilePath, filename, gtFilePath, gtfilename, f_normals_list_temp, GTfn_list_temp, f_adj_list_temp, GTfdisp_list_temp, toDelFaces_list, training_meshes_num)


                #         if training_meshes_num[0]>4:
                #             if pickleSave:
                #                 # Training
                #                 with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                #                     pickle.dump(f_normals_list_temp, fp)
                #                 with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                #                     pickle.dump(f_adj_list_temp, fp)
                #                 with open(binDumpPath+'GTfn_list'+str(pickleNum), 'wb') as fp:
                #                     pickle.dump(GTfn_list_temp, fp)
                #                 with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'wb') as fp:
                #                     pickle.dump(GTfdisp_list_temp, fp)
                #                 with open(binDumpPath+'toDelFaces_list'+str(pickleNum), 'wb') as fp:
                #                     pickle.dump(toDelFaces_list, fp)

                #             if pickleNum==0:
                #                 f_normals_list = f_normals_list_temp
                #                 f_adj_list = f_adj_list_temp
                #                 GTfn_list = GTfn_list_temp
                #                 GTfdisp_list = GTfdisp_list_temp
                #             else:
                #                 f_normals_list+=f_normals_list_temp
                #                 f_adj_list+=f_adj_list_temp
                #                 GTfn_list+=GTfn_list_temp
                #                 GTfdisp_list+=GTfdisp_list_temp

                #             pickleNum+=1
                #             f_normals_list_temp = []
                #             f_adj_list_temp = []
                #             GTfn_list_temp = []
                #             GTfdisp_list_temp = []
                #             training_meshes_num[0] = 0

                #     if (pickleSave) and training_meshes_num[0]>0:
                #         if pickleSave:
                #             # Training
                #             with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                #                 pickle.dump(f_normals_list_temp, fp)
                #             with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                #                 pickle.dump(f_adj_list_temp, fp)
                #             with open(binDumpPath+'GTfn_list'+str(pickleNum), 'wb') as fp:
                #                 pickle.dump(GTfn_list_temp, fp)
                #             with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'wb') as fp:
                #                 pickle.dump(GTfdisp_list_temp, fp)

                #         if pickleNum==0:
                #             f_normals_list = f_normals_list_temp
                #             f_adj_list = f_adj_list_temp
                #             GTfn_list = GTfn_list_temp
                #             GTfdisp_list = GTfdisp_list_temp
                #         else:
                #             f_normals_list+=f_normals_list_temp
                #             f_adj_list+=f_adj_list_temp
                #             GTfn_list+=GTfn_list_temp
                #             GTfdisp_list+=GTfdisp_list_temp

                #         pickleNum+=1
                #         f_normals_list_temp = []
                #         f_adj_list_temp = []
                #         GTfn_list_temp = []
                #         GTfdisp_list_temp = []
                #         training_meshes_num[0] = 0

                    
                #     # Validation set
                #     for filename in os.listdir(validFilePath):
                #         if (filename.endswith(".obj")):
                #             # gtfilename = filename[:-gtnameoffset]+".obj"
                #             #For FAUST
                #             fileNumStr = filename[5:8]
                #             gtfilename = 'gt'+fileNumStr+'.obj'
                #             addMeshWGTDispAndNormals(validFilePath, filename, gtFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_GTfdisp_list, [], valid_meshes_num)

                #     if pickleSave:
                #         # Validation
                #         with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
                #             pickle.dump(valid_f_normals_list, fp)
                #         with open(binDumpPath+'valid_GTfn_list', 'wb') as fp:
                #             pickle.dump(valid_GTfn_list, fp)
                #         with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
                #             pickle.dump(valid_f_adj_list, fp)
                #         with open(binDumpPath+'valid_GTfdisp_list', 'wb') as fp:
                #             pickle.dump(valid_GTfdisp_list, fp)
                
                # train6DNetWGT(f_normals_list, GTfn_list, f_adj_list, GTfdisp_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_GTfdisp_list)


    # Inference
    if running_mode == 1:
        in_list = []
        adj_list = []
        GT_list = []

        maxSize = 50000
        patchSize = maxSize
        patchIndices_list, nodesNum = addMesh(dataFolder, dataBaseName, in_list, GT_list, adj_list, training_meshes_num)
        # inPos = GT_list[0][:,:,:3]

        inPos_list=[]

        patchNum = len(GT_list)

        for i in range(patchNum):
            inPos_list.append(GT_list[i][:,:,:3])
            in_list[i] = 2*in_list[i]-1

        outColor_list = inferNet(in_list,inPos_list,adj_list)

        outColor = np.zeros([nodesNum,3],dtype=np.float32)
        fullPos = np.zeros([nodesNum,3],dtype=np.float32)
        for i in range(patchNum):

            print("patchInd shape = ",patchIndices_list[i].shape)
            print("outColor update shape = ",outColor_list[i][0,:,:].shape)
            print("outColor shape = ",outColor.shape)
            print("outColor[patchInd] shape = ",outColor[patchIndices_list[i]].shape)
            maxUpdate = np.amax(outColor_list[i][0,:,:])
            minUpdate = np.amin(outColor_list[i][0,:,:])
            print("maxUpdate = ",maxUpdate)
            print("minUpdate = ",minUpdate)
            outColor[patchIndices_list[i]] = outColor_list[i][0,:,:]
            fullPos[patchIndices_list[i]] = inPos_list[i][0,:,:]


        outColor = (outColor+1.0)/2.0
        print("outColor shape = ",outColor.shape)
        print("outColor type = ",outColor.dtype)
        # print("inPos shape = ",inPos.shape)
        # print("inPos type = ",inPos.dtype)
        outColor=255*outColor
        outColor = np.maximum.reduce([outColor,np.zeros_like(outColor)])
        array255 = np.zeros_like(outColor)+255.0
        outColor = np.minimum.reduce([outColor,array255])

        print("fullPos shape = ",fullPos.shape)
        # fullVec = np.concatenate((inPos[0],outColor[0]),axis=-1)
        fullVec = np.concatenate((fullPos,outColor),axis=-1)
        print("fullVec slice: ",fullVec[:3,:])
        write_coff(fullVec,RESULTS_PATH+"test.off")

        
    
    print("Complete: mode = "+str(RUNNING_MODE)+", architecture "+str(ARCHITECTURE)+", net path = "+NETWORK_PATH)
    #


def pickleMesh(inputFolder,meshName,pickleFolder):

    # adjMatFormat = "%s_adjMat.txt"%meshName
    # gtMatFormat = "%s_GTMat.txt"%meshName
    # inputFormat = "%s_inMat.txt"%meshName

    # pickleAdjMatFormat = "%s_adjMat.pkl"%meshName
    # pickleGtMatFormat = "%s_GTMat.pkl"%meshName
    # pickleInputFormat = "%s_inMat.pkl"%meshName

    adjMatName = adjMatFormat%meshName
    gtMatName = gtMatFormat%meshName
    inputName = inputFormat%meshName

    pickleAdjMatName = pickleAdjMatFormat%meshName
    pickleGtMatName = pickleGtMatFormat%meshName
    pickleInputName = pickleInputFormat%meshName
    # Load, pickle and delete each matrix

    # GT Matrix
    gtMatPos, gtMatColor = load_coff_PC(inputFolder+"/"+gtMatName)
    nbNodes = gtMatPos.shape[0]
    gtMat = np.concatenate([gtMatPos,gtMatColor],axis=-1)
    with open(pickleFolder+"/"+pickleGtMatName, 'wb') as fp:
                pickle.dump(gtMat, fp, protocol=2)
    os.remove(inputFolder+"/"+gtMatName)

    # Adjacency Matrix
    adjMat = load_text_adjMat(inputFolder+"/"+adjMatName)
    adjMat = adjMat[:nbNodes]
    with open(pickleFolder+"/"+pickleAdjMatName, 'wb') as fp:
                pickle.dump(adjMat, fp, protocol=2)
    os.remove(inputFolder+"/"+adjMatName)

    # input Matrix
    inputMat = load_text_adjMat(inputFolder+"/"+inputName)
    inputMat = inputMat[:nbNodes]
    with open(pickleFolder+"/"+pickleInputName, 'wb') as fp:
                pickle.dump(inputMat, fp, protocol=2)
    os.remove(inputFolder+"/"+inputName)

    return



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=int, default=0)
    #parser.add_argument('--dataset_path')
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--network_path', type=str)
    # parser.add_argument('--num_iterations', type=int, default=30000)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--net_name', type=str, default='net')
    parser.add_argument('--running_mode', type=int, default=0)
    parser.add_argument('--coarsening_steps', type=int, default=2)
    parser.add_argument('--sequence_name', type=str, default='')
    parser.add_argument('--N2', type=bool, default=False)

    FLAGS = parser.parse_args()

    ARCHITECTURE = FLAGS.architecture
    #DATASET_PATH = FLAGS.dataset_path
    RESULTS_PATH = FLAGS.results_path
    NETWORK_PATH = FLAGS.network_path
    # NUM_ITERATIONS = FLAGS.num_iterations
    NUM_EPOCHS = FLAGS.num_epochs
    DEVICE = FLAGS.device
    NET_NAME = FLAGS.net_name
    RUNNING_MODE = FLAGS.running_mode
    COARSENING_STEPS = FLAGS.coarsening_steps
    SEQUENCE_NAME = FLAGS.sequence_name
    BOOL_N2 = FLAGS.N2

    adjMatFormat = "%s_adjMat.txt"
    gtMatFormat = "%s_GTMat.txt"
    inputFormat = "%s_inMat.txt"

    pickleAdjMatFormat = "%s_adjMat.pkl"
    pickleGtMatFormat = "%s_GTMat.pkl"
    pickleInputFormat = "%s_inMat.pkl"


    if SEQUENCE_NAME=="knight":


        knightFolder = "/morpheo-nas2/marmando/Appearance/ASR/early_tests/allCam/res8/knight_419/"
        dataFolder = knightFolder
        dataBaseName = "knight419"
        resultsPathFormat = "/morpheo-nas2/marmando/Appearance/ASR/results/allCam/archi%d/"
        networkPathFormat = "/morpheo-nas2/marmando/Appearance/ASR/networks/allCam/archi%d/"


    if RESULTS_PATH is None:
        RESULTS_PATH = resultsPathFormat%ARCHITECTURE
    if NETWORK_PATH is None:
        NETWORK_PATH = networkPathFormat%ARCHITECTURE



    # pickleMesh(knightFolder,"knight419",knightFolder)

    # # # verts_test = load_off_PC("/morpheo-nas2/marmando/Appearance/ASR/early_tests/samplesPos.txt")
    # verts_test, colors_test = load_coff_PC("/morpheo-nas2/marmando/Appearance/ASR/early_tests/samplesPosAndColor.txt")

    # # filename=dataBaseName
    # # with open(knightFolder+"/"+pickleGtMatFormat%filename, 'rb') as fp:
    # #     gtMat = pickle.load(fp)


    # # verts_test = gtMat[:,:3]
    # # colors_test = gtMat[:,3:]


    # # print("verts_test shape = ",verts_test.shape)
    # # print("verts_test extract = ",verts_test[:10,:])

    # # # adjMat = load_text_adjMat("/morpheo-nas2/marmando/Appearance/ASR/early_tests/adjMat.txt")
    # nbVerts = verts_test.shape[0]
    # # # adjMat = adjMat[:nbVerts]
    # # # print("adjMat shape = ",adjMat.shape)

    # # with open(knightFolder+"/"+pickleAdjMatFormat%filename, 'rb') as fp:
    # #     adjMat = pickle.load(fp)
    # # adjMat = adjMat+1


    # colorsInput = load_text_adjMat("/morpheo-nas2/marmando/Appearance/ASR/early_tests/colorsTotal.txt")
    # colorsInput = colorsInput[:nbVerts]
    # print("max colorsInput = ",np.amax(colorsInput))
    # print("min colorsInput = ",np.amin(colorsInput))
    # print("type colorsInput = ",colorsInput.dtype)

    # # with open(knightFolder+"/"+pickleInputFormat%filename, 'rb') as fp:
    # #     colorsInput = pickle.load(fp)

    # colorsInput = colorsInput.astype(float)
    # # colorsInput=colorsInput/255.0
    # # # vertsNormals = np.array([[1.0,0.0,0.0]],np.float32)
    # # # vertsNormals=np.tile(vertsNormals,[nbVerts,1])
    # # vertsNormals=colors_test

    # for cam in range(int(colorsInput.shape[1]/3)):
    #     fullVec = np.concatenate((verts_test,colorsInput[:,3*cam:3*cam+3]),axis=-1)
    #     write_coff(fullVec, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/allCam/"+"PC_cam_"+str(cam)+".off")
    #     print("written cam ",cam)

    # faceMeshV, faceMeshF = makeFacesMesh(adjMat,verts_test,vertsNormals)
    # write_mesh(faceMeshV, faceMeshF, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/facemesh.obj")

    # faceMeshV, faceMeshF = makeFacesMesh(adjMat,verts_test,colorsInput[:,0:3])
    # write_mesh(faceMeshV, faceMeshF, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/facemesh_in0.obj")
    # faceMeshV, faceMeshF = makeFacesMesh(adjMat,verts_test,colorsInput[:,3:6])
    # write_mesh(faceMeshV, faceMeshF, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/facemesh_in1.obj")
    # faceMeshV, faceMeshF = makeFacesMesh(adjMat,verts_test,colorsInput[:,6:9])
    # write_mesh(faceMeshV, faceMeshF, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/facemesh_in2.obj")
    # faceMeshV, faceMeshF = makeFacesMesh(adjMat,verts_test,colorsInput[:,9:12])
    # write_mesh(faceMeshV, faceMeshF, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/facemesh_in3.obj")
    # faceMeshV, faceMeshF = makeFacesMesh(adjMat,verts_test,colorsInput[:,12:])
    # write_mesh(faceMeshV, faceMeshF, "/morpheo-nas2/marmando/Appearance/ASR/early_tests/facemesh_in4.obj")



    mainFunction()


