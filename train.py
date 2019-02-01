from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
#import h5py
import argparse
import os
import pickle
from model import *
from utils import *
from tensorflow.python import debug as tf_debug
import random
from lib.coarsening import *



def inferNetOld(in_points, f_normals, f_adj, edge_map, v_e_map,num_wofake_nodes,patch_indices,old_to_new_permutations,num_faces):

    with tf.Graph().as_default():
        random_seed = 0
        np.random.seed(random_seed)

        sess = tf.InteractiveSession()
        if(FLAGS.debug):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)

        """
        Load dataset
        x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
        adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                                  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
        """

        BATCH_SIZE=f_normals[0].shape[0]
        NUM_POINTS=in_points.shape[1]
        MAX_EDGES = v_e_map.shape[2]
        NUM_EDGES = edge_map.shape[1]
        K_faces = f_adj[0][0].shape[2]
        NUM_IN_CHANNELS = f_normals[0].shape[2]

        xp_ = tf.placeholder('float32', shape=(BATCH_SIZE, NUM_POINTS,3),name='xp_')

        fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')

        fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
        fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
        fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

        e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_EDGES,4], name='e_map_')
        ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_POINTS,MAX_EDGES], name='ve_map_')
        keep_prob = tf.placeholder(tf.float32)
        
        fadjs = [fadj0,fadj1,fadj2]
        
        # --- Starting iterative process ---
        #rotTens = getRotationToAxis(fn_)
        with tf.variable_scope("model"):
            n_conv, _, _ = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)

        n_conv = normalizeTensor(n_conv)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("ERROR! Neural network not found! Aborting mission.")
            return

        # points shape should now be [NUM_POINTS, 3]
        predicted_normals = np.zeros([num_faces,3])
        for i in range(len(f_normals)):
            print("Patch "+str(i+1)+" / "+str(len(f_normals)))
            my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2],
                            keep_prob:1.0}
            outN = sess.run(tf.squeeze(n_conv),feed_dict=my_feed_dict)
            #outN = f_normals[i][0]

            # Permute back patch
            temp_perm = np.array(inv_perm(old_to_new_permutations[i]))
            outN = outN[temp_perm]
            outN = outN[0:num_wofake_nodes[i]]
            # remove fake nodes from prediction
            if len(patch_indices[i]) == 0:
                predicted_normals = outN
            else:
                for count in range(len(patch_indices[i])):
                    predicted_normals[patch_indices[i][count]] = outN[count]
        #Update vertices position
        new_normals = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='fn_')
        #refined_x = update_position(xp_,fadj, n_conv)
        refined_x = update_position2(xp_, new_normals, e_map_, ve_map_, iter_num=60)
        points = tf.squeeze(refined_x)

        update_feed_dict = {xp_:in_points, new_normals: [predicted_normals], e_map_: edge_map, ve_map_: v_e_map}
        outPoints = sess.run(points,feed_dict=update_feed_dict)
        sess.close()

        return outPoints, predicted_normals


def inferNet(in_points, faces, f_normals, f_adj, v_faces, new_to_old_v_list, new_to_old_f_list, num_points, num_faces, adjPerm_list, real_nodes_num_list):

    with tf.Graph().as_default():
        random_seed = 0
        np.random.seed(random_seed)

        sess = tf.InteractiveSession()
        if(FLAGS.debug):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)

        """
        Load dataset
        x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
        adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                                  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
        """

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
            n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
            # n_conv0 = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
            # n_conv1 = n_conv0
            # n_conv2 = n_conv0
        n_conv0 = normalizeTensor(n_conv0)
        n_conv1 = normalizeTensor(n_conv1)
        n_conv2 = normalizeTensor(n_conv2)
        n_conv_list = [n_conv0, n_conv1, n_conv2]

        # refined_x = update_position_MS(xp_, new_normals, faces_, v_faces_, coarsening_steps=3)


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

        upN1 = custom_upsampling(new_normals1,COARSENING_STEPS)
        upN2 = custom_upsampling(new_normals2,COARSENING_STEPS*2)
        new_normals = [new_normals0, new_normals1, new_normals2]
        
        # refined_x, dx_list = update_position_disp(xp_, new_normals, faces_, v_faces_, coarsening_steps=COARSENING_STEPS)
        refined_x, dx_list = update_position_MS(xp_, new_normals, faces_, v_faces_, coarsening_steps=COARSENING_STEPS)

        refined_x = refined_x #+ dx_list[1] #+ dx_list[2]

        finalOutPoints = np.zeros((num_points,3),dtype=np.float32)
        pointsWeights = np.zeros((num_points,3),dtype=np.float32)

        finalFineNormals = np.zeros((num_faces,3),dtype=np.float32)
        finalMidNormals = np.zeros((num_faces,3),dtype=np.float32)
        finalCoarseNormals = np.zeros((num_faces,3),dtype=np.float32)
        for i in range(len(f_normals)):
            print("Patch "+str(i+1)+" / "+str(len(f_normals)))
            my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2], 
                            keep_prob:1.0}
            # outN0, outN1, outN2 = sess.run([tf.squeeze(n_conv0), tf.squeeze(n_conv1), tf.squeeze(n_conv2)],feed_dict=my_feed_dict)
            print("Running normals...")
            outN0, outN1, outN2 = sess.run([n_conv0, n_conv1, n_conv2],feed_dict=my_feed_dict)
            print("Normals: check")
            # outN = f_normals[i][0]

            fnum0 = f_adj[i][0].shape[1]
            fnum1 = f_adj[i][1].shape[1]
            fnum2 = f_adj[i][2].shape[1]

            # outN0 = np.slice(f_normals[0],[0,0,0],[-1,-1,3])
            # outN0 = f_normals[0][:,:,:3]
            # outN0 = np.tile(np.array([[[0,0,1]]]),[1,f_normals[0].shape[1],1])

            points = tf.reshape(refined_x,[-1,3])
            # points = tf.squeeze(refined_x)
            

            update_feed_dict = {xp_:in_points[i], new_normals0: outN0, new_normals1: outN1, new_normals2: outN2,
                                faces_: faces[i], v_faces_: v_faces[i]}
            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}
            # testNorm = f_normals[i][:,:,:3]/100
            # update_feed_dict = {xp_:in_points[i], new_normals0: testNorm, new_normals1: outN1, new_normals2: outN2,
            #                     faces_: faces[i], v_faces_: v_faces[i]}
            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}

            print("Running points...")

            normalised_disp_fine = new_normals0
            # normalised_disp_fine = normalizeTensor(new_normals0)
            normalised_disp_mid = normalizeTensor(upN1)
            normalised_disp_coarse = normalizeTensor(upN2)

            # outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, new_normals0, upN1, upN2],feed_dict=update_feed_dict)
            outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse],feed_dict=update_feed_dict)

            print("Points: check")
            print("Updating mesh...")
            if len(f_normals)>1:
                finalOutPoints[new_to_old_v_list[i]] += outPoints
                pointsWeights[new_to_old_v_list[i]]+=1

                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
                midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
                coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
                coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]

                finalFineNormals[new_to_old_f_list[i]] = fineNormalsP
                finalMidNormals[new_to_old_f_list[i]] = midNormalsP
                finalCoarseNormals[new_to_old_f_list[i]] = coarseNormalsP
            else:
                finalOutPoints = outPoints
                pointsWeights +=1

                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
                midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
                coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
                coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]
                
                finalFineNormals = fineNormalsP
                finalMidNormals = midNormalsP
                finalCoarseNormals = coarseNormalsP
            print("Mesh update: check")
        sess.close()

        finalOutPoints = np.true_divide(finalOutPoints,pointsWeights)

        return finalOutPoints, finalFineNormals, finalMidNormals, finalCoarseNormals


def inferNet6D(in_points, faces, f_normals, f_adj, v_faces, new_to_old_v_list, new_to_old_f_list, num_points, num_faces, adjPerm_list, real_nodes_num_list):

    with tf.Graph().as_default():
        random_seed = 0
        np.random.seed(random_seed)

        sess = tf.InteractiveSession()
        if(FLAGS.debug):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)

        """
        Load dataset
        x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
        adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                                  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
        """

        BATCH_SIZE=f_normals[0].shape[0]
        K_faces = f_adj[0][0].shape[2]
        K_vertices = v_faces[0].shape[2]
        NUM_IN_CHANNELS = f_normals[0].shape[2]
        print("f_normals shape = "+str(f_normals[0].shape))
        print("f_normals type = "+str(f_normals[0].dtype))

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
            n_conv0, fDisp = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
            # n_conv0 = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
            # n_conv1 = n_conv0
            # n_conv2 = n_conv0
        n_conv0 = normalizeTensor(n_conv0)
        

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
        
        # new_normals1 = custom_binary_tree_pooling(new_normals0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        # new_normals1 = normalizeTensor(new_normals1)
        # new_normals2 = custom_binary_tree_pooling(new_normals1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
        # new_normals2 = normalizeTensor(new_normals2)


        
        refined_x, dx_list = update_position_disp(xp_, [fDisp], faces_, v_faces_, coarsening_steps=COARSENING_STEPS)
        refined_x2, dx_list = update_position_MS(refined_x, [n_conv0], faces_, v_faces_, coarsening_steps=COARSENING_STEPS)

        # refined_x = refined_x #+ dx_list[1] #+ dx_list[2]

        finalOutPoints = np.zeros((num_points,3),dtype=np.float32)
        pointsWeights = np.zeros((num_points,3),dtype=np.float32)

        finalFineNormals = np.zeros((num_faces,3),dtype=np.float32)
        for i in range(len(f_normals)):
            print("Patch "+str(i+1)+" / "+str(len(f_normals)))
            my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2], 
                            keep_prob:1.0}
            # outN0, outN1, outN2 = sess.run([tf.squeeze(n_conv0), tf.squeeze(n_conv1), tf.squeeze(n_conv2)],feed_dict=my_feed_dict)
            # print("Running normals...")
            # outN0 = sess.run(n_conv0,feed_dict=my_feed_dict)
            # print("Normals: check")
            # outN = f_normals[i][0]

            fnum0 = f_adj[i][0].shape[1]
            fnum1 = f_adj[i][1].shape[1]
            fnum2 = f_adj[i][2].shape[1]

            # outN0 = np.slice(f_normals[0],[0,0,0],[-1,-1,3])
            # outN0 = f_normals[0][:,:,:3]
            # outN0 = np.tile(np.array([[[0,0,1]]]),[1,f_normals[0].shape[1],1])

            points = tf.reshape(refined_x2,[-1,3])
            # points = tf.squeeze(refined_x)
            

            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}
            update_feed_dict = {xp_:in_points[i], fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2], 
                                keep_prob:1.0,
                                faces_: faces[i], v_faces_: v_faces[i]}
            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}
            # testNorm = f_normals[i][:,:,:3]/100
            # update_feed_dict = {xp_:in_points[i], new_normals0: testNorm, new_normals1: outN1, new_normals2: outN2,
            #                     faces_: faces[i], v_faces_: v_faces[i]}
            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}

            print("Running points...")

            # normalised_disp_fine = new_normals0
            normalised_disp_fine = n_conv0

            # outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, new_normals0, upN1, upN2],feed_dict=update_feed_dict)
            outPoints, fineNormals = sess.run([points, normalised_disp_fine],feed_dict=update_feed_dict)

            print("Points: check")
            print("Updating mesh...")
            if len(f_normals)>1:
                finalOutPoints[new_to_old_v_list[i]] += outPoints
                pointsWeights[new_to_old_v_list[i]]+=1

                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]

                finalFineNormals[new_to_old_f_list[i]] = fineNormalsP
                
            else:
                finalOutPoints = outPoints
                pointsWeights +=1

                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                
                finalFineNormals = fineNormalsP
            
            print("Mesh update: check")
        sess.close()

        finalOutPoints = np.true_divide(finalOutPoints,pointsWeights)

        return finalOutPoints, finalFineNormals



def trainNet(f_normals_list, GTfn_list, f_adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list):
    
    random_seed = 0
    np.random.seed(random_seed)

    sess = tf.InteractiveSession()
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


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

    tfn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='tfn_')

    sample_ind = tf.placeholder(tf.int32, shape=[10000], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    #rotTens = getRotationToAxis(fn_)

    #Add random rotation
    fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,2,3])    # 2 because of normal + position
    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
    tfn_rot = tf.reshape(tfn_,[BATCH_SIZE,-1,3,1])
    
    fn_rot = tf.matmul(rot_mat,fn_rot)
    tfn_rot = tf.matmul(rot_mat,tfn_rot)

    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
    fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,6])
    tfn_rot = tf.reshape(tfn_rot,[BATCH_SIZE,-1,3])

    fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        # n_conv = get_model_reg(fn_, fadj0, ARCHITECTURE, keep_prob)
        n_conv,_,_ = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)


    # n_conv = normalizeTensor(n_conv)
    # n_conv = tf.expand_dims(n_conv,axis=-1)
    # n_conv = tf.matmul(tf.transpose(rotTens,[0,1,3,2]),n_conv)
    # n_conv = tf.reshape(n_conv,[BATCH_SIZE,-1,3])
    # n_conv = tf.slice(fn_,[0,0,0],[-1,-1,3])+n_conv
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
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)

    saver = tf.train.Saver()

    # # get variables to restore...
    # dictVar = {}
    # listVar = []
    # for opname in ["weight", "weight_1", "weight_2", "bias", "bias_1", "bias_2", "assignment", "assignment_1", "assignment_2",
    #               "assignment_3", "assignment_4", "assignment_5", "assignment_6", "assignment_7", "assignment_8"]:
    #   print("opname = "+ opname)
    #   varname = "model/"+opname
    #   myvar = [var for var in tf.global_variables() if var.op.name==varname][0]
    #   dictVar[varname] = myvar
    #   listVar.append(myvar)

    # #extras:
    # # dictVar["model/weight_3"] = [var for var in tf.global_variables() if var.op.name=="model/weight_5"][0]
    # # dictVar["model/bias_3"] = [var for var in tf.global_variables() if var.op.name=="model/bias_5"][0]
    # dictVar["model/weight_4"] = [var for var in tf.global_variables() if var.op.name=="model/weight_6"][0]
    # dictVar["model/bias_4"] = [var for var in tf.global_variables() if var.op.name=="model/bias_6"][0]

    # saver = tf.train.Saver(dictVar)

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
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

    with tf.device(DEVICE):
        lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        last_loss = 0
        for iter in range(NUM_ITERATIONS):


            # Get random sample from training dictionary
            batch_num = random.randint(0,len(f_normals_list)-1)

            while batch_num in forbidden_examples:
                batch_num = random.randint(0,len(f_normals_list)-1)
            num_p = f_normals_list[batch_num].shape[1]
            random_ind = np.random.randint(num_p,size=10000)

            random_R = rand_rotation_matrix()
            tens_random_R = np.reshape(random_R,(1,1,3,3))
            tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

            # train_fd = {fn_: f_normals_list[batch_num], fadj: f_adj_list[batch_num], tfn_: GTfn_list[batch_num],
            #               sample_ind: random_ind, keep_prob:1}

            # train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], tfn_: GTfn_list[batch_num],
            #               sample_ind: random_ind, keep_prob:1}

            train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
                            fadj2: f_adj_list[batch_num][2], tfn_: GTfn_list[batch_num], rot_mat:tens_random_R2,
                            sample_ind: random_ind, keep_prob:1}


            #i = train_shuffle[iter%(len(train_data))]
            #in_points = train_data[i]

            #sess.run(customLoss,feed_dict=train_fd)
            train_loss += customLoss.eval(feed_dict=train_fd)
            train_samp+=1
            # Show smoothed training loss
            if (iter%10 == 0):
                train_loss = train_loss/train_samp
                # sess.run(customLoss2,feed_dict=my_feed_dict)
                # train_loss2 = customLoss2.eval(feed_dict=my_feed_dict)
                # sess.run(customLoss3,feed_dict=my_feed_dict)
                # train_loss3 = customLoss3.eval(feed_dict=my_feed_dict)

                print("Iteration %d, training loss %g"%(iter, train_loss))
                # print("Iteration %d, training loss2 %g"%(iter, train_loss2))
                # print("Iteration %d, training loss3 %g"%(iter, train_loss3))

                lossArray[int(iter/10),0]=train_loss
                train_loss=0
                train_samp=0

            # Compute validation loss
            if (iter%20 ==0):
                valid_loss = 0
                valid_samp = len(valid_f_normals_list)
                valid_random_ind = np.random.randint(num_p,size=10000)
                for vbm in range(valid_samp):
                    # valid_fd = {fn_: valid_f_normals_list[vbm], fadj: valid_f_adj_list[vbm], tfn_: valid_GTfn_list[vbm],
                    #       sample_ind: valid_random_ind, keep_prob:1}

                    # valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], tfn_: valid_GTfn_list[vbm],
                    #       sample_ind: valid_random_ind, keep_prob:1}
                    num_p = valid_f_normals_list[vbm].shape[1]
                    tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
                    valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                            fadj2: valid_f_adj_list[vbm][2], tfn_: valid_GTfn_list[vbm], rot_mat:tens_random_R2,
                            sample_ind: valid_random_ind, keep_prob:1}

                    valid_loss += customLoss.eval(feed_dict=valid_fd)
                valid_loss/=valid_samp
                print("Iteration %d, validation loss %g"%(iter, valid_loss))
                lossArray[int(iter/10),1]=valid_loss
                if iter>0:
                    lossArray[int(iter/10)-1,1] = (valid_loss+last_loss)/2
                    last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            # sess.run(train_step2,feed_dict=my_feed_dict)
            # sess.run(train_step3,feed_dict=my_feed_dict)
            if sess.run(isNanNConv,feed_dict=train_fd):
                hasNan = True
                print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(f_normals_list)))
                print("patch size: "+str(f_normals_list[batch_num].shape))
            if (iter%2000 == 0):
                saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
                if sess.run(isFullNanNConv, feed_dict=train_fd):
                    break
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()


def trainAccuracyNet(in_points_list, GT_points_list, faces_list, f_normals_list, f_adj_list, v_faces_list, valid_in_points_list, valid_GT_points_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list):
    
    random_seed = 0
    np.random.seed(random_seed)
    SAMP_NUM = 500
    keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


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
    #fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj')

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


    #Add random rotation
    fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,2,3])    # 2 because of normal + position
    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
    
    vp_rot = tf.reshape(vp_,[BATCH_SIZE,-1,3,1])
    gtvp_rot = tf.reshape(gtvp_,[BATCH_SIZE,-1,3,1])
    
    if keep_rot_inv:
        fn_rot = tf.matmul(rot_mat,fn_rot)
        vp_rot = tf.matmul(rot_mat_vert,vp_rot)
        gtvp_rot = tf.matmul(rot_mat_gt,gtvp_rot)
    else:
        print("WARNING: hard-coded rot inv removal")

    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
    fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,6])

    vp_rot = tf.reshape(vp_rot,[BATCH_SIZE,-1,3])
    gtvp_rot = tf.reshape(gtvp_rot,[BATCH_SIZE,-1,3])

    fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)
        # n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)


    # n_conv0 = normalizeTensor(n_conv0)
    # n_conv1 = normalizeTensor(n_conv1)
    # n_conv2 = normalizeTensor(n_conv2)

    n_conv_list = [n_conv0, n_conv1, n_conv2]
    # isNanNConv = tf.reduce_any(tf.is_nan(n_conv), name="isNanNConv")
    # isFullNanNConv = tf.reduce_all(tf.is_nan(n_conv), name="isNanNConv")

    # refined_x, _ = update_position_MS(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num=80)

    refined_x, _ = update_position_disp(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS)
    # refined_x = update_position2(vp_rot, n_conv, e_map_, ve_map_, iter_num=2000)
    

    # samp_x = tf.transpose(refined_x,[1,0,2])
    # samp_x = tf.gather(samp_x,sample_ind0)
    # samp_x = tf.transpose(samp_x,[1,0,2])
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

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
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

    # with tf.device(DEVICE):
    # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
    lossArray = np.zeros([int(50),2])                  # 200 = 2000/10: save csv file every 2000 iter
    last_loss = 0
    lossArrayIter = 0
    for iter in range(NUM_ITERATIONS):


        # Get random sample from training dictionary
        batch_num = random.randint(0,len(f_normals_list)-1)
        # print("Selecting patch "+str(batch_num)+" on "+str(len(f_normals_list)))
        num_vgt = GT_points_list[batch_num].shape[1]
        num_vnoisy = in_points_list[batch_num].shape[1]

        while (num_vgt==0):
            print("WOLOLOLOLOLO " +str(batch_num))
            batch_num = random.randint(0,len(f_normals_list)-1)
            num_vgt = GT_points_list[batch_num].shape[1]


        while batch_num in forbidden_examples:
            batch_num = random.randint(0,len(f_normals_list)-1)
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
                # print("valid sample "+str(vbm)+": loss = "+str(valid_loss_cur))
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
            saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
            # if sess.run(isFullNanNConv, feed_dict=train_fd):
            #     break
            csv_filename = RESULTS_PATH+NET_NAME+".csv"
            f = open(csv_filename,'ab')
            np.savetxt(f,lossArray, delimiter=",")
            f.close()
            lossArray = np.zeros([int(50),2]) 
            lossArrayIter=0

        lossArrayIter+=1
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()


def train6DNetWGT(f_normals_list, GTfn_list, f_adj_list, gt_disp_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_gt_disp_list):
    
    random_seed = 0
    np.random.seed(random_seed)
    SAMP_NUM = 500
    keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


    """
    Load dataset 
    x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
    adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                              K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
    """
    
    dropout_prob = 0.8
    BATCH_SIZE=1
    K_faces = f_adj_list[0][0].shape[2]
    NUM_IN_CHANNELS = f_normals_list[0].shape[2]
    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')
    tfn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='tfn_')
    tdisp_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='tdisp_')
    fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    # sample_ind0 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind0')
    # sample_ind1 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind1')

    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.

    keep_prob = tf.placeholder(tf.float32)

    batch = tf.Variable(0, trainable=False)

    


    # --- Starting iterative process ---


    #Add random rotation
    fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,2,3])    # 2 because of normal + position
    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
    tfn_rot = tf.reshape(tfn_,[BATCH_SIZE,-1,3,1])
    tdisp_rot = tf.reshape(tdisp_,[BATCH_SIZE,-1,3,1])
    
    fn_rot = tf.matmul(rot_mat,fn_rot)
    tfn_rot = tf.matmul(rot_mat,tfn_rot)
    tdisp_rot = tf.matmul(rot_mat,tdisp_rot)

    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
    fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,6])
    tfn_rot = tf.reshape(tfn_rot,[BATCH_SIZE,-1,3])
    tdisp_rot = tf.reshape(tdisp_rot,[BATCH_SIZE,-1,3])




    fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        n_conv, disp = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)
        

    n_conv = normalizeTensor(n_conv)
    

    

    
    with tf.device(DEVICE):
        
        gtfn_abs_sum = tf.reduce_sum(tf.abs(tfn_rot),axis=2)
        fakenodes = tf.less_equal(gtfn_abs_sum,10e-4)

        dispLoss = 100*mseLoss(disp, tdisp_rot, fakenodes)
        nLoss = squareFaceNormalsLoss(n_conv,tfn_rot)
        # customLoss = dispLoss + nLoss
        customLoss = tf.add(dispLoss,nLoss)
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
    if ckpt and ckpt.model_checkpoint_path:
        splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
        if splitCkpt[0] == NET_NAME:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #Extract from checkpoint filename
            globalStep = int(splitCkpt[1])
    

    # Training

    train_loss=0
    disp_loss=0
    normals_loss=0
    train_samp=0

    hasNan = False
    forbidden_examples = []

    # with tf.device(DEVICE):
    # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
    lossArray = np.zeros([int(50),2])                  # 200 = 2000/10: save csv file every 2000 iter
    last_loss = 0
    lossArrayIter = 0
    for iter in range(NUM_ITERATIONS):


        # Get random sample from training dictionary
        batch_num = random.randint(0,len(f_normals_list)-1)
        
        while batch_num in forbidden_examples:
            batch_num = random.randint(0,len(f_normals_list)-1)
        num_p = f_normals_list[batch_num].shape[1]
        

        random_R = rand_rotation_matrix()
        tens_random_R = np.reshape(random_R,(1,1,3,3))
        tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))
        
        # batch_in_points = np.transpose(in_points_list[batch_num],[1,0,2])
        # batch_in_points = batch_in_points[random_ind]
        # batch_in_points = np.transpose(batch_in_points,[1,0,2])

        train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
                        fadj2: f_adj_list[batch_num][2], tfn_: GTfn_list[batch_num], rot_mat:tens_random_R2,
                        tdisp_: gt_disp_list[batch_num], keep_prob:dropout_prob}


        # train_loss_cur = customLoss.eval(feed_dict=train_fd)
        disp_loss_cur, normals_loss_cur, train_loss_cur = sess.run([dispLoss,nLoss,customLoss],feed_dict=train_fd) 

        disp_loss += disp_loss_cur
        train_loss += train_loss_cur
        normals_loss += normals_loss_cur
        train_samp+=1
        # Show smoothed training loss
        if (iter%10 == 0)and(iter>-1):
            train_loss = train_loss/train_samp
            disp_loss = disp_loss/train_samp
            normals_loss = normals_loss/train_samp


            print("Iteration %d, training loss: (%g, %g)"%(iter, disp_loss, normals_loss))

            lossArray[int(lossArrayIter/10)-1,0]=train_loss
            train_loss=0
            disp_loss=0
            normals_loss=0
            train_samp=0

        # Compute validation loss
        if ((iter%20 ==0)and(iter>0)):
            valid_loss = 0
            valid_disp_loss = 0
            valid_normals_loss = 0
            valid_samp = len(valid_f_normals_list)
            for vbm in range(valid_samp):
                num_p = valid_f_normals_list[vbm].shape[1]

                tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

                valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                        fadj2: valid_f_adj_list[vbm][2], tfn_: valid_GTfn_list[vbm],
                        rot_mat:tens_random_R2, tdisp_: valid_gt_disp_list[vbm], keep_prob:1.0}

                # valid_loss_cur = customLoss.eval(feed_dict=valid_fd)
                valid_disp_loss_cur, valid_normals_loss_cur, valid_loss_cur = sess.run([dispLoss, nLoss, customLoss], feed_dict=valid_fd)
                valid_loss += valid_loss_cur
                valid_disp_loss += valid_disp_loss_cur
                valid_normals_loss += valid_normals_loss_cur
            valid_loss/=valid_samp
            valid_disp_loss/= valid_samp
            valid_normals_loss/= valid_samp
            print("Iteration %d, validation loss: (%g, %g)"%(iter, valid_disp_loss, valid_normals_loss))
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
            saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
            # if sess.run(isFullNanNConv, feed_dict=train_fd):
            #     break
            csv_filename = RESULTS_PATH+NET_NAME+".csv"
            f = open(csv_filename,'ab')
            np.savetxt(f,lossArray, delimiter=",")
            f.close()
            lossArray = np.zeros([int(50),2]) 
            lossArrayIter=0

        lossArrayIter+=1
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()



def trainSoftNormalsNet(
    in_points_list, GT_points_list, faces_list, f_normals_list, f_adj_list, v_faces_list, gt_faces_list, face_assignment_list,
    valid_in_points_list, valid_GT_points_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list, valid_gt_faces_list, valid_face_assignment_list):
    
    random_seed = 0
    np.random.seed(random_seed)
    SAMP_NUM = 500
    keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


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
    print("face_assignment_list shape: "+str(face_assignment_list[41].shape))
    K_assignment = face_assignment_list[0].shape[2]

    # # Hard values test
    # Num_faces = faces_list[41].shape[1]
    # Num_gt_faces = gt_faces_list[41].shape[1]
    # Num_gt_points = GT_points_list[41].shape[1]


    # coarsening_steps=3
    # # training data
    # fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, Num_faces, NUM_IN_CHANNELS], name='fn_')
    # #fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj')

    # vp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, 3], name='vp_')
    # gtvp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, Num_gt_points, 3], name='gtvp_')

    # fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, Num_faces, K_faces], name='fadj0')
    # fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, Num_faces/(2**coarsening_steps), K_faces], name='fadj1')
    # fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, Num_faces/(2**(2*coarsening_steps)), K_faces], name='fadj2')

    # # Needed for vertices update
    # # e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,4], name='e_map_')
    # # ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,MAX_EDGES], name='ve_map_')
    # faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, Num_faces, 3], name='faces_')
    # gt_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, Num_gt_faces, 3], name='gt_faces_')

    # face_assignment_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, Num_faces, K_assignment], name='face_assignment_')
    # v_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K_vertices], name='v_faces_')

    # sample_ind0 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind0')
    # sample_ind1 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind1')


    # keep_prob = tf.placeholder(tf.float32)
    
    # rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,Num_faces,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    # rot_mat_vert = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_POINTS,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of vertices rather than number of faces.
    # rot_mat_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE,Num_gt_points,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of GT vertices rather than number of faces.







    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, NUM_IN_CHANNELS], name='fn_')
    #fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj')

    vp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='vp_')
    gtvp_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 3], name='gtvp_')

    fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    # Needed for vertices update
    # e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,4], name='e_map_')
    # ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,None,MAX_EDGES], name='ve_map_')
    faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 3], name='faces_')
    gt_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, 3], name='gt_faces_')

    face_assignment_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_assignment], name='face_assignment_')
    v_faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_vertices], name='v_faces_')

    sample_ind0 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind0')
    sample_ind1 = tf.placeholder(tf.int32, shape=[SAMP_NUM], name='sample_ind1')


    keep_prob = tf.placeholder(tf.float32)
    
    rot_mat = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Random rotation matrix, used for data augmentation. Generated anew for each training iteration. None correspond to the tiling for each face.
    rot_mat_vert = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of vertices rather than number of faces.
    rot_mat_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE,None,3,3),name='rot_mat')    #Same rotation matrix, but tiled by number of GT vertices rather than number of faces.
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    #Add random rotation
    fn_rot = tf.reshape(fn_,[BATCH_SIZE,-1,2,3])    # 2 because of normal + position
    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # switch dimensions
    
    vp_rot = tf.reshape(vp_,[BATCH_SIZE,-1,3,1])
    gtvp_rot = tf.reshape(gtvp_,[BATCH_SIZE,-1,3,1])
    
    if keep_rot_inv:
        fn_rot = tf.matmul(rot_mat,fn_rot)
        vp_rot = tf.matmul(rot_mat_vert,vp_rot)
        gtvp_rot = tf.matmul(rot_mat_gt,gtvp_rot)
    else:
        print("WARNING: hard-coded rot inv removal")

    fn_rot = tf.transpose(fn_rot,[0,1,3,2])         # Put it back
    fn_rot = tf.reshape(fn_rot,[BATCH_SIZE,-1,6])

    vp_rot = tf.reshape(vp_rot,[BATCH_SIZE,-1,3])
    gtvp_rot = tf.reshape(gtvp_rot,[BATCH_SIZE,-1,3])

    fadjs = [fadj0,fadj1,fadj2]

    print("fn_rot shape: "+str(fn_rot.shape))
    print("fadj0 shape: "+str(fadj0.shape))

    with tf.variable_scope("model"):
        n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)
        # n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)


    n_conv0 = normalizeTensor(n_conv0)
    n_conv1 = normalizeTensor(n_conv1)
    n_conv2 = normalizeTensor(n_conv2)

    n_conv_list = [n_conv0, n_conv1, n_conv2]
    # isNanNConv = tf.reduce_any(tf.is_nan(n_conv), name="isNanNConv")
    # isFullNanNConv = tf.reduce_all(tf.is_nan(n_conv), name="isNanNConv")

    refined_x, _ = update_position_MS(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS, iter_num=80)

    # refined_x, _ = update_position_disp(vp_rot, n_conv_list, faces_, v_faces_, coarsening_steps=COARSENING_STEPS)
    # refined_x = update_position2(vp_rot, n_conv, e_map_, ve_map_, iter_num=2000)
    

    # samp_x = tf.transpose(refined_x,[1,0,2])
    # samp_x = tf.gather(samp_x,sample_ind0)
    # samp_x = tf.transpose(samp_x,[1,0,2])
    samp_x = refined_x
    
    with tf.device(DEVICE):
        # customLoss = accuracyLoss(refined_x, gtvp_rot, sample_ind0)
        # customLoss = fullLoss(refined_x, gtvp_rot, sample_ind0, sample_ind1) + connectivityRegularizer(refined_x,faces_,v_faces_, sample_ind0)
        
        gtfn = tfComputeNormals(gtvp_rot,gt_faces_)
        fC = updateFacesCenter(refined_x, faces_,1)
        fC = fC[0]
        gt_fC = updateFacesCenter(gtvp_rot,gt_faces_,1)
        gt_fC = gt_fC[0]
        customLoss = softFaceNormalsLoss(n_conv0, gtfn, fC, gt_fC, face_assignment_)
        # customLoss = sampledAccuracyLoss(samp_x, gtvp_rot)
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
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

    # with tf.device(DEVICE):
    # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
    lossArray = np.zeros([int(50),2])                  # 200 = 2000/10: save csv file every 2000 iter
    last_loss = 0
    lossArrayIter = 0

    for iter in range(NUM_ITERATIONS):


        # Get random sample from training dictionary
        batch_num = random.randint(0,len(f_normals_list)-1)
        # print("Selecting patch "+str(batch_num)+" on "+str(len(f_normals_list)))
        num_vgt = GT_points_list[batch_num].shape[1]
        num_vnoisy = in_points_list[batch_num].shape[1]

        while (num_vgt==0):
            print("WOLOLOLOLOLO " +str(batch_num))
            batch_num = random.randint(0,len(f_normals_list)-1)
            num_vgt = GT_points_list[batch_num].shape[1]


        while batch_num in forbidden_examples:
            batch_num = random.randint(0,len(f_normals_list)-1)
        # batch_num=41
        num_p = f_normals_list[batch_num].shape[1]
        num_v = in_points_list[batch_num].shape[1]
        num_vgt = GT_points_list[batch_num].shape[1]
        random_ind0 = np.random.randint(num_v,size=SAMP_NUM)
        random_ind1 = np.random.randint(num_vgt,size=SAMP_NUM)

        # print("face num = "+str(num_p)+", gt face num = "+str(gt_faces_list[batch_num].shape[1]))
        # print("batch_num = "+str(batch_num))
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
                        faces_: faces_list[batch_num], v_faces_: v_faces_list[batch_num], gt_faces_: gt_faces_list[batch_num],
                        rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
                        sample_ind0: random_ind0, sample_ind1: random_ind1, keep_prob:dropout_prob, face_assignment_: face_assignment_list[batch_num]}
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
                        faces_:valid_faces_list[vbm], v_faces_:valid_v_faces_list[vbm], gt_faces_: valid_gt_faces_list[vbm],
                        rot_mat:tens_random_R2, rot_mat_vert:tens_random_Rv, rot_mat_gt: tens_random_Rgt,
                        sample_ind0: valid_random_ind0, sample_ind1: valid_random_ind1, keep_prob:1.0, face_assignment_: valid_face_assignment_list[vbm]}

                valid_loss_cur = customLoss.eval(feed_dict=valid_fd)
                # print("valid sample "+str(vbm)+": loss = "+str(valid_loss_cur))
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
            saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
            # if sess.run(isFullNanNConv, feed_dict=train_fd):
            #     break
            csv_filename = RESULTS_PATH+NET_NAME+".csv"
            f = open(csv_filename,'ab')
            np.savetxt(f,lossArray, delimiter=",")
            f.close()
            lossArray = np.zeros([int(50),2]) 
            lossArrayIter=0

        lossArrayIter+=1
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()




def faceNormalsLoss(fn,gt_fn):

    #version 1
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
    loss = 180*loss/math.pi
    # loss = 1 - n_d

    #Set loss to zero for fake nodes
    loss = tf.where(fakenodes,zeroVec,loss)
    loss = tf.reduce_sum(loss)/tf.reduce_sum(realnodes)
    #loss = tf.reduce_mean(loss)
    return loss

def squareFaceNormalsLoss(fn,gt_fn):

    #version 1
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
    loss = tf.square(loss)
    # loss = 180*loss/math.pi
    # loss = 1 - n_d

    #Set loss to zero for fake nodes
    loss = tf.where(fakenodes,zeroVec,loss)
    loss = tf.reduce_sum(loss)/tf.reduce_sum(realnodes)
    #loss = tf.reduce_mean(loss)
    return loss

def softFaceNormalsLoss(fn,gt_fn, fC, gt_fC, face_assignment):

    fn = tf.reshape(fn,[-1,3])
    gt_fn = tf.reshape(gt_fn,[-1,3])
    gt_fC = tf.reshape(gt_fC,[-1,3])
    fC = tf.reshape(fC,[-1,3])
    _,_,K_assignment = face_assignment.get_shape().as_list()
    print("gt_fn shape: "+str(gt_fn.shape))
    print("gt_fC shape: "+str(gt_fC.shape))
    face_assignment = tf.reshape(face_assignment,[-1,K_assignment])
    print("face_assignment shape: "+str(face_assignment.shape))
    face_assignment = face_assignment+1
    # Test
    # face_assignment = tf.slice(face_assignment,(0,0),(-1,1))
    # [gtFNum, 3]
    gt_fCOff = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),gt_fC),axis=0)
    gt_fnOff = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),gt_fn),axis=0)
    ngtC = tf.gather(gt_fCOff,face_assignment)
    ngtN = tf.gather(gt_fnOff,face_assignment)
    print("ngtC shape: "+str(ngtC.shape))
    # [FNum, assignNum, 3]

    diff = tf.expand_dims(fC,axis=1) - ngtC
    print("diff shape: "+str(diff.shape))
    dist = tf.norm(diff, axis=-1)
    print("dist shape: "+str(dist.shape))
    # [FNum, assignNum]
    sigma = tf.reduce_min(dist,axis=-1)
    # [FNum]
    print("sigma shape: "+str(sigma.shape))
    # weights = tf.exp(-tf.square(dist)/(2*tf.expand_dims(tf.square(sigma+0.0001),axis=-1)))
    weights = tf.ones_like(dist)
    # [FNum, assignNum]
    print("weights shape: "+str(weights.shape))
    weights = tf.expand_dims(weights,axis=-1)
    newGtN = tf.multiply(weights,ngtN)
    # [fNum, assignNum,3]
    print("newGtN shape: "+str(newGtN.shape))

    newGtN = tf.reduce_sum(newGtN,axis=1)/tf.reduce_sum(weights,axis=1)
    print("newGtN shape: "+str(newGtN.shape))
    # [FNum, 3]

    # Test
    # newGtN = tf.slice(ngtN,(0,0,0),(-1,1,-1))
    # newGtN = tf.reshape(newGtN,[-1,3])

    n_dt = tensorDotProduct(fn,newGtN)
    # [1, fnum]
    #loss = tf.acos(n_dt-1e-5)    # So that it stays differentiable close to 1
    close_to_one = 0.999999999
    loss = tf.acos(tf.minimum(tf.maximum(n_dt,-close_to_one),close_to_one))    # So that it stays differentiable close to 1 and -1
    fakenodes = tf.reduce_any(tf.equal(face_assignment,0),axis=-1)
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


def mseLoss(prediction, gt, fakenodes):


    # gt_abs_sum = tf.reduce_sum(tf.abs(gt),axis=-1)
    # fakenodes = tf.less_equal(gt_abs_sum,10e-4)
    
    
    
    loss = tf.reduce_sum(tf.square(tf.subtract(gt,prediction)),axis=-1)

    zeroVec = tf.zeros_like(loss)
    oneVec = tf.ones_like(loss)
    realnodes = tf.where(fakenodes,zeroVec,oneVec)

    #Set loss to zero for fake nodes
    loss = tf.where(fakenodes,zeroVec,loss)
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

    accuracyThreshold = 5      # Completely empirical
    compThreshold = 0.05
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
def fullLossWNormals(P0, P1, N1, sample_ind0, sample_ind1):

    accuracyThreshold = 5      # Completely empirical
    compThreshold = 0.05
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


def AwesomeLoss(P0, P1, N1, sample_ind0, sample_ind1):

    accuracyThreshold = 5      # Completely empirical
    compThreshold = 0.05
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


def connectivityRegularizer(x,faces, v_faces, sample_ind):


    x = tf.reshape(x,[-1,3])
    _, _, K = v_faces.get_shape().as_list()

    new_fpos = updateFacesCenter(x,faces,1)
    face_pos = new_fpos[0]
    face_pos = tf.reshape(face_pos,[-1,3])
    v_faces = tf.squeeze(v_faces)

    # Add fake face with null normal and switch to 1-indexing for v_faces
    v_faces = v_faces+1
    face_pos = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),face_pos),axis=0)
    v_c = tf.gather(face_pos,v_faces)

    xm = tf.expand_dims(x,axis=1)
    xm = tf.tile(xm,[1,K,1])

    e = v_c-xm

    dist = tf.norm(e,axis=-1, name='norm')
    dist = tf.multiply(dist,dist)
    dist = tf.gather(dist,sample_ind)

    cost = 1000 * tf.reduce_mean(dist)

    return cost

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
def update_position2(x, face_normals, edge_map, v_edges, iter_num=20):

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


def update_position_MS(x, face_normals_list, faces, v_faces0, coarsening_steps, iter_num=180):

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
        print("WARNING! Hard-coded fine scale vertex update")
        if cur_scale>0:
            continue

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
        if cur_scale==2:
            iter_num=iter_num
        else:
            iter_num=iter_num

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

def update_position_disp(x, face_normals_list, faces, v_faces0, coarsening_steps):

    batch_size, num_points, space_dims = x.get_shape().as_list()
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

        # print("WARNING! Hard-coded mid-and-fine scale vertex update")
        # if cur_scale>1:
        #     continue
        print("WARNING! Hard-coded fine scale vertex update")
        if cur_scale>0:
            continue

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

        v_disp = tf.reduce_sum(v_fn,axis=1)
        v_disp = tf.multiply(lmbd,v_disp)
        x = tf.add(x,v_disp)
        
    

        dx_list.append(v_disp)
    x = tf.expand_dims(x,axis=0)
    return x, dx_list


def update_position_MS_damp(x, face_normals_list, faces, v_faces0, coarsening_steps, iter_num=80):

    batch_size, num_points, space_dims = x.get_shape().as_list()
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

        # print("WARNING! Hard-coded mid-and-fine scale vertex update")
        # if cur_scale>1:
        #     continue
        print("WARNING! Hard-coded fine scale vertex update")
        if cur_scale>0:
            continue
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
        if cur_scale==2:
            iter_num=iter_num
        else:
            iter_num=iter_num

        for it in range(iter_num):
            w_c = max(0, 0.5 - 0.5 * (it/iter_num))
            print("Scale "+str(cur_scale)+", iter "+str(it))
            
            new_fpos, new_fnorm = updateFacesCenterAndNormals(x,faces,coarsening_steps)
            face_pos = new_fpos[cur_scale]
            face_pos = tf.reshape(face_pos,[-1,3])

            face_norm = new_fnorm[cur_scale]
            face_norm = tf.reshape(face_norm,[-1,3])

            face_pos = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),face_pos),axis=0)
            face_norm = tf.concat((tf.constant([[0,0,0]], dtype=tf.float32),face_norm),axis=0)

            v_c = tf.gather(face_pos,v_faces)
            # [vnum, v_faces_num, 3]

            # xm = tf.reshape(x,[-1,1,3])
            xm = tf.expand_dims(x,axis=1)
            xm = tf.tile(xm,[1,K,1])

            v_cfn = tf.gather(face_norm, v_faces)


            v_cfn = w_c * v_cfn + (1-w_c) * v_fn
            v_cfn = normalizeTensor(v_cfn)

            e = v_c - xm

            n_w = tensorDotProduct(v_cfn, e)
            # [vnum, v_faces_num]
            # Since face_n should be null for 'non-faces' in v_faces, dot product should be null, so there is no need to deal with this explicitly

            n_w = tf.tile(tf.expand_dims(n_w,axis=-1),[1,1,3])
            # [vnum, v_faces_num, 3]

            update = tf.multiply(n_w,v_cfn)
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


def updateFacesCenterAndNormals(vertices, faces, coarsening_steps):

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

    fvpos0 = tf.gather(vertices,faces)
    # [fnum, 3, 3]

    # --- Position part ---
    fpos0 = tf.reduce_mean(fvpos0,axis=1)
    # [fnum, 3]

    fpos0 = tf.expand_dims(fpos0,axis=0)
    # We have position of finest graph nodes
    fpos1 = custom_binary_tree_pooling(fpos0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
    fpos2 = custom_binary_tree_pooling(fpos1, steps=coarsening_steps, pooltype='avg_ignore_zeros')

    # --- Normals part ---
    fv0 = tf.slice(fvpos0,[0,0,0],[-1,1,-1])
    fv1 = tf.slice(fvpos0,[0,1,0],[-1,1,-1])
    fv2 = tf.slice(fvpos0,[0,2,0],[-1,1,-1])
    # [fnum,1,3]
    N = tf.cross(fv1-fv0, fv2-fv0)
    N = tf.reshape(N,[-1,3])
    N = tf.expand_dims(N,axis=0)
    N0 = normalizeTensor(N)

    N1 = custom_binary_tree_pooling(N0, steps=coarsening_steps, pooltype='avg_ignore_zeros')
    N1 = normalizeTensor(N1)
    N2 = custom_binary_tree_pooling(N1, steps=coarsening_steps, pooltype='avg_ignore_zeros')
    N2 = normalizeTensor(N2)

    return [fpos0, fpos1, fpos2], [N0, N1, N2]


def mainFunction():


    pickleLoad = True
    pickleSave = True

    K_faces = 30

    maxSize = 30000 #35000
    patchSize = 30000 #15000

    training_meshes_num = [0]
    valid_meshes_num = [0]

    #binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/smallAdj/"
    # binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/bigAdj/"
    # binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/coarsening4/"

    # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/coarsening8/patches5/"

    # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/newTest/"

    # if COARSENING_STEPS==3:
    #     # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/msVertices/"
    #     binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/tola/c8/"
    # elif COARSENING_STEPS==2:
    #     # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/msVertices_c4/"
    #     # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/furu/cleaned_c4/"
    #     binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/BinaryDump/tola/c4/"

    # if COARSENING_STEPS==3:
    #     binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/Synthetic/BinaryDump/msVertices_c8/"
    # elif COARSENING_STEPS==2:
    #     binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/Synthetic/BinaryDump/msVertices_c4/"

    if COARSENING_STEPS==3:
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/msVertices_c8_clean/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c8_decim_gauss/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c8_debug/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c8_gauss/"

        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/faceAssignment_c8/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/faceAssignment_c8_30/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/fullGT_c8/"
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/msVertices_c8_decim/"
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c8_decim/"
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c8_decim_gauss_clean_full/"
    elif COARSENING_STEPS==2:
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/msVertices_c4/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/msVertices_c4_decim/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c4_decim/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c4_decim_gauss_full/"
        binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c4_decim_gauss_clean_full/"
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/normals_c4_decim_gauss/"
        # binDumpPath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/BinaryDump/newT2/"

    empiricMax = 30.0

    # Coarsening parameters
    coarseningLvlNum = 3
    coarseningStepNum = COARSENING_STEPS


    #binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/kinect_v1/coarsening4/"

    # binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/kinect_v2/coarsening4/"

    # binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/kinect_fusion/coarsening4/"

    #binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/MS9_res/"


    running_mode = RUNNING_MODE
    ###################################################################################################
    #   0 - Training on all meshes in a given folder
    #   1 - Run checkpoint on a given mesh as input
    #   2 - Run checkpoint on all meshes in a folder. Compute angular diff and Haus dist
    #   3 - Test mode
    ###################################################################################################


    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMesh(inputFilePath,filename, gtFilePath, gtfilename, in_list, gt_list, adj_list, mesh_count_list):
        patch_indices = []
        new_to_old_permutations_list = []
        num_faces = []

        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
        print("faces0 shape: "+str(faces0.shape))
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0)

        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)
        # f_area0 = getTrianglesArea(V0,faces0)
        # f_area0 = np.reshape(f_area0, (-1,1))
        # f_normals0 = np.concatenate((f_normals0, f_area0), axis=1)

        # Load GT
        GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)
        GTf_normals0 = computeFacesNormals(GT0, faces0)

        # Get patches if mesh is too big
        facesNum = faces0.shape[0]

        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>maxSize:
            patchNum = 0
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = GTf_normals0[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue
                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                patchGTFNormals = patchGTFNormals[newToOld]

                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)
                new_to_old_permutations_list.append(newToOld)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                #f_adj = np.expand_dims(testPatchAdj, axis=0)
                GTf_normals = np.expand_dims(patchGTFNormals, axis=0)

                in_list.append(f_normals)
                adj_list.append(fAdjs)
                gt_list.append(GTf_normals)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            print("f_adj0 shape: "+str(f_adj0.shape))
            print("f_pos0 shape: "+str(f_pos0.shape))
            coo_adj = listToSparse(f_adj0, f_pos0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            padding6 =np.zeros((new_N-old_N,6))
            padding3 =np.zeros((new_N-old_N,3))
            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            GTf_normals0 = np.concatenate((GTf_normals0, padding3),axis=0)

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            new_to_old_permutations_list.append(newToOld) # Nothing to append here, faces are already correctly ordered
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            GTf_normals0 = GTf_normals0[newToOld]

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            #f_adj = np.expand_dims(f_adj0, axis=0)
            GTf_normals = np.expand_dims(GTf_normals0, axis=0)

            in_list.append(f_normals)
            adj_list.append(fAdjs)
            gt_list.append(GTf_normals)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return num_faces, patch_indices, new_to_old_permutations_list

    
    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMeshWithVertices(inputFilePath,filename, gtFilePath, gtfilename, v_list, gtv_list, faces_list, n_list, adj_list, v_faces_list, mesh_count_list):
        patch_indices = []
        new_to_old_permutations_list = []
        num_faces = []
        vOldInd_list = []
        fOldInd_list = []

        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)

        fNum = faces0.shape[0]
        vNum = V0.shape[0]
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0, normalize=False)
        # f_pos0 = np.reshape(f_pos0,(-1,3))

        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)

        # Load GT
        GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)


        # Normalize vertices
        # V0, GT0 = normalizePointSets(V0,GT0)


        # Get patches if mesh is too big
        facesNum = faces0.shape[0]
        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        print("maxSize = "+str(maxSize))
        print("facesNum = "+str(facesNum))
        if facesNum>maxSize:
            patchNum = 0
            while((np.any(faceCheck==0))and(patchNum<3)):
            # while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)
                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue
                
                # For CNR dataset: one-one correspondence between vertices
                # patchGTV = GT0[vOldInd]

                # For DTU: take slice of GT points
                patchBB = getBoundingBox(testPatchV)
                patchGTV = takePointSetSlice(GT0,patchBB)
                
                # If no GT in the window, skip this patch (fake surface)
                if patchGTV.shape[0]<testPatchV.shape[0]:
                    continue


                vOldInd_list.append(vOldInd)
                fOldInd_list.append(fOldInd)

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                minusPadding3 = padding3-1
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                testPatchF = np.concatenate((testPatchF,minusPadding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                testPatchF = testPatchF[newToOld]


                oldToNew = np.array(inv_perm(newToOld))


                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)

                # new_to_old_permutations_list.append(newToOld)
                new_to_old_permutations_list.append(oldToNew)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)
                        # fAdjs = []
                        # f_adj = np.expand_dims(testPatchAdj, axis=0)
                        # fAdjs.append(f_adj)

                v_faces = getVerticesFaces(testPatchF,25,testPatchV.shape[0])

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                v_pos = np.expand_dims(testPatchV,axis=0)
                faces = np.expand_dims(testPatchF, axis=0)
                gtv_pos = np.expand_dims(patchGTV,axis=0)
                v_faces = np.expand_dims(v_faces,axis=0)

                v_list.append(v_pos)
                gtv_list.append(gtv_pos)
                n_list.append(f_normals)
                adj_list.append(fAdjs)
                faces_list.append(faces)
                v_faces_list.append(v_faces)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            coo_adj = listToSparse(f_adj0, f_pos0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            padding6 =np.zeros((new_N-old_N,6))
            padding3 =np.zeros((new_N-old_N,3))
            minusPadding3 = padding3-1
            minusPadding3 = minusPadding3.astype(int)

            faces0 = np.concatenate((faces0,minusPadding3),axis=0)

            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)

            oldToNew = np.array(inv_perm(newToOld))

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            # new_to_old_permutations_list.append(newToOld) # Nothing to append here, faces are already correctly ordered
            new_to_old_permutations_list.append(oldToNew)
            fOldInd_list.append([])
            vOldInd_list.append([])
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            faces0 = faces0[newToOld]

            

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)


            # fadj = sparseToList(adjs[4],K_faces)
            # fadj = np.expand_dims(fadj, axis=0)
            # fAdjs.append(fadj)
            # fadj = sparseToList(adjs[5],K_faces)
            # fadj = np.expand_dims(fadj, axis=0)
            # fAdjs.append(fadj)

            v_faces = getVerticesFaces(faces0,25,V0.shape[0])

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            v_pos = np.expand_dims(V0,axis=0)
            gtv_pos = np.expand_dims(GT0,axis=0)
            faces = np.expand_dims(faces0, axis=0)
            v_faces = np.expand_dims(v_faces,axis=0)

            v_list.append(v_pos)
            gtv_list.append(gtv_pos)
            n_list.append(f_normals)
            adj_list.append(fAdjs)
            faces_list.append(faces)
            v_faces_list.append(v_faces)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return vOldInd_list, fOldInd_list, vNum, fNum, new_to_old_permutations_list, num_faces


    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMeshWithVerticesAndFaceAssignment(inputFilePath,filename, gtFilePath, gtfilename, v_list, gtv_list, faces_list, n_list, adj_list, v_faces_list, gtfaces_list, f_assign_list, mesh_count_list):
        ASSIGNMENT_NUM = 30

        patch_indices = []
        new_to_old_permutations_list = []
        num_faces = []
        vOldInd_list = []
        fOldInd_list = []

        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)

        fNum = faces0.shape[0]
        vNum = V0.shape[0]
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0)
        # f_pos0 = np.reshape(f_pos0,(-1,3))

        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)

        # Load GT
        GT0,_,_,gt_faces0,_ = load_mesh(gtFilePath, gtfilename, 0, False)


        # Normalize vertices
        # V0, GT0 = normalizePointSets(V0,GT0)

        faceAssignment = getFaceAssignment(V0, faces0, GT0, gt_faces0, ASSIGNMENT_NUM)

        print("min faceAssignment = "+str(np.amin(faceAssignment)))
        # Get patches if mesh is too big
        facesNum = faces0.shape[0]
        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        print("maxSize = "+str(maxSize))
        print("facesNum = "+str(facesNum))
        print("faceAssignment shape: "+str(faceAssignment.shape))
        if faceAssignment.shape[0] != facesNum:
            print("WARNING !! inconsistency. Aborting...")
            return

        if facesNum>maxSize:
            patchNum = 0
            while((np.any(faceCheck==0))and(patchNum<10)):
            # while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)
                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchFaceAssignment = faceAssignment[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue


                vOldInd_list.append(vOldInd)
                fOldInd_list.append(fOldInd)

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                minusPadding3 = padding3-1
                paddingAssignment = np.zeros((new_N-old_N,ASSIGNMENT_NUM),dtype=np.int32)
                paddingAssignment = paddingAssignment - 1
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                testPatchF = np.concatenate((testPatchF,minusPadding3),axis=0)
                patchFaceAssignment = np.concatenate((patchFaceAssignment, paddingAssignment),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                testPatchF = testPatchF[newToOld]
                patchFaceAssignment = patchFaceAssignment[newToOld]


                oldToNew = np.array(inv_perm(newToOld))


                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)

                # new_to_old_permutations_list.append(newToOld)
                new_to_old_permutations_list.append(oldToNew)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)
                        # fAdjs = []
                        # f_adj = np.expand_dims(testPatchAdj, axis=0)
                        # fAdjs.append(f_adj)

                v_faces = getVerticesFaces(testPatchF,25,testPatchV.shape[0])

                if patchFaceAssignment.shape[0] != f_normals.shape[0]:
                    print("WARNING !! patch inconsistency. Aborting...")
                    return
                if patchFaceAssignment.shape[0] != faces.shape[0]:
                    print("WARNING !! patch faces inconsistency. Aborting...")
                    return

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                v_pos = np.expand_dims(testPatchV,axis=0)
                faces = np.expand_dims(testPatchF, axis=0)
                gtv_pos = np.expand_dims(GT0,axis=0)
                v_faces = np.expand_dims(v_faces,axis=0)
                gtfaces = np.expand_dims(gt_faces0, axis=0)
                fAss = np.expand_dims(patchFaceAssignment, axis=0)

                v_list.append(v_pos)
                gtv_list.append(gtv_pos)
                n_list.append(f_normals)
                adj_list.append(fAdjs)
                faces_list.append(faces)
                v_faces_list.append(v_faces)
                gtfaces_list.append(gtfaces)
                f_assign_list.append(fAss)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            coo_adj = listToSparse(f_adj0, f_pos0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            padding6 =np.zeros((new_N-old_N,6))
            padding3 =np.zeros((new_N-old_N,3))
            minusPadding3 = padding3-1
            minusPadding3 = minusPadding3.astype(int)
            paddingAssignment = np.zeros((new_N-old_N,ASSIGNMENT_NUM),dtype=np.int32)
            paddingAssignment = paddingAssignment - 1
            faces0 = np.concatenate((faces0,minusPadding3),axis=0)

            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            faceAssignment = np.concatenate((faceAssignment, paddingAssignment),axis=0)
            oldToNew = np.array(inv_perm(newToOld))

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            # new_to_old_permutations_list.append(newToOld) # Nothing to append here, faces are already correctly ordered
            new_to_old_permutations_list.append(oldToNew)
            fOldInd_list.append([])
            vOldInd_list.append([])
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            faces0 = faces0[newToOld]
            faceAssignment = faceAssignment[newToOld]
            

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)


            # fadj = sparseToList(adjs[4],K_faces)
            # fadj = np.expand_dims(fadj, axis=0)
            # fAdjs.append(fadj)
            # fadj = sparseToList(adjs[5],K_faces)
            # fadj = np.expand_dims(fadj, axis=0)
            # fAdjs.append(fadj)

            v_faces = getVerticesFaces(faces0,25,V0.shape[0])

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            v_pos = np.expand_dims(V0,axis=0)
            gtv_pos = np.expand_dims(GT0,axis=0)
            faces = np.expand_dims(faces0, axis=0)
            v_faces = np.expand_dims(v_faces,axis=0)
            gtfaces = np.expand_dims(gt_faces0, axis=0)
            fAss = np.expand_dims(faceAssignment, axis=0)

            v_list.append(v_pos)
            gtv_list.append(gtv_pos)
            n_list.append(f_normals)
            adj_list.append(fAdjs)
            faces_list.append(faces)
            v_faces_list.append(v_faces)
            gtfaces_list.append(gtfaces)
            f_assign_list.append(fAss)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return vOldInd_list, fOldInd_list, vNum, fNum, new_to_old_permutations_list, num_faces


    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMeshTransposeNormals(inputFilePath,filename, gtFilePath, gtfilename, in_list, gt_list, adj_list, mesh_count_list):
        patch_indices = []
        new_to_old_permutations_list = []
        num_faces = []

        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
        print("faces0 shape: "+str(faces0.shape))
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0)

        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)
        # f_area0 = getTrianglesArea(V0,faces0)
        # f_area0 = np.reshape(f_area0, (-1,1))
        # f_normals0 = np.concatenate((f_normals0, f_area0), axis=1)

        # Load GT
        GT0,_,_,faces_gt,_ = load_mesh(gtFilePath, gtfilename, 0, False)
        GTf_normals0 = computeFacesNormals(GT0, faces_gt)

        transNormals = transposeNormals(V0, faces0, GT0, faces_gt, 0, 0, mode='varying_gaussian')

        # Get patches if mesh is too big
        facesNum = faces0.shape[0]

        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>maxSize:
            patchNum = 0
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = transNormals[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue
                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                patchGTFNormals = patchGTFNormals[newToOld]

                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)
                new_to_old_permutations_list.append(newToOld)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                #f_adj = np.expand_dims(testPatchAdj, axis=0)
                GTf_normals = np.expand_dims(patchGTFNormals, axis=0)

                in_list.append(f_normals)
                adj_list.append(fAdjs)
                gt_list.append(GTf_normals)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            print("f_adj0 shape: "+str(f_adj0.shape))
            print("f_pos0 shape: "+str(f_pos0.shape))
            coo_adj = listToSparse(f_adj0, f_pos0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            padding6 =np.zeros((new_N-old_N,6))
            padding3 =np.zeros((new_N-old_N,3))
            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            GTf_normals0 = np.concatenate((transNormals, padding3),axis=0)

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            new_to_old_permutations_list.append(newToOld) # Nothing to append here, faces are already correctly ordered
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            GTf_normals0 = GTf_normals0[newToOld]

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            #f_adj = np.expand_dims(f_adj0, axis=0)
            GTf_normals = np.expand_dims(GTf_normals0, axis=0)

            in_list.append(f_normals)
            adj_list.append(fAdjs)
            gt_list.append(GTf_normals)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return num_faces, patch_indices, new_to_old_permutations_list

    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMeshWGTDispAndNormals(inputFilePath,filename, gtFilePath, gtfilename, in_list, gt_list, adj_list, gt_disp_list, faces_list, mesh_count_list):
        patch_indices = []
        new_to_old_permutations_list = []
        num_faces = []

        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
        print("faces0 shape: "+str(faces0.shape))
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0, normalize=False)

        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)
        # f_area0 = getTrianglesArea(V0,faces0)
        # f_area0 = np.reshape(f_area0, (-1,1))
        # f_normals0 = np.concatenate((f_normals0, f_area0), axis=1)

        # Load GT
        GT0,_,_,faces_gt,_ = load_mesh(gtFilePath, gtfilename, 0, False)
        # GTf_normals0 = computeFacesNormals(GT0, faces0)

        newPoints, fDisp, lastNormals = stickMesh(V0, faces0, GT0, faces_gt, 8, 80)


        # Get patches if mesh is too big
        facesNum = faces0.shape[0]

        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>maxSize:
            patchNum = 0
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = lastNormals[fOldInd]
                patchGTFDisp = fDisp[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue
                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                minusPadding3 = padding3-1
                minusPadding3 = minusPadding3.astype(int)
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                patchGTFDisp = np.concatenate((patchGTFDisp, padding3),axis=0)
                testPatchF = np.concatenate((testPatchF,minusPadding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                patchGTFNormals = patchGTFNormals[newToOld]
                patchGTFDisp = patchGTFDisp[newToOld]
                patchF = testPatchF[newToOld]

                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)
                new_to_old_permutations_list.append(newToOld)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                GTf_normals = np.expand_dims(patchGTFNormals, axis=0)
                GTf_disp = np.expand_dims(patchGTFDisp, axis=0)
                faces_new = np.expand_dims(patchF, axis=0)

                in_list.append(f_normals)
                adj_list.append(fAdjs)
                gt_list.append(GTf_normals)
                gt_disp_list.append(GTf_disp)
                faces_list.append(faces_new)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            print("f_adj0 shape: "+str(f_adj0.shape))
            print("f_pos0 shape: "+str(f_pos0.shape))
            coo_adj = listToSparse(f_adj0, f_pos0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            padding6 =np.zeros((new_N-old_N,6))
            padding3 =np.zeros((new_N-old_N,3))
            minusPadding3 = padding3-1
            minusPadding3 = minusPadding3.astype(int)
            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            GTf_normals0 = np.concatenate((lastNormals, padding3),axis=0)
            GTf_disp0 = np.concatenate((fDisp, padding3),axis=0)
            faces0 = np.concatenate((faces0,minusPadding3),axis=0)

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            new_to_old_permutations_list.append(newToOld) # Nothing to append here, faces are already correctly ordered
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            GTf_normals0 = GTf_normals0[newToOld]
            GTf_disp0 = GTf_disp0[newToOld]
            faces_new = faces0[newToOld]

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            #f_adj = np.expand_dims(f_adj0, axis=0)
            GTf_normals = np.expand_dims(GTf_normals0, axis=0)
            GTf_disp = np.expand_dims(GTf_disp0, axis=0)
            faces_new = np.expand_dims(faces_new,axis=0)

            in_list.append(f_normals)
            adj_list.append(fAdjs)
            gt_list.append(GTf_normals)
            gt_disp_list.append(GTf_disp)
            faces_list.append(faces_new)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return num_faces, patch_indices, new_to_old_permutations_list


    # Train network
    if running_mode == 0:

        gtnameoffset = 10
        f_normals_list = []
        f_adj_list = []
        GTfn_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_GTfn_list = []


        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/original/"


        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/original/"
        
        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid/"
        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/train_cleaned/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid_cleaned/"
        gtFilePath = "/morpheo-nas2/marmando/MPI-FAUST/training/rescaled/"

        #print("training_meshes_num 0 " + str(training_meshes_num))
        if pickleLoad:
            # Training
            with open(binDumpPath+'f_normals_list', 'rb') as fp:
                f_normals_list = pickle.load(fp)
            with open(binDumpPath+'GTfn_list', 'rb') as fp:
                GTfn_list = pickle.load(fp)
            with open(binDumpPath+'f_adj_list', 'rb') as fp:
                f_adj_list = pickle.load(fp)
            # Validation
            with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
                valid_f_normals_list = pickle.load(fp)
            with open(binDumpPath+'valid_GTfn_list', 'rb') as fp:
                valid_GTfn_list = pickle.load(fp)
            with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
                valid_f_adj_list = pickle.load(fp)


        else:
            # Training set
            for filename in os.listdir(inputFilePath):
                #print("training_meshes_num start_iter " + str(training_meshes_num))
                if training_meshes_num[0]>1000:
                    break
                #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
                #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
                if (filename.endswith(".obj")):


                    #For FAUST
                    fileNumStr = filename[5:8]
                    # if int(fileNumStr)>2:
                    #     continue
                    gtfilename = 'M'+fileNumStr+'/Ground_Truth/000.obj'
                    print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")
                    # gtfilename = filename[:-gtnameoffset]+".obj"
                    addMeshTransposeNormals(inputFilePath, filename, gtFilePath, gtfilename, f_normals_list, GTfn_list, f_adj_list, training_meshes_num)

            # Validation set
            for filename in os.listdir(validFilePath):
                if (filename.endswith(".obj")):
                    # gtfilename = filename[:-gtnameoffset]+".obj"
                    #For FAUST
                    fileNumStr = filename[5:8]
                    gtfilename = 'M'+fileNumStr+'/Ground_Truth/000.obj'
                    addMeshTransposeNormals(validFilePath, filename, gtFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_meshes_num)
                    

                #print("training_meshes_num end_iter " + str(training_meshes_num))
            if pickleSave:
                # Training
                with open(binDumpPath+'f_normals_list', 'wb') as fp:
                    pickle.dump(f_normals_list, fp)
                with open(binDumpPath+'GTfn_list', 'wb') as fp:
                    pickle.dump(GTfn_list, fp)
                with open(binDumpPath+'f_adj_list', 'wb') as fp:
                    pickle.dump(f_adj_list, fp)
                # Validation
                with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
                    pickle.dump(valid_f_normals_list, fp)
                with open(binDumpPath+'valid_GTfn_list', 'wb') as fp:
                    pickle.dump(valid_GTfn_list, fp)
                with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
                    pickle.dump(valid_f_adj_list, fp)



        trainNet(f_normals_list, GTfn_list, f_adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list)

    # Simple inference, no GT mesh involved
    elif running_mode == 1:
        
        maxSize = 100000
        patchSize = 100000

        # noisyFolder = "/morpheo-nas2/vincent/DTU_Robot_Image_Dataset/Surface/furu/"


        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/test_bits/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/test/"
        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/Kinovis/"
        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/tola/test_bits/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid/"
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid_cleaned/"
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/results/1stStep/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/kickVH/results/1stStep/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/realData/vincentSlip/decimVH/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/realData/vincentSlip/results/1stStep/"
        # noisyFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/registrations/"

        # Get GT mesh
        for noisyFile in os.listdir(noisyFolder):

            
            if (not noisyFile.endswith(".obj")):
                continue
            print("noisyFile: "+noisyFile)
            # if (not noisyFile.startswith("chinese")):
            #     continue
            mesh_count = [0]


            denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"


            noisyFilesList = [noisyFile]
            denoizedFilesList = [denoizedFile]

            for fileNum in range(len(denoizedFilesList)):
                
                randToDel = np.random.rand(1)
                if randToDel>0.1:
                    continue
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                # noisyFileWInferredColor = noisyFile[:-4]+"_inferred_normals.obj"

                noisyFileWInferredColor0 = noisyFile[:-4]+"_fine_normals.obj"
                noisyFileWInferredColor1 = noisyFile[:-4]+"_mid_normals.obj"
                noisyFileWInferredColor2 = noisyFile[:-4]+"_coarse_normals.obj"

                noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
                denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"

                if not os.path.isfile(RESULTS_PATH+denoizedFile):
                # if True:

                    f_normals_list = []
                    GTfn_list = []
                    f_adj_list = []
                    faces_list = []
                    v_faces_list = []
                    v_list = []
                    gtv_list = []


                    V0,_,_, faces_noisy, _ = load_mesh(noisyFolder, noisyFile, 0, False)
                    f_normals0 = computeFacesNormals(V0, faces_noisy)

                    print("Adding mesh "+noisyFile+"...")
                    t0 = time.clock()
                    # faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, noisyFolder, noisyFile, faces_list, f_normals_list, GTfn_list, f_adj_list, mesh_count)
                    vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list, = addMeshWithVertices(noisyFolder, noisyFile, noisyFolder, noisyFile, v_list, gtv_list, faces_list, f_normals_list, f_adj_list, v_faces_list, mesh_count)
                    print("mesh added ("+str(1000*(time.clock()-t0))+"ms)")
                    # Now recover vertices positions and create Edge maps

                    V0 = np.expand_dims(V0, axis=0)

                    # print("WARNING!!!!! Hardcoded a change in faces adjacency")
                    # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
                    

                    faces_noisy = np.array(faces_noisy).astype(np.int32)
                    faces = np.expand_dims(faces_noisy,axis=0)

                    # v_faces = getVerticesFaces(np.squeeze(faces_list[0]), 15, V0.shape[1])
                    # v_faces = np.expand_dims(v_faces,axis=0)

                    print("Inference ...")
                    t0 = time.clock()
                    #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    # upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, faces_num, patch_indices, permutations,facesNum)
                    # upV0, upN0, upN1, upN2 = inferNet(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
                    upV0, upN0 = inferNet6D(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
                    print("Inference complete ("+str(1000*(time.clock()-t0))+"ms)")

                    write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)

                    angColor0 = (upN0+1)/2
                    # angColor1 = (-upN1+1)/2
                    # angColor2 = (-upN2+1)/2

                    # f_normals0 = np.squeeze(f_normals_list[0])
                    # print("f_normals0 shape: "+str(f_normals0.shape))
                    # f_normals0 = f_normals0[:,:3]
                    # print("f_normals0 shape: "+str(f_normals0.shape))
                    angColorNoisy = (f_normals0+1)/2
                    
                    print("faces_noisy shape: "+str(faces_noisy.shape))

                    print("angColor0 shape: "+str(angColor0.shape))
                    # print("angColor1 shape: "+str(angColor1.shape))
                    # print("angColor2 shape: "+str(angColor2.shape))
                    print("V0 shape: "+str(V0.shape))
                    # newV, newF = getColoredMesh(upV0, faces_gt, angColor)
                    newVn0, newFn0 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor0)
                    # newVn1, newFn1 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor1)
                    # newVn2, newFn2 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor2)
                    

                    # write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
                    write_mesh(newVn0, newFn0, RESULTS_PATH+noisyFileWInferredColor0)
                    # write_mesh(newVn1, newFn1, RESULTS_PATH+noisyFileWInferredColor1)
                    # write_mesh(newVn2, newFn2, RESULTS_PATH+noisyFileWInferredColor2)

                    print("angColorNoisy shape: "+str(angColorNoisy.shape))
                    newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_noisy, angColorNoisy)
                    write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFileWColor)

    # master branch inference (old school, w/o multi-scale vertex update)
    elif running_mode == 12:
        
        maxSize = 100000
        patchSize = 100000

        # noisyFolder = "/morpheo-nas2/vincent/DTU_Robot_Image_Dataset/Surface/furu/"


        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/test_bits/"
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/test/"
        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/Kinovis/"
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid/"
        noisyFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/registrations/"


        # Get GT mesh
        for noisyFile in os.listdir(noisyFolder):


            if (not noisyFile.endswith(".obj")):
                continue
            mesh_count = [0]


            denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"


            noisyFilesList = [noisyFile]
            denoizedFilesList = [denoizedFile]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                noisyFileWInferredColor = noisyFile[:-4]+"_inferred_normals.obj"
                noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
                denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"

                if not os.path.isfile(RESULTS_PATH+denoizedFile):
                    

                    f_normals_list = []
                    GTfn_list = []
                    f_adj_list = []

                    V0,_,_, faces_noisy, _ = load_mesh(noisyFolder, noisyFile, 0, False)
                    f_normals0 = computeFacesNormals(V0, faces_noisy)

                    print("Adding mesh "+noisyFile+"...")
                    t0 = time.clock()
                    faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, noisyFolder, noisyFile, f_normals_list, GTfn_list, f_adj_list, mesh_count)
                    print("mesh added ("+str(1000*(time.clock()-t0))+"ms)")
                    # Now recover vertices positions and create Edge maps

                    

                    facesNum = faces_noisy.shape[0]
                    V0 = np.expand_dims(V0, axis=0)

                    _, edge_map, v_e_map = getFacesAdj2(faces_noisy)
                    f_adj = getFacesLargeAdj(faces_noisy,K_faces)
                    # print("WARNING!!!!! Hardcoded a change in faces adjacency")
                    # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
                    

                    faces_noisy = np.array(faces_noisy).astype(np.int32)
                    faces = np.expand_dims(faces_noisy,axis=0)
                    edge_map = np.expand_dims(edge_map, axis=0)
                    v_e_map = np.expand_dims(v_e_map, axis=0)

                    print("Inference ...")
                    t0 = time.clock()
                    #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    upV0, upN0 = inferNetOld(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    print("Inference complete ("+str(1000*(time.clock()-t0))+"ms)")

                    write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)

                    angColor = (upN0+1)/2

                    angColorNoisy = (f_normals0+1)/2
                    
                    # newV, newF = getColoredMesh(upV0, faces_gt, angColor)
                    newVn, newFn = getColoredMesh(np.squeeze(V0), faces_noisy, angColor)
                    newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_noisy, angColorNoisy)

                    # write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
                    write_mesh(newVn, newFn, RESULTS_PATH+noisyFileWInferredColor)
                    write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFileWColor)

    # Inference: Denoise set, save meshes (colored with heatmap), compute metrics
    elif running_mode == 2:
        
        maxSize = 100000
        patchSize = 100000


        # Take the opportunity to generate array of metrics on reconstructions
        nameArray = []      # String array, to now which row is what
        resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                            # [Max distance, Mean distance, Mean angle, std angle, face num]

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results.csv"
        
        

        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("aarmadillo")):
                continue
            mesh_count = [0]

            # Get all 3 noisy meshes
            # noisyFile0 = gtFileName[:-4]+"_noisy.obj"
            # denoizedFile0 = gtFileName[:-4]+"_denoized.obj"
            
            # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
            

            noisyFile0 = gtFileName[:-4]+"_n1.obj"
            noisyFile1 = gtFileName[:-4]+"_n2.obj"
            noisyFile2 = gtFileName[:-4]+"_n3.obj"

            denoizedFile0 = gtFileName[:-4]+"_denoized_gray_1.obj"
            denoizedFile1 = gtFileName[:-4]+"_denoized_gray_2.obj"
            denoizedFile2 = gtFileName[:-4]+"_denoized_gray_3.obj"

            # if (os.path.isfile(RESULTS_PATH+denoizedFile0)) and (os.path.isfile(RESULTS_PATH+denoizedFile1)) and (os.path.isfile(RESULTS_PATH+denoizedFile2)):
            #     continue

            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            GTf_normals = computeFacesNormals(GT, faces_gt)

            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj2(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            

            # noisyFilesList = [noisyFile0]
            # denoizedFilesList = [denoizedFile0]

            noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                denoizedHeatmap = denoizedFile[:-4]+"_H.obj"
                noisyFile = noisyFilesList[fileNum]
                
                if not os.path.isfile(RESULTS_PATH+denoizedFile):
                    

                    f_normals_list = []
                    GTfn_list = []
                    f_adj_list = []

                    print("Adding mesh "+noisyFile+"...")
                    t0 = time.clock()
                    faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, gtFolder, gtFileName, f_normals_list, GTfn_list, f_adj_list, mesh_count)
                    print("mesh added ("+str(1000*(time.clock()-t0))+"ms)")
                    # Now recover vertices positions and create Edge maps
                    V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)
                    V0 = np.expand_dims(V0, axis=0)

                    print("Inference ...")
                    t0 = time.clock()
                    #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    print("Inference complete ("+str(1000*(time.clock()-t0))+"ms)")


                    print("computing Hausdorff "+str(fileNum+1)+"...")
                    t0 = time.clock()
                    haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                    print("Hausdorff complete ("+str(1000*(time.clock()-t0))+"ms)")
                    print("computing Angular diff "+str(fileNum+1)+"...")
                    t0 = time.clock()
                    angDistVec = angularDiffVec(upN0, GTf_normals)
                    angDist0, angStd0 = angularDiff(upN0, GTf_normals)
                    print("Angular diff complete ("+str(1000*(time.clock()-t0))+"ms)")
                    print("max angle: "+str(np.amax(angDistVec)))


                     # --- heatmap ---
                    angColor = angDistVec / empiricMax

                    angColor = 1 - angColor
                    angColor = np.maximum(angColor, np.zeros_like(angColor))

                    print("getting colormap "+str(fileNum+1)+"...")
                    t0 = time.clock()
                    colormap = getHeatMapColor(1-angColor)
                    print("colormap shape: "+str(colormap.shape))
                    newV, newF = getColoredMesh(upV0, faces_gt, colormap)
                    print("colormap complete ("+str(1000*(time.clock()-t0))+"ms)")
                    #newV, newF = getHeatMapMesh(upV0, faces_gt, angColor)
                    print("writing mesh...")
                    t0 = time.clock()
                    write_mesh(newV, newF, RESULTS_PATH+denoizedHeatmap)
                    print("mesh written ("+str(1000*(time.clock()-t0))+"ms)")
                    write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)

                    # Fill arrays
                    nameArray.append(denoizedFile)
                    resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum])

            outputFile = open(csv_filename,'a')
            nameArray = np.array(nameArray)
            resultsArray = np.array(resultsArray,dtype=np.float32)

            tempArray = resultsArray.flatten()
            resStr = ["%.7f" % number for number in tempArray]
            resStr = np.reshape(resStr,resultsArray.shape)

            nameArray = np.expand_dims(nameArray, axis=-1)

            finalArray = np.concatenate((nameArray,resStr),axis=1)
            for row in range(finalArray.shape[0]):
                for col in range(finalArray.shape[1]):
                    outputFile.write(finalArray[row,col])
                    outputFile.write(' ')
                outputFile.write('\n')

            outputFile.close()

    elif running_mode == 3:
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"
        gtFileName = "block.obj"

        GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
        GTf_normals = computeFacesNormals(GT, faces_gt)
        facesNum = faces_gt.shape[0]
        vertNum = GT.shape[0]
        GT_pos = getTrianglesBarycenter(GT,faces_gt)

        normal_pos = np.concatenate((GTf_normals,GT_pos),axis=-1)

        myRot = rand_rotation_matrix()
        tensMyRot = np.reshape(myRot,(1,3,3))
        tensMyRot2 = np.tile(tensMyRot,(facesNum,1,1))

        normal_pos = np.reshape(normal_pos,(-1,2,3))
        normal_pos = np.transpose(normal_pos,(0,2,1))

        rot_normal_pos = np.matmul(tensMyRot2,normal_pos)

        rot_normal_pos = np.transpose(rot_normal_pos,(0,2,1))
        rot_normal_pos = np.reshape(rot_normal_pos,(-1,6))

        newN = rot_normal_pos[:,:3]
        # newN = np.slice(rot_normal_pos,(0,0),(-1,3))

        tensMyRotV = np.tile(tensMyRot,(vertNum,1,1))

        rot_vert = np.expand_dims(GT,axis=-1)
        rot_vert = np.matmul(tensMyRotV,rot_vert)

        rot_vert = np.reshape(rot_vert,(-1,3))

        angColor = (newN+1)/2

        angColor0 = (GTf_normals+1)/2

        newV, newF = getColoredMesh(rot_vert, faces_gt, angColor)

        VO, FO = getColoredMesh(GT,faces_gt,angColor0)

        write_mesh(newV,newF,'/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/rotated_colored.obj')
        write_mesh(VO,FO,'/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/ori_colored.obj')

    # Train network with accuracy loss on point sets (rather than normals angular loss)
    elif running_mode == 4:


        gtnameoffset = 7
        f_normals_list = []
        f_adj_list = []
        v_pos_list = []
        gtv_pos_list = []
        faces_list = []
        v_faces_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_v_pos_list = []
        valid_gtv_pos_list = []
        valid_faces_list = []
        valid_v_faces_list = []

        f_normals_list_temp = []
        f_adj_list_temp = []
        v_pos_list_temp = []
        gtv_pos_list_temp = []
        faces_list_temp = []
        v_faces_list_temp = []

        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train2/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
        # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"


        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/tola/train/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/tola/valid/"
        # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/train_cleaned/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid_cleaned/"
        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid/"

        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Ground_Truth/"

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"
        
        #print("training_meshes_num 0 " + str(training_meshes_num))
        if pickleLoad:

            # Training
            pickleNum=0
            while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
                with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
                    f_normals_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
                    f_adj_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'v_pos_list'+str(pickleNum), 'rb') as fp:
                    v_pos_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'rb') as fp:
                    gtv_pos_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'faces_list'+str(pickleNum), 'rb') as fp:
                    faces_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'v_faces_list'+str(pickleNum), 'rb') as fp:
                    v_faces_list_temp = pickle.load(fp, encoding='latin1')

                if pickleNum==0:
                    f_normals_list = f_normals_list_temp
                    f_adj_list = f_adj_list_temp
                    v_pos_list = v_pos_list_temp
                    gtv_pos_list = gtv_pos_list_temp
                    faces_list = faces_list_temp
                    v_faces_list = v_faces_list_temp
                else:

                    f_normals_list+=f_normals_list_temp
                    f_adj_list+=f_adj_list_temp
                    v_pos_list+=v_pos_list_temp
                    gtv_pos_list+=gtv_pos_list_temp
                    faces_list+=faces_list_temp
                    v_faces_list+=v_faces_list_temp


                print("loaded training pickle "+str(pickleNum))
                pickleNum+=1


            # Validation
            with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
                valid_f_normals_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
                valid_f_adj_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_v_pos_list', 'rb') as fp:
                valid_v_pos_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_gtv_pos_list', 'rb') as fp:
                valid_gtv_pos_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_faces_list', 'rb') as fp:
                valid_faces_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_v_faces_list', 'rb') as fp:
                valid_v_faces_list = pickle.load(fp, encoding='latin1')


        else:

            pickleNum=0
            # Training set
            for filename in os.listdir(inputFilePath):
                #print("training_meshes_num start_iter " + str(training_meshes_num))
                if training_meshes_num[0]>1000:
                    break
                #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
                #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
                if (filename.endswith(".obj")):
                    if filename.startswith("a") or filename.startswith("b"):
                        continue
                    print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")

                    # # For DTU
                    # fileNumStr = filename[4:7]
                    # gtfilename = 'stl'+fileNumStr+'_total.obj'

                    # # For CNR dataset
                    # gtfilename = filename[:-gtnameoffset]+".obj"

                    #For FAUST
                    fileNumStr = filename[5:8]
                    # if int(fileNumStr)>2:
                    #     continue
                    gtfilename = 'gt'+fileNumStr+'.obj'

                    addMeshWithVertices(inputFilePath, filename, gtFilePath, gtfilename, v_pos_list_temp, gtv_pos_list_temp, faces_list_temp, f_normals_list_temp, f_adj_list_temp, v_faces_list_temp, training_meshes_num)

                    # Save batches of meshes/patches (for training only)
                    if training_meshes_num[0]>30:
                        if pickleSave:
                            # Training
                            with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(f_normals_list_temp, fp)
                            with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(f_adj_list_temp, fp)
                            with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(v_pos_list_temp, fp)
                            with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(gtv_pos_list_temp, fp)
                            with open(binDumpPath+'faces_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(faces_list_temp, fp)
                            with open(binDumpPath+'v_faces_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(v_faces_list_temp, fp)
                        if pickleNum==0:
                            f_normals_list = f_normals_list_temp
                            f_adj_list = f_adj_list_temp
                            v_pos_list = v_pos_list_temp
                            gtv_pos_list = gtv_pos_list_temp
                            faces_list = faces_list_temp
                            v_faces_list = v_faces_list_temp
                        else:
                            f_normals_list+=f_normals_list_temp
                            f_adj_list+=f_adj_list_temp
                            v_pos_list+=v_pos_list_temp
                            gtv_pos_list+=gtv_pos_list_temp
                            faces_list+=faces_list_temp
                            v_faces_list+=v_faces_list_temp

                        pickleNum+=1
                        f_normals_list_temp = []
                        f_adj_list_temp = []
                        v_pos_list_temp = []
                        gtv_pos_list_temp = []
                        faces_list_temp = []
                        v_faces_list_temp = []
                        training_meshes_num[0] = 0

            if (pickleSave) and training_meshes_num[0]>0:
                # Training
                with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(f_normals_list_temp, fp)
                with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(f_adj_list_temp, fp)
                with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(v_pos_list_temp, fp)
                with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(gtv_pos_list_temp, fp)
                with open(binDumpPath+'faces_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(faces_list_temp, fp)
                with open(binDumpPath+'v_faces_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(v_faces_list_temp, fp)

            if pickleNum==0:
                f_normals_list = f_normals_list_temp
                f_adj_list = f_adj_list_temp
                v_pos_list = v_pos_list_temp
                gtv_pos_list = gtv_pos_list_temp
                faces_list = faces_list_temp
                v_faces_list = v_faces_list_temp
            else:
                f_normals_list+=f_normals_list_temp
                f_adj_list+=f_adj_list_temp
                v_pos_list+=v_pos_list_temp
                gtv_pos_list+=gtv_pos_list_temp
                faces_list+=faces_list_temp
                v_faces_list+=v_faces_list_temp

            # Validation set
            for filename in os.listdir(validFilePath):
                if (filename.endswith(".obj")):
                    if valid_meshes_num[0]>200:
                        break
                    # # For DTU
                    # fileNumStr = filename[4:7]
                    # gtfilename = 'stl'+fileNumStr+'_total.obj'

                    # # For CNR dataset
                    # gtfilename = filename[:-gtnameoffset]+".obj"

                    #For FAUST
                    fileNumStr = filename[5:8]
                    gtfilename = 'gt'+fileNumStr+'.obj'

                    addMeshWithVertices(validFilePath, filename, gtFilePath, gtfilename, valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list, valid_meshes_num)
                    

                #print("training_meshes_num end_iter " + str(training_meshes_num))
            
            if pickleSave:
                # Validation
                with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
                    pickle.dump(valid_f_normals_list, fp)
                with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
                    pickle.dump(valid_f_adj_list, fp)
                with open(binDumpPath+'valid_v_pos_list', 'wb') as fp:
                    pickle.dump(valid_v_pos_list, fp)
                with open(binDumpPath+'valid_gtv_pos_list', 'wb') as fp:
                    pickle.dump(valid_gtv_pos_list, fp)
                with open(binDumpPath+'valid_faces_list', 'wb') as fp:
                    pickle.dump(valid_faces_list, fp)
                with open(binDumpPath+'valid_v_faces_list', 'wb') as fp:
                    pickle.dump(valid_v_faces_list, fp)

        trainAccuracyNet(v_pos_list, gtv_pos_list, faces_list, f_normals_list, f_adj_list, v_faces_list, valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list)
        # trainAccuracyNet(valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list, valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list)

    # Test on faces clustering
    elif running_mode == 5:


        inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"
        
        # Training set
        for filename in os.listdir(inputFilePath):
            if (filename.endswith(".obj")):

                V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
                f_normals0 = computeFacesNormals(V0, faces0)
                f_adj0 = getFacesLargeAdj(faces0,K_faces)
                fpos0 = getTrianglesBarycenter(V0, faces0)
                curv_stat0 = computeCurvature(fpos0, f_normals0,f_adj0)

                if not 'curv_stat' in locals():
                    curv_stat = curv_stat0
                else:
                    curv_stat = np.concatenate((curv_stat,curv_stat0),axis=0)
        
        # inputFileName = "bunny_hi.obj"
        # inputFileName = "sharp_sphere.obj"
        # #inputFileName = "armadillo.obj"
        
        # V0,_,_, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)

        # f_normals0 = computeFacesNormals(V0, faces0)
        # _, edge_map0, v_e_map0 = getFacesAdj2(faces0)
        # f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # fpos0 = getTrianglesBarycenter(V0, faces0)

        # curv_stat = computeCurvature(fpos0, f_normals0,f_adj0)

        centroids, closest = customKMeans(curv_stat, 8)

        binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/class_centroids/"

        with open(binDumpPath+'centroids_8', 'wb') as fp:
            pickle.dump(centroids, fp)
        
        cc1 = 0.7
        cc2 = 0.4
        colors = np.array([[cc1,1,1],[1,cc1,1],[1,1,cc1],[cc1,cc1,1],[cc1,1,cc1],[1,cc1,cc1],[1,cc1,cc2],[1,cc2,cc1]])

        for clu in range(8):
            clu_color = colors[clu]
            v_clu = V0
            clu_color = np.tile(clu_color,[v_clu.shape[0],1])
            v_clu = np.concatenate((v_clu,clu_color),axis=1)
            f_clu = faces0[closest==clu]

            write_mesh(v_clu,f_clu, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/clustertri" + str(clu) + ".obj")

    # Test on faces clustering
    elif running_mode == 6:

        cc1 = 0.7
        cc2 = 0.4
        colors = np.array([[cc1,1,1],[1,cc1,1],[1,1,cc1],[cc1,cc1,1],[cc1,1,cc1],[1,cc1,cc1],[1,cc1,cc2],[1,cc2,cc1]])

        inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/class_centroids/"

        with open(binDumpPath+'centroids_8', 'rb') as fp:
            centroids = pickle.load(fp)
        
        # Training set
        for filename in os.listdir(inputFilePath):
            if (filename.endswith(".obj")):

                V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
                f_normals0 = computeFacesNormals(V0, faces0)
                f_adj0 = getFacesLargeAdj(faces0,K_faces)
                fpos0 = getTrianglesBarycenter(V0, faces0)
                curv_stat0 = computeCurvature(fpos0, f_normals0,f_adj0)

                closest0 = closest_centroid(curv_stat0, centroids)

                for clu in range(8):
                    clu_color = colors[clu]
                    v_clu = V0
                    clu_color = np.tile(clu_color,[v_clu.shape[0],1])
                    v_clu = np.concatenate((v_clu,clu_color),axis=1)
                    f_clu = faces0[closest0==clu]

                    write_mesh(v_clu,f_clu, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/" + filename + str(clu) + ".obj")

        # write_xyz(fpos0[closest==0], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster0.xyz")
        # write_xyz(fpos0[closest==1], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster1.xyz")
        # write_xyz(fpos0[closest==2], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster2.xyz")
        # write_xyz(fpos0[closest==3], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster3.xyz")
        # write_xyz(fpos0[closest==4], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster4.xyz")
        # write_xyz(fpos0[closest==5], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster5.xyz")
        # write_xyz(fpos0[closest==6], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster6.xyz")
        # write_xyz(fpos0[closest==7], "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/cluster7.xyz")

    elif running_mode == 7:

        # Take the opportunity to generate array of metrics on reconstructions
        nameArray = []      # String array, to now which row is what
        resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                            # [Max distance, Mean distance, Mean angle, std angle, face num]

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"


        # results file name
        csv_filename = RESULTS_PATH+"results.csv"
        

        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []

            if (not gtFileName.endswith(".obj")):
                continue

            noisyFile0 = gtFileName[:-4]+"_noisy.obj"
            

            denoizedFile0 = gtFileName[:-4]+"_denoized.obj"

            if (os.path.isfile(RESULTS_PATH+denoizedFile0)):
                continue

            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            GTf_normals = computeFacesNormals(GT, faces_gt)

            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj2(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            #edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            if not os.path.isfile(RESULTS_PATH+denoizedFile0):

                V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile0, 0, False)
                f_normals0 = computeFacesNormals(V0, faces_gt)

                f_pos0 = getTrianglesBarycenter(V0, faces_gt)
                f_pos0 = np.reshape(f_pos0,(-1,3))
                f_normals_pos0 = np.concatenate((f_normals0, f_pos0), axis=1)

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(f_adj, f_pos0)
                adjs, newToOld = coarsen(coo_adj,4)

                oldToNew = np.array(inv_perm(newToOld))
                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                old_N = facesNum
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                f_normals_pos0 = np.concatenate((f_normals_pos0,padding6),axis=0)
                GTf_normals0 = np.concatenate((GTf_normals, padding3),axis=0)
                faces0 = np.concatenate((faces_gt, padding3),axis=0)
                # Reorder nodes
                f_normals_pos0 = f_normals_pos0[newToOld]
                GTf_normals0 = GTf_normals0[newToOld]
                faces0 = faces0[newToOld]

                # Change adj format
                fAdjs = []
                for lvl in range(3):
                    fadj = sparseToList(adjs[2*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)

                #update edge_map
                emap_f = edge_map[:,2:]
                emap_v = edge_map[:,:2]
                emap_f = emap_f.flatten()
                emap_f = oldToNew[emap_f]
                emap_f = np.reshape(emap_f, (-1,2))
                edge_map0 = np.concatenate((emap_v,emap_f),axis=-1)
                edge_map0 = np.expand_dims(edge_map0, axis=0)

                V0 = np.expand_dims(V0, axis=0)
                faces0 = np.expand_dims(faces0,axis=0)
                f_normals_pos0 = np.expand_dims(f_normals_pos0, axis=0)

                print("running n1...")
                upV0, upN0 = inferNet(V0, f_normals_pos0, fAdjs, edge_map0, v_e_map)
                print("computing Hausdorff 1...")
                haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                angDist0, angStd0 = angularDiff(upN0, GTf_normals0)
                write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile0)

                # Fill arrays
                nameArray.append(denoizedFile0)
                resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum])

            # print("Hausdorff distances: ("+str(haus_dist0)+", "+str(haus_dist1)+", "+str(haus_dist2)+")")
            # print("Average angular differences: ("+str(angDist0)+", "+str(angDist1)+", "+str(angDist2)+")")

            outputFile = open(csv_filename,'a')
            nameArray = np.array(nameArray)
            resultsArray = np.array(resultsArray,dtype=np.float32)

            tempArray = resultsArray.flatten()
            resStr = ["%.7f" % number for number in tempArray]
            resStr = np.reshape(resStr,resultsArray.shape)

            nameArray = np.expand_dims(nameArray, axis=-1)

            finalArray = np.concatenate((nameArray,resStr),axis=1)
            for row in range(finalArray.shape[0]):
                for col in range(finalArray.shape[1]):
                    outputFile.write(finalArray[row,col])
                    outputFile.write(' ')
                outputFile.write('\n')

            outputFile.close()

    # Compute metrics and heatmaps on denoised meshes + GT
    elif running_mode == 8:
        # Take the opportunity to generate array of metrics on reconstructions
        nameArray = []      # String array, to now which row is what
        resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                            # [Max distance, Mean distance, Mean angle, std angle, face num]

        #gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results_heat.csv"
        

        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")):
                continue

            # denoizedFile0 = gtFileName[:-4]+"_n1-dtree3.obj"
            # denoizedFile1 = gtFileName[:-4]+"_n2-dtree3.obj"
            # denoizedFile2 = gtFileName[:-4]+"_n3-dtree3.obj"

            denoizedFile0 = gtFileName[:-4]+"_noisy-dtree3.obj"
            # denoizedFile0 = gtFileName[:-4]+"_denoized.obj"
            heatFile0 = gtFileName[:-4]+"_heatmap.obj"

            # denoizedFile0 = gtFileName[:-4]+"_denoized_gray_1.obj"
            # denoizedFile1 = gtFileName[:-4]+"_denoized_gray_2.obj"
            # denoizedFile2 = gtFileName[:-4]+"_denoized_gray_3.obj"

            denoizedFile0 = gtFileName[:-4]+"_n1_denoised_gray.obj"
            denoizedFile1 = gtFileName[:-4]+"_n2_denoised_gray.obj"
            denoizedFile2 = gtFileName[:-4]+"_n3_denoised_gray.obj"

            # heatFile0 = gtFileName[:-4]+"_dtree3_heatmap_1.obj"
            # heatFile1 = gtFileName[:-4]+"_dtree3_heatmap_2.obj"
            # heatFile2 = gtFileName[:-4]+"_dtree3_heatmap_3.obj"

            heatFile0 = gtFileName[:-4]+"_heatmap_1.obj"
            heatFile1 = gtFileName[:-4]+"_heatmap_2.obj"
            heatFile2 = gtFileName[:-4]+"_heatmap_3.obj"

            # if (os.path.isfile(RESULTS_PATH+heatFile0)) and (os.path.isfile(RESULTS_PATH+heatFile1)) and (os.path.isfile(RESULTS_PATH+heatFile2)):
            #     continue

            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            GTf_normals = computeFacesNormals(GT, faces_gt)

            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj2(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            #edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]
            heatMapFilesList = [heatFile0,heatFile1,heatFile2]

            # denoizedFilesList = [denoizedFile0]
            # heatMapFilesList = [heatFile0]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                heatFile = heatMapFilesList[fileNum]
                
                if not os.path.isfile(RESULTS_PATH+heatFile):
                    
                    V0,_,_, _, _ = load_mesh(RESULTS_PATH, denoizedFile, 0, False)
                    f_normals0 = computeFacesNormals(V0, faces_gt)

                    print("computing Hausdorff "+str(fileNum+1)+"...")
                    haus_dist0, avg_dist0 = oneSidedHausdorff(V0, GT)
                    angDistVec = angularDiffVec(f_normals0, GTf_normals)
                    angDist0, angStd0 = angularDiff(f_normals0, GTf_normals)
                    print("max angle: "+str(np.amax(angDistVec)))

                    # --- Test heatmap ---
                    angColor = angDistVec / empiricMax
                    angColor = 1 - angColor
                    angColor = np.maximum(angColor, np.zeros_like(angColor))

                    colormap = getHeatMapColor(1-angColor)
                    newV, newF = getColoredMesh(V0, faces_gt, colormap)

                    # newV, newF = getHeatMapMesh(V0, faces_gt, angColor)

                    write_mesh(newV, newF, RESULTS_PATH+heatFile)
                    
                    # Fill arrays
                    nameArray.append(denoizedFile)
                    resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum])

            outputFile = open(csv_filename,'a')
            nameArray = np.array(nameArray)
            resultsArray = np.array(resultsArray,dtype=np.float32)

            tempArray = resultsArray.flatten()
            resStr = ["%.7f" % number for number in tempArray]
            resStr = np.reshape(resStr,resultsArray.shape)

            nameArray = np.expand_dims(nameArray, axis=-1)

            finalArray = np.concatenate((nameArray,resStr),axis=1)
            for row in range(finalArray.shape[0]):
                for col in range(finalArray.shape[1]):
                    outputFile.write(finalArray[row,col])
                    outputFile.write(' ')
                outputFile.write('\n')

            outputFile.close()

    # Color mesh by estimated normals
    elif running_mode == 9:
        
        maxSize = 90000
        patchSize = 90000

        # Take the opportunity to generate array of metrics on reconstructions
        nameArray = []      # String array, to now which row is what
        resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                            # [Max distance, Mean distance, Mean angle, std angle, face num]

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results.csv"
        
        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("aMerlion")) or (gtFileName.startswith("aarmadillo")) or (gtFileName.startswith("agargoyle")) or \
            (gtFileName.startswith("adragon")):
                continue

            mesh_count = [0]
            # Get all 3 noisy meshes
            # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
            noisyFile0 = gtFileName[:-4]+"_n1.obj"
            noisyFile1 = gtFileName[:-4]+"_n2.obj"
            noisyFile2 = gtFileName[:-4]+"_n3.obj"

            noisyFileWColor0 = gtFileName[:-4]+"_n1C.obj"
            noisyFileWColor1 = gtFileName[:-4]+"_n2C.obj"
            noisyFileWColor2 = gtFileName[:-4]+"_n3C.obj"

            denoizedFile0 = gtFileName[:-4]+"_denoizedC_1.obj"
            denoizedFile1 = gtFileName[:-4]+"_denoizedC_2.obj"
            denoizedFile2 = gtFileName[:-4]+"_denoizedC_3.obj"

            if (os.path.isfile(RESULTS_PATH+denoizedFile0)) and (os.path.isfile(RESULTS_PATH+denoizedFile1)) and (os.path.isfile(RESULTS_PATH+denoizedFile2)):
                continue

            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            GTf_normals = computeFacesNormals(GT, faces_gt)

            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj2(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            

            noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            noisyFilesWColorList = [noisyFileWColor0,noisyFileWColor1,noisyFileWColor2]
            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                noisyFileWColor = noisyFilesWColorList[fileNum]
                
                if not os.path.isfile(RESULTS_PATH+denoizedFile):
                    

                    f_normals_list = []
                    GTfn_list = []
                    f_adj_list = []


                    faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, gtFolder, gtFileName, f_normals_list, GTfn_list, f_adj_list, mesh_count)

                    # Now recover vertices positions and create Edge maps
                    V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)
                    f_normals0 = computeFacesNormals(V0, faces_gt)
                    V0 = np.expand_dims(V0, axis=0)

                    print("running ...")
                    #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    
                    upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)

                    
                    print("computing Hausdorff "+str(fileNum+1)+"...")
                    haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                    angDistVec = angularDiffVec(upN0, GTf_normals)
                    angDist0, angStd0 = angularDiff(upN0, GTf_normals)
                    print("max angle: "+str(np.amax(angDistVec)))

                    
                    angColor = (upN0+1)/2

                    angColorNoisy = (f_normals0+1)/2
               
                    newV, newF = getColoredMesh(upV0, faces_gt, angColor)
                    newVn, newFn = getColoredMesh(np.squeeze(V0), faces_gt, angColor)
                    newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_gt, angColorNoisy)

                    write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
                    write_mesh(newVn, newFn, RESULTS_PATH+noisyFileWColor)
                    write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFile)

                    # Fill arrays
                    nameArray.append(denoizedFile)
                    resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum])

            outputFile = open(csv_filename,'a')
            nameArray = np.array(nameArray)
            resultsArray = np.array(resultsArray,dtype=np.float32)

            tempArray = resultsArray.flatten()
            resStr = ["%.7f" % number for number in tempArray]
            resStr = np.reshape(resStr,resultsArray.shape)

            nameArray = np.expand_dims(nameArray, axis=-1)

            finalArray = np.concatenate((nameArray,resStr),axis=1)
            for row in range(finalArray.shape[0]):
                for col in range(finalArray.shape[1]):
                    outputFile.write(finalArray[row,col])
                    outputFile.write(' ')
                outputFile.write('\n')

            outputFile.close()

    # Test. Load and write pickled data
    elif running_mode == 10:

        f_normals_list = []
        f_adj_list = []
        v_pos_list = []
        gtv_pos_list = []
        e_map_list = []
        v_emap_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_v_pos_list = []
        valid_gtv_pos_list = []
        valid_e_map_list = []
        valid_v_emap_list = []

        f_normals_list_temp = []
        f_adj_list_temp = []
        v_pos_list_temp = []
        gtv_pos_list_temp = []
        e_map_list_temp = []
        v_emap_list_temp = []

        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

        # Training
        pickleNum=4
        # while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
        with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
            f_normals_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
            f_adj_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'v_pos_list'+str(pickleNum), 'rb') as fp:
            v_pos_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'rb') as fp:
            gtv_pos_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'faces_list'+str(pickleNum), 'rb') as fp:
            faces_list_temp = pickle.load(fp, encoding='latin1')
        # with open(binDumpPath+'e_map_list'+str(pickleNum), 'rb') as fp:
        #     e_map_list_temp = pickle.load(fp, encoding='latin1')
        # with open(binDumpPath+'v_emap_list'+str(pickleNum), 'rb') as fp:
        #     v_emap_list_temp = pickle.load(fp, encoding='latin1')

        if pickleNum>=0:
            f_normals_list = f_normals_list_temp
            f_adj_list = f_adj_list_temp
            v_pos_list = v_pos_list_temp
            gtv_pos_list = gtv_pos_list_temp
            faces_list = faces_list_temp
            # e_map_list = e_map_list_temp
            # v_emap_list = v_emap_list_temp
        else:

            f_normals_list+=f_normals_list_temp
            f_adj_list+=f_adj_list_temp
            v_pos_list+=v_pos_list_temp
            gtv_pos_list+=gtv_pos_list_temp
            faces_list += faces_list_temp
            # e_map_list+=e_map_list_temp
            # v_emap_list+=v_emap_list_temp


        print("loaded training pickle "+str(pickleNum))
        pickleNum+=1


        # Validation
        with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
            valid_f_normals_list = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
            valid_f_adj_list = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'valid_v_pos_list', 'rb') as fp:
            valid_v_pos_list = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'valid_gtv_pos_list', 'rb') as fp:
            valid_gtv_pos_list = pickle.load(fp, encoding='latin1')
        # with open(binDumpPath+'valid_e_map_list', 'rb') as fp:
        #     valid_e_map_list = pickle.load(fp, encoding='latin1')
        # with open(binDumpPath+'valid_v_emap_list', 'rb') as fp:
        #     valid_v_emap_list = pickle.load(fp, encoding='latin1')
        testWriteFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Test/"
        for samp in range(len(v_pos_list)):
            V0 = np.squeeze(v_pos_list[samp])
            GT0 = np.squeeze(gtv_pos_list[samp])
            faces0 = np.squeeze(faces_list[samp])
            
            write_xyz(V0, testWriteFolder+'points_'+str(samp)+'.xyz')
            write_xyz(GT0, testWriteFolder+'GT0_'+str(samp)+'.xyz')

            write_mesh(V0,faces0,testWriteFolder+'mesh_'+str(samp)+'.obj')
            # haus0 = oneSidedHausdorff(V0,GT0)
            # print("Haus0: "+str(haus0))
            # validV0 = np.squeeze(valid_v_pos_list[samp])
            # validGT0 = np.squeeze(valid_gtv_pos_list[samp])
            # write_xyz(validV0, testWriteFolder+'v_points_'+str(samp)+'.xyz')
            # write_xyz(validGT0, testWriteFolder+'v_GT_'+str(samp)+'.xyz')

        # vhaus0 = oneSidedHausdorff(validV0,validGT0)
        # print("Haus0: "+str(vhaus0))

    # Load and pickle training data.
    elif running_mode == 11:

        f_normals_list_temp = []
        f_adj_list_temp = []
        v_pos_list_temp = []
        gtv_pos_list_temp = []
        e_map_list_temp = []
        v_emap_list_temp = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_v_pos_list = []
        valid_gtv_pos_list = []
        valid_e_map_list = []
        valid_v_emap_list = []

        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train2/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

        pickleNum=0
        # Training set
        for filename in os.listdir(inputFilePath):
            #print("training_meshes_num start_iter " + str(training_meshes_num))
            if training_meshes_num[0]>1000:
                break
            #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
            #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
            if (filename.endswith(".obj")):

                print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")

                # For DTU
                fileNumStr = filename[4:7]
                gtfilename = 'stl'+fileNumStr+'_total.obj'

                addMeshWithVertices(inputFilePath, filename, gtFilePath, gtfilename, v_pos_list_temp, gtv_pos_list_temp, f_normals_list_temp, f_adj_list_temp, e_map_list_temp, v_emap_list_temp, training_meshes_num)

                # Save batches of meshes/patches (for training only)
                if training_meshes_num[0]>30:
                    # Training
                    with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(f_normals_list_temp, fp)
                    with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(f_adj_list_temp, fp)
                    with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(v_pos_list_temp, fp)
                    with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(gtv_pos_list_temp, fp)
                    with open(binDumpPath+'e_map_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(e_map_list_temp, fp)
                    with open(binDumpPath+'v_emap_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(v_emap_list_temp, fp)

                    pickleNum+=1
                    f_normals_list_temp = []
                    f_adj_list_temp = []
                    v_pos_list_temp = []
                    gtv_pos_list_temp = []
                    e_map_list_temp = []
                    v_emap_list_temp = []
                    training_meshes_num[0] = 0

        if training_meshes_num[0]>0:
            # Training
            with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                pickle.dump(f_normals_list_temp, fp)
            with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                pickle.dump(f_adj_list_temp, fp)
            with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                pickle.dump(v_pos_list_temp, fp)
            with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                pickle.dump(gtv_pos_list_temp, fp)
            with open(binDumpPath+'e_map_list'+str(pickleNum), 'wb') as fp:
                pickle.dump(e_map_list_temp, fp)
            with open(binDumpPath+'v_emap_list'+str(pickleNum), 'wb') as fp:
                pickle.dump(v_emap_list_temp, fp)

        # # Validation set
        # for filename in os.listdir(validFilePath):
        #     if (filename.endswith(".obj")):

        #         # For DTU
        #         fileNumStr = filename[4:7]
        #         gtfilename = 'stl'+fileNumStr+'_total.obj'

        #         addMeshWithVertices(validFilePath, filename, gtFilePath, gtfilename, valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list, valid_meshes_num)
                
        # # Validation
        # with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
        #     pickle.dump(valid_f_normals_list, fp)
        # with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
        #     pickle.dump(valid_f_adj_list, fp)
        # with open(binDumpPath+'valid_v_pos_list', 'wb') as fp:
        #     pickle.dump(valid_v_pos_list, fp)
        # with open(binDumpPath+'valid_gtv_pos_list', 'wb') as fp:
        #     pickle.dump(valid_gtv_pos_list, fp)
        # with open(binDumpPath+'valid_e_map_list', 'wb') as fp:
        #     pickle.dump(valid_e_map_list, fp)
        # with open(binDumpPath+'valid_v_emap_list', 'wb') as fp:
        #     pickle.dump(valid_v_emap_list, fp)


    # Test normals transpose
    elif running_mode == 13:
        fileNumStr = '099'
        transpose_mode='varying_gaussian'         #'varying_gaussian'
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid_cleaned/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/results/fullLoss15Disp/"
        gtFileName = '000.obj'
        noisyFile = 'proc_'+fileNumStr+'.obj'
        # noisyFile = 'proc_'+fileNumStr+'_denoised_gray.obj'
        # denoizedFile = fileNumStr+'_'+transpose_mode+'.obj'
        # denoizedFile = fileNumStr+'_'+transpose_mode+'_disp.obj'
        gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Ground_Truth_Meshes/"

        for noisyFile in os.listdir(noisyFolder):
            if (not noisyFile.endswith(".obj")):
                continue

            #For FAUST
            fileNumStr = noisyFile[5:8]

            # if fileNumStr!='096':
            #     continue

            # gtFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/rescaled/M"+fileNumStr+"/Ground_Truth/"
            # denoizedFile = transpose_mode+'_'+fileNumStr+'.obj'
            gtFileName = "gt"+fileNumStr+".obj"
            denoizedFile = "dispColor"+'_'+fileNumStr+'.obj'
            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            
            
            # Now recover vertices positions and create Edge maps
            V0,_,_, faces0, _ = load_mesh(noisyFolder, noisyFile, 0, False)
            
            noisyNormals = computeFacesNormals(V0, faces0)

            # f_adj0 = getFacesLargeAdj(faces0,K_faces)
            # gt_adj = getFacesLargeAdj(faces_gt, K_faces)

            # f_adj0, _, _ = getFacesAdj2(faces0)
            # gt_adj, _, _ = getFacesAdj2(faces_gt)
            # newNormals = transposeNormals(V0, faces0, GT, faces_gt,f_adj0,gt_adj,mode=transpose_mode)
            newNormals = transposeNormals(V0, faces0, GT, faces_gt,0,0,mode=transpose_mode)

            newPoints, fDisp, lastNormals = stickMesh(V0, faces0, GT, faces_gt, 5, 20)

            write_mesh(newPoints,faces0,RESULTS_PATH+"stick_"+fileNumStr+".obj")

            dispColor = ((10*fDisp)+1)/2
            newV, newF = getColoredMesh(V0, faces0, dispColor)
            write_mesh(newV, newF, RESULTS_PATH+denoizedFile)

            # -- Test: apply displacement in one step:
            targetVFaces = getVerticesFaces(faces0,25, V0.shape[0])
            targetVFaces = targetVFaces+1
            vFacesCount = np.sum(targetVFaces>=0,axis=-1,keepdims=True)
            zeroLine = np.array([[0,0,0]],dtype=np.float32)

            fdispOff = np.concatenate((zeroLine,fDisp),axis=0)
            vDisp = fdispOff[targetVFaces]
            vDisp = np.divide(np.sum(vDisp,axis=1),vFacesCount)
            oneStepPoints = V0 + vDisp
            

            
            write_mesh(oneStepPoints, faces0, RESULTS_PATH+"oneStepDisp_"+fileNumStr+".obj")

            # Now, change normals

            for normIter in range(200):
                targetC = getTrianglesBarycenter(oneStepPoints, faces0, normalize=False)
                targetCOff = np.concatenate((zeroLine,targetC),axis=0)
                vFPos = targetCOff[targetVFaces]
                e = vFPos - np.expand_dims(oneStepPoints,axis=1)
                lastNormalsOff = np.concatenate((zeroLine,lastNormals),axis=0)

                vFNormals = lastNormalsOff[targetVFaces]

                dp = np.sum(np.multiply(e,vFNormals),axis=-1,keepdims=True)
                update = np.multiply(dp,vFNormals)
                update = np.sum(update,axis=1)
                update = np.divide(update,vFacesCount)
                oneStepPoints = oneStepPoints + update

            write_mesh(oneStepPoints, faces0, RESULTS_PATH+"oneStepFinal_"+fileNumStr+".obj")

            angColor = (lastNormals+1)/2
            # noisyColor = (noisyNormals+1)/2
            # gtColor = (gtNormals+1)/2
       
            newV, newF = getColoredMesh(V0, faces0, angColor)
            write_mesh(newV, newF, RESULTS_PATH+"finalNormals_"+fileNumStr+".obj")
            # newGT, newGTF = getColoredMesh(GT, faces_gt, gtColor)

            # newNoisyV, newNoisyF = getColoredMesh(V0, faces0, noisyColor)

            # write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
            # # write_mesh(newGT, newGTF, RESULTS_PATH+fileNumStr+'_gt.obj')
            # # write_mesh(newNoisyV, newNoisyF, RESULTS_PATH+fileNumStr+'_noisy.obj')
            # write_mesh(newGT, newGTF, RESULTS_PATH+'gt_'+fileNumStr+'.obj')
            # write_mesh(newNoisyV, newNoisyF, RESULTS_PATH+'noisy_'+fileNumStr+'.obj')

    # Test Hausdorff distance
    elif running_mode == 14:
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid/"
        # gtFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/rescaled/M"+fileNumStr+"/Ground_Truth/"
        gtFileName = '000.obj'
        # noisyFile = 'proc_'+fileNumStr+'.obj'

        for noisyFile in os.listdir(noisyFolder):
            if (not noisyFile.endswith(".obj")):
                continue

            print("filename = "+noisyFile)
            #For FAUST
            fileNumStr = noisyFile[5:8]

            if fileNumStr!='096':
                continue

            gtFolder = "/morpheo-nas2/marmando/MPI-FAUST/training/rescaled/M"+fileNumStr+"/Ground_Truth/"
            gtDFileName = "gt"+fileNumStr+".obj"
            # Load GT mesh
            GTD,_,_,_,_ = load_mesh(gtFolder, gtDFileName, 0, False)
            GT,_,_,_,_ = load_mesh(gtFolder, gtFileName, 0, False)
            V0,_,_,_,_ = load_mesh(noisyFolder, noisyFile, 0, False)

            t0 = time.clock()
            # accm, compm, accavg, compavg = hausdorffOverSampled(V0,GT,V0,GTD)
            results = hausdorffOverSampled(V0,GT,V0,GTD)

            # v0_list, v1_list = hausdorffOverSampled(V0,GT,V0,GTD)
            # curInd=0
            # for slice in v0_list:
            #     write_xyz(slice,RESULTS_PATH+"v0slice_"+str(curInd)+".xyz")
            #     curInd+=1
            # curInd=0
            # for slice in v1_list:
            #     write_xyz(slice,RESULTS_PATH+"v1slice_"+str(curInd)+".xyz")
            #     curInd+=1

            print("results: "+str(results))
            print("Took ("+str(1000*(time.clock()-t0))+"ms)")

    # Train network with soft normals loss
    elif running_mode == 15:


        gtnameoffset = 7
        f_normals_list = []
        f_adj_list = []
        v_pos_list = []
        gtv_pos_list = []
        faces_list = []
        v_faces_list = []
        gtfaces_list = []
        f_assign_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_v_pos_list = []
        valid_gtv_pos_list = []
        valid_faces_list = []
        valid_v_faces_list = []
        valid_gtfaces_list = []
        valid_f_assign_list = []

        f_normals_list_temp = []
        f_adj_list_temp = []
        v_pos_list_temp = []
        gtv_pos_list_temp = []
        faces_list_temp = []
        v_faces_list_temp = []
        gtfaces_list_temp = []
        f_assign_list_temp = []

        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train2/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
        # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"


        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/tola/train/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/tola/valid/"
        # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/train_cleaned/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/valid_cleaned/"
        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid_cleaned/"

        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Ground_Truth_Meshes/"

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"
        
        #print("training_meshes_num 0 " + str(training_meshes_num))
        if pickleLoad:

            # Training
            pickleNum=0
            while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
                with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
                    f_normals_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
                    f_adj_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'v_pos_list'+str(pickleNum), 'rb') as fp:
                    v_pos_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'rb') as fp:
                    gtv_pos_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'faces_list'+str(pickleNum), 'rb') as fp:
                    faces_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'v_faces_list'+str(pickleNum), 'rb') as fp:
                    v_faces_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'gtfaces_list'+str(pickleNum), 'rb') as fp:
                    gtfaces_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'f_assign_list'+str(pickleNum), 'rb') as fp:
                    f_assign_list_temp = pickle.load(fp, encoding='latin1')

                if pickleNum==0:
                    f_normals_list = f_normals_list_temp
                    f_adj_list = f_adj_list_temp
                    v_pos_list = v_pos_list_temp
                    gtv_pos_list = gtv_pos_list_temp
                    faces_list = faces_list_temp
                    v_faces_list = v_faces_list_temp
                    gtfaces_list = gtfaces_list_temp
                    f_assign_list = f_assign_list_temp
                else:

                    f_normals_list+=f_normals_list_temp
                    f_adj_list+=f_adj_list_temp
                    v_pos_list+=v_pos_list_temp
                    gtv_pos_list+=gtv_pos_list_temp
                    faces_list+=faces_list_temp
                    v_faces_list+=v_faces_list_temp
                    gtfaces_list+=gtfaces_list_temp
                    f_assign_list+=f_assign_list_temp


                print("loaded training pickle "+str(pickleNum))
                pickleNum+=1


            # Validation
            with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
                valid_f_normals_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
                valid_f_adj_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_v_pos_list', 'rb') as fp:
                valid_v_pos_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_gtv_pos_list', 'rb') as fp:
                valid_gtv_pos_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_faces_list', 'rb') as fp:
                valid_faces_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_v_faces_list', 'rb') as fp:
                valid_v_faces_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_gtfaces_list', 'rb') as fp:
                valid_gtfaces_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'valid_f_assign_list', 'rb') as fp:
                valid_f_assign_list = pickle.load(fp, encoding='latin1')


        else:

            pickleNum=0
            # Training set
            for filename in os.listdir(inputFilePath):
                #print("training_meshes_num start_iter " + str(training_meshes_num))
                if training_meshes_num[0]>1000:
                    break
                #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
                #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
                if (filename.endswith(".obj")):
                    if filename.startswith("a") or filename.startswith("b"):
                        continue
                    print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")

                    # # For DTU
                    # fileNumStr = filename[4:7]
                    # gtfilename = 'stl'+fileNumStr+'_total.obj'

                    # # For CNR dataset
                    # gtfilename = filename[:-gtnameoffset]+".obj"

                    #For FAUST
                    fileNumStr = filename[5:8]
                    # if int(fileNumStr)>2:
                    #     continue
                    gtfilename = 'gt'+fileNumStr+'.obj'

                    addMeshWithVerticesAndFaceAssignment(inputFilePath, filename, gtFilePath, gtfilename, v_pos_list_temp, gtv_pos_list_temp, faces_list_temp, f_normals_list_temp, f_adj_list_temp, v_faces_list_temp, gtfaces_list_temp, f_assign_list_temp, training_meshes_num)

                    # Save batches of meshes/patches (for training only)
                    if training_meshes_num[0]>30:
                        if pickleSave:
                            # Training
                            with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(f_normals_list_temp, fp)
                            with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(f_adj_list_temp, fp)
                            with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(v_pos_list_temp, fp)
                            with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(gtv_pos_list_temp, fp)
                            with open(binDumpPath+'faces_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(faces_list_temp, fp)
                            with open(binDumpPath+'v_faces_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(v_faces_list_temp, fp)
                            with open(binDumpPath+'gtfaces_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(gtfaces_list_temp, fp)
                            with open(binDumpPath+'f_assign_list'+str(pickleNum), 'wb') as fp:
                                pickle.dump(f_assign_list_temp, fp)
                        if pickleNum==0:
                            f_normals_list = f_normals_list_temp
                            f_adj_list = f_adj_list_temp
                            v_pos_list = v_pos_list_temp
                            gtv_pos_list = gtv_pos_list_temp
                            faces_list = faces_list_temp
                            v_faces_list = v_faces_list_temp
                            gtfaces_list = gtfaces_list_temp
                            f_assign_list = f_assign_list_temp
                        else:
                            f_normals_list+=f_normals_list_temp
                            f_adj_list+=f_adj_list_temp
                            v_pos_list+=v_pos_list_temp
                            gtv_pos_list+=gtv_pos_list_temp
                            faces_list+=faces_list_temp
                            v_faces_list+=v_faces_list_temp
                            gtfaces_list+=gtfaces_list_temp
                            f_assign_list+=f_assign_list_temp

                        pickleNum+=1
                        f_normals_list_temp = []
                        f_adj_list_temp = []
                        v_pos_list_temp = []
                        gtv_pos_list_temp = []
                        faces_list_temp = []
                        v_faces_list_temp = []
                        f_assign_list_temp = []
                        gtfaces_list_temp = []
                        training_meshes_num[0] = 0

            if (pickleSave) and training_meshes_num[0]>0:
                # Training
                with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(f_normals_list_temp, fp)
                with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(f_adj_list_temp, fp)
                with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(v_pos_list_temp, fp)
                with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(gtv_pos_list_temp, fp)
                with open(binDumpPath+'faces_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(faces_list_temp, fp)
                with open(binDumpPath+'v_faces_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(v_faces_list_temp, fp)
                with open(binDumpPath+'gtfaces_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(gtfaces_list_temp, fp)
                with open(binDumpPath+'f_assign_list'+str(pickleNum), 'wb') as fp:
                    pickle.dump(f_assign_list_temp, fp)

            if pickleNum==0:
                f_normals_list = f_normals_list_temp
                f_adj_list = f_adj_list_temp
                v_pos_list = v_pos_list_temp
                gtv_pos_list = gtv_pos_list_temp
                faces_list = faces_list_temp
                v_faces_list = v_faces_list_temp
                gtfaces_list = gtfaces_list_temp
                f_assign_list = f_assign_list_temp
            else:
                f_normals_list+=f_normals_list_temp
                f_adj_list+=f_adj_list_temp
                v_pos_list+=v_pos_list_temp
                gtv_pos_list+=gtv_pos_list_temp
                faces_list+=faces_list_temp
                v_faces_list+=v_faces_list_temp
                gtfaces_list+=gtfaces_list_temp
                f_assign_list+=f_assign_list_temp

            # Validation set
            for filename in os.listdir(validFilePath):
                if (filename.endswith(".obj")):
                    if valid_meshes_num[0]>200:
                        break
                    # # For DTU
                    # fileNumStr = filename[4:7]
                    # gtfilename = 'stl'+fileNumStr+'_total.obj'

                    # # For CNR dataset
                    # gtfilename = filename[:-gtnameoffset]+".obj"

                    #For FAUST
                    fileNumStr = filename[5:8]
                    gtfilename = 'gt'+fileNumStr+'.obj'

                    addMeshWithVerticesAndFaceAssignment(validFilePath, filename, gtFilePath, gtfilename, valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list, valid_gtfaces_list, valid_f_assign_list, valid_meshes_num)
                    

                #print("training_meshes_num end_iter " + str(training_meshes_num))
            
            if pickleSave:
                # Validation
                with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
                    pickle.dump(valid_f_normals_list, fp)
                with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
                    pickle.dump(valid_f_adj_list, fp)
                with open(binDumpPath+'valid_v_pos_list', 'wb') as fp:
                    pickle.dump(valid_v_pos_list, fp)
                with open(binDumpPath+'valid_gtv_pos_list', 'wb') as fp:
                    pickle.dump(valid_gtv_pos_list, fp)
                with open(binDumpPath+'valid_faces_list', 'wb') as fp:
                    pickle.dump(valid_faces_list, fp)
                with open(binDumpPath+'valid_v_faces_list', 'wb') as fp:
                    pickle.dump(valid_v_faces_list, fp)
                with open(binDumpPath+'valid_gtfaces_list', 'wb') as fp:
                    pickle.dump(valid_gtfaces_list, fp)
                with open(binDumpPath+'valid_f_assign_list', 'wb') as fp:
                    pickle.dump(valid_f_assign_list, fp)

        # oldSchoolGTN = []
        # valid_oldSchoolGTN = []
        # for i in range(len(f_assign_list)):
        #     curGTV = np.reshape(gtv_pos_list[i],[-1,3])
        #     curGTF = np.reshape(gtfaces_list[i],[-1,3])
        #     curGTN = computeFacesNormals(curGTV,curGTF)
        #     curInd = np.reshape(f_assign_list[i],[-1,10])
        #     # print("curInd shape: "+str(curInd.shape))
        #     # print(curInd[15140:15160,:])
        #     # print("min curInd = "+str(np.argmin(curInd)))
        #     curInd = curInd[:,0]
        #     # print("curInd shape: "+str(curInd.shape))
        #     # print("curGTN shape: "+str(curGTN.shape))
        #     curGTN = np.concatenate((np.zeros((1,3),dtype=np.float32),curGTN),axis=0)
        #     curInd = curInd+1
        #     oldSchoolGTN.append(np.expand_dims(curGTN[curInd],axis=0))

        # trainNet(f_normals_list, oldSchoolGTN, f_adj_list, f_normals_list, oldSchoolGTN, f_adj_list)

        # print("f_assign_list length: "+str(len(f_assign_list)))
        # print("f_normals_list length: "+str(len(f_normals_list)))
        # print("gtfaces_list length: "+str(len(gtfaces_list)))
        # # for i in range(len(f_assign_list)):
        # #     print("f_assign_list["+str(i)+"] shape: "+str(f_assign_list[i].shape))
        # #     print("f_normals_list["+str(i)+"] shape: "+str(f_normals_list[i].shape))
        # return

        trainSoftNormalsNet(v_pos_list, gtv_pos_list, faces_list, f_normals_list, f_adj_list, v_faces_list, gtfaces_list, f_assign_list,
                            valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list, valid_gtfaces_list, valid_f_assign_list)
        

    # Train network
    if running_mode == 16:

        gtnameoffset = 10
        f_normals_list = []
        f_adj_list = []
        GTfn_list = []
        GTfdisp_list = []

        f_normals_list_temp = []
        f_adj_list_temp = []
        GTfn_list_temp = []
        GTfdisp_list_temp = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_GTfn_list = []
        valid_GTfdisp_list = []

        toDelFaces_list = []

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"

        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/train/original/"


        # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/noisy/"
        # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/valid/"
        # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/original/"
        
        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid/"
        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid_cleaned/"
        # gtFilePath = "/morpheo-nas2/marmando/MPI-FAUST/training/rescaled/"
        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Ground_Truth_Meshes/"

        #print("training_meshes_num 0 " + str(training_meshes_num))
        if pickleLoad:

            # Training
            pickleNum=0
            while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
                with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
                    f_normals_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
                    f_adj_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'rb') as fp:
                    GTfdisp_list_temp = pickle.load(fp, encoding='latin1')
                with open(binDumpPath+'GTfn_list'+str(pickleNum), 'rb') as fp:
                    GTfn_list_temp = pickle.load(fp, encoding='latin1')

                if pickleNum==0:
                    f_normals_list = f_normals_list_temp
                    f_adj_list = f_adj_list_temp
                    GTfn_list = GTfn_list_temp
                    GTfdisp_list = GTfdisp_list_temp
                else:
                    f_normals_list+=f_normals_list_temp
                    f_adj_list+=f_adj_list_temp
                    GTfn_list+=GTfn_list_temp
                    GTfdisp_list+=GTfdisp_list_temp

                print("loaded training pickle "+str(pickleNum))
                pickleNum+=1

            # Validation
            with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
                valid_f_normals_list = pickle.load(fp)
            with open(binDumpPath+'valid_GTfn_list', 'rb') as fp:
                valid_GTfn_list = pickle.load(fp)
            with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
                valid_f_adj_list = pickle.load(fp)
            with open(binDumpPath+'valid_GTfdisp_list', 'rb') as fp:
                valid_GTfdisp_list = pickle.load(fp)


        else:
            pickleNum=400
            # Training set
            for filename in os.listdir(inputFilePath):
                #print("training_meshes_num start_iter " + str(training_meshes_num))
                if training_meshes_num[0]>300:
                    break
                #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
                #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
                if (filename.endswith(".obj")):


                    #For FAUST
                    fileNumStr = filename[5:8]
                    fileNum = int(fileNumStr)

                    if fileNum<100:
                        continue

                    gtfilename = 'gt'+fileNumStr+'.obj'
                    # if int(fileNumStr)>2:
                    #     continue
                    print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")
                    # gtfilename = filename[:-gtnameoffset]+".obj"
                    addMeshWGTDispAndNormals(inputFilePath, filename, gtFilePath, gtfilename, f_normals_list_temp, GTfn_list_temp, f_adj_list_temp, GTfdisp_list_temp, toDelFaces_list, training_meshes_num)


                if training_meshes_num[0]>4:
                    if pickleSave:
                        # Training
                        with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                            pickle.dump(f_normals_list_temp, fp)
                        with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                            pickle.dump(f_adj_list_temp, fp)
                        with open(binDumpPath+'GTfn_list'+str(pickleNum), 'wb') as fp:
                            pickle.dump(GTfn_list_temp, fp)
                        with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'wb') as fp:
                            pickle.dump(GTfdisp_list_temp, fp)
                        with open(binDumpPath+'toDelFaces_list'+str(pickleNum), 'wb') as fp:
                            pickle.dump(toDelFaces_list, fp)

                    if pickleNum==0:
                        f_normals_list = f_normals_list_temp
                        f_adj_list = f_adj_list_temp
                        GTfn_list = GTfn_list_temp
                        GTfdisp_list = GTfdisp_list_temp
                    else:
                        f_normals_list+=f_normals_list_temp
                        f_adj_list+=f_adj_list_temp
                        GTfn_list+=GTfn_list_temp
                        GTfdisp_list+=GTfdisp_list_temp

                    pickleNum+=1
                    f_normals_list_temp = []
                    f_adj_list_temp = []
                    GTfn_list_temp = []
                    GTfdisp_list_temp = []
                    training_meshes_num[0] = 0

            if (pickleSave) and training_meshes_num[0]>0:
                if pickleSave:
                    # Training
                    with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(f_normals_list_temp, fp)
                    with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(f_adj_list_temp, fp)
                    with open(binDumpPath+'GTfn_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(GTfn_list_temp, fp)
                    with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'wb') as fp:
                        pickle.dump(GTfdisp_list_temp, fp)

                if pickleNum==0:
                    f_normals_list = f_normals_list_temp
                    f_adj_list = f_adj_list_temp
                    GTfn_list = GTfn_list_temp
                    GTfdisp_list = GTfdisp_list_temp
                else:
                    f_normals_list+=f_normals_list_temp
                    f_adj_list+=f_adj_list_temp
                    GTfn_list+=GTfn_list_temp
                    GTfdisp_list+=GTfdisp_list_temp

                pickleNum+=1
                f_normals_list_temp = []
                f_adj_list_temp = []
                GTfn_list_temp = []
                GTfdisp_list_temp = []
                training_meshes_num[0] = 0

            
            # Validation set
            for filename in os.listdir(validFilePath):
                if (filename.endswith(".obj")):
                    # gtfilename = filename[:-gtnameoffset]+".obj"
                    #For FAUST
                    fileNumStr = filename[5:8]
                    gtfilename = 'gt'+fileNumStr+'.obj'
                    addMeshWGTDispAndNormals(validFilePath, filename, gtFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_GTfdisp_list, [], valid_meshes_num)

            if pickleSave:
                # Validation
                with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
                    pickle.dump(valid_f_normals_list, fp)
                with open(binDumpPath+'valid_GTfn_list', 'wb') as fp:
                    pickle.dump(valid_GTfn_list, fp)
                with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
                    pickle.dump(valid_f_adj_list, fp)
                with open(binDumpPath+'valid_GTfdisp_list', 'wb') as fp:
                    pickle.dump(valid_GTfdisp_list, fp)
        
        train6DNetWGT(f_normals_list, GTfn_list, f_adj_list, GTfdisp_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_GTfdisp_list)

    # Test. Load and write pickled data
    elif running_mode == 17:

        f_normals_list = []
        f_adj_list = []
        GTfn_list = []
        GTfdisp_list = []
        toDelFaces_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_GTfn_list = []
        valid_GTfdisp_list = []

        f_normals_list_temp = []
        f_adj_list_temp = []
        GTfn_list_temp = []
        GTfdisp_list_temp = []
        toDelFaces_list_temp = []

        inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train/"
        validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

        # Training
        pickleNum=0
        # while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
        with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
            f_normals_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
            f_adj_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'GTfn_list'+str(pickleNum), 'rb') as fp:
            GTfn_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'GTfdisp_list'+str(pickleNum), 'rb') as fp:
            GTfdisp_list_temp = pickle.load(fp, encoding='latin1')
        with open(binDumpPath+'toDelFaces_list'+str(pickleNum), 'rb') as fp:
            toDelFaces_list_temp = pickle.load(fp, encoding='latin1')


        if pickleNum>=0:
            f_normals_list = f_normals_list_temp
            f_adj_list = f_adj_list_temp
            GTfn_list = GTfn_list_temp
            GTfdisp_list = GTfdisp_list_temp
            toDelFaces_list = toDelFaces_list_temp

        else:

            f_normals_list+=f_normals_list_temp
            f_adj_list+=f_adj_list_temp
            GTfn_list += GTfn_list_temp
            GTfdisp_list += GTfdisp_list_temp
            toDelFaces_list += toDelFaces_list_temp

        print("loaded training pickle "+str(pickleNum))
        pickleNum+=1


        testWriteFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Test/fullGT/"
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train_cleaned/"
        sampnum=60
        # for samp in range(len(f_normals_list)):
        for  samp in range(1):

            noisyFile = "proc_%03d.obj" % sampnum
            fileNumStr = "%03d" % sampnum
            V0,_,_, faces0, _ = load_mesh(noisyFolder, noisyFile, 0, False)

            fDisp = np.squeeze(GTfdisp_list[samp])
            lastNormals = np.squeeze(GTfn_list[samp])
            faces0 = np.squeeze(toDelFaces_list[samp])
            # faces0 = np.squeeze()



            # -- Test: apply displacement in one step:
            targetVFaces = getVerticesFaces(faces0,25, V0.shape[0])
            targetVFaces = targetVFaces+1
            vFacesCount = np.sum(targetVFaces>=0,axis=-1,keepdims=True)
            zeroLine = np.array([[0,0,0]],dtype=np.float32)

            fdispOff = np.concatenate((zeroLine,fDisp),axis=0)
            vDisp = fdispOff[targetVFaces]
            vDisp = np.divide(np.sum(vDisp,axis=1),vFacesCount)
            oneStepPoints = V0 + vDisp


            write_mesh(oneStepPoints, faces0, RESULTS_PATH+"oneStepDisp_"+fileNumStr+".obj")

            # Now, change normals

            for normIter in range(200):
                targetC = getTrianglesBarycenter(oneStepPoints, faces0, normalize=False)
                targetCOff = np.concatenate((zeroLine,targetC),axis=0)
                vFPos = targetCOff[targetVFaces]
                e = vFPos - np.expand_dims(oneStepPoints,axis=1)
                lastNormalsOff = np.concatenate((zeroLine,lastNormals),axis=0)

                vFNormals = lastNormalsOff[targetVFaces]

                dp = np.sum(np.multiply(e,vFNormals),axis=-1,keepdims=True)
                update = np.multiply(dp,vFNormals)
                update = np.sum(update,axis=1)
                update = np.divide(update,vFacesCount)
                oneStepPoints = oneStepPoints + update

            write_mesh(oneStepPoints, faces0, RESULTS_PATH+"oneStepFinal_"+fileNumStr+".obj")

            angColor = (lastNormals+1)/2
            # noisyColor = (noisyNormals+1)/2
            # gtColor = (gtNormals+1)/2
       
            newV, newF = getColoredMesh(V0, faces0, angColor)
            write_mesh(newV, newF, RESULTS_PATH+"finalNormals_"+fileNumStr+".obj")



    #

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=int, default=0)
    #parser.add_argument('--dataset_path')
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--net_name', type=str, default='net')
    parser.add_argument('--running_mode', type=int, default=0)
    parser.add_argument('--coarsening_steps', type=int, default=3)
    

    #parser.add_argument('--num_classes', type=int)

    FLAGS = parser.parse_args()

    ARCHITECTURE = FLAGS.architecture
    #DATASET_PATH = FLAGS.dataset_path
    RESULTS_PATH = FLAGS.results_path
    NETWORK_PATH = FLAGS.network_path
    NUM_ITERATIONS = FLAGS.num_iterations
    DEVICE = FLAGS.device
    NET_NAME = FLAGS.net_name
    RUNNING_MODE = FLAGS.running_mode
    COARSENING_STEPS = FLAGS.coarsening_steps
    #NUM_CLASSES = FLAGS.num_classes

    mainFunction()


