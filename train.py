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
        # random_seed = 0
        # np.random.seed(random_seed)

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
            # n_conv,_,_ = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
            n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)

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
            # print("Random R = "+str(random_R))
            # random_R = np.identity(3)
            num_p = f_normals[i].shape[1]
            tens_random_R = np.reshape(random_R,(1,1,3,3))
            tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

            # my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2], rot_mat:tens_random_R2,
            #                 keep_prob:1.0}
            my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0]}

            # writer = tf.summary.FileWriter("/morpheo-nas2/marmando/DeepMeshRefinement/TensorboardTest/", sess.graph)
            # print(sess.run(n_conv,feed_dict=my_feed_dict))
            # writer.close()
            # return

            if len(f_adj[0])>1:
                my_feed_dict[fadj1]=f_adj[i][1]
                my_feed_dict[fadj2]=f_adj[i][2]
            if len(f_adj[0])>3:
                my_feed_dict[fadj3]=f_adj[i][3]
            outN = sess.run(squeezed_n_conv,feed_dict=my_feed_dict)
            #outN = f_normals[i][0]

            if len(f_adj[0])>1:
                # Permute back patch
                # temp_perm = inv_perm(old_to_new_permutations[i])
                temp_perm = old_to_new_permutations[i]
                outN = outN[temp_perm]
                outN = outN[0:num_wofake_nodes[i]]

            if patchNumber==1:
            # if len(patch_indices[i]) == 0:
                predicted_normals = outN
            else:
                predicted_normals[patch_indices[i]] = predicted_normals[patch_indices[i]] + outN
                
        #Update vertices position
        new_normals = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='fn_')
        #refined_x = update_position(xp_,fadj, n_conv)
        refined_x = update_position2(xp_, new_normals, e_map_, ve_map_, iter_num=60, max_edges = MAX_EDGES)
        # refined_x, x_update = update_position_with_depth(xp_, new_normals, e_map_, ve_map_, depth_dir, iter_num=200)
        points = tf.squeeze(refined_x)
        # points_update = tf.squeeze(x_update)
        points_update = points


        predicted_normals = normalize(predicted_normals)

        update_feed_dict = {xp_:in_points, new_normals: [predicted_normals], e_map_: edge_map, ve_map_: v_e_map}
        outPoints, x_disp = sess.run([points, points_update],feed_dict=update_feed_dict)
        sess.close()

        x_disp = normalize(x_disp)

        # print("x_disp sample: "+str(x_disp[:10,:]))

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
        if(FLAGS.debug):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        

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
        
            # n_conv1 = custom_binary_tree_pooling(n_conv0, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')
            # n_conv2 = custom_binary_tree_pooling(n_conv1, steps=COARSENING_STEPS, pooltype='avg_ignore_zeros')

        # n_conv0 = fn_
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
        # points = tf.squeeze(refined_x)

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
            # outN0 = np.tile(np.array([[[0,0,1]]]),[1,f_normals[0].shape[1],1])

            
            

            update_feed_dict = {xp_:in_points[i], new_normals0: outN0, pos0: outP0,
                                faces_: faces[i], v_faces_: v_faces[i]}
            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}
            # testNorm = f_normals[i][:,:,:3]/100
            update_feed_dict = {xp_:in_points[i], new_normals0: outN0, new_normals1: outN1, new_normals2: outN2, pos0: outP0,
                                faces_: faces[i], v_faces_: v_faces[i]}
            # update_feed_dict = {xp_:in_points[i], new_normals0: outN0,
            #                     faces_: faces[i], v_faces_: v_faces[i]}

            print("Running points...")

            

            # outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, new_normals0, upN1, upN2],feed_dict=update_feed_dict)
            # outPoints, fineNormals, midNormals, coarseNormals, coarseNormals2, coarseNormals3 = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse, normalised_disp_coarse2, normalised_disp_coarse3],feed_dict=update_feed_dict)
            # outPoints, fineNormals, midNormals, coarseNormals = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse],feed_dict=update_feed_dict)

            # outPoints, fineNormals, midNormals, coarseNormals, finePos, midPos, coarsePos = sess.run([points, normalised_disp_fine, normalised_disp_mid, normalised_disp_coarse, pos0, pos1, pos2],feed_dict=update_feed_dict)
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

                # fineNormalsP = np.squeeze(fineNormals)
                # midNormalsP = np.squeeze(midNormals)
                # coarseNormalsP = np.squeeze(coarseNormals)

                finePosP = np.squeeze(finePos)
                midPosP = np.squeeze(midPos)
                coarsePosP = np.squeeze(coarsePos)

                # finePosP = np.squeeze(finePos)[adjPerm_list[i]]
                # midPosP = np.squeeze(midPos)[adjPerm_list[i]]
                # coarsePosP = np.squeeze(coarsePos)[adjPerm_list[i]]

                # finePosP = finePosP[:real_nodes_num_list[i],:]
                # midPosP = midPosP[:real_nodes_num_list[i],:]
                # coarsePosP = coarsePosP[:real_nodes_num_list[i],:]

                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
                midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
                coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
                coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]


                # coarseNormalsP2 = np.squeeze(coarseNormals2)[adjPerm_list[i]]
                # coarseNormalsP2 = coarseNormalsP2[:real_nodes_num_list[i],:]
                # coarseNormalsP3 = np.squeeze(coarseNormals3)[adjPerm_list[i]]
                # coarseNormalsP3 = coarseNormalsP3[:real_nodes_num_list[i],:]

                finalFineNormals[new_to_old_f_list[i]] = fineNormalsP
                finalMidNormals[new_to_old_f_list[i]] = midNormalsP
                finalCoarseNormals[new_to_old_f_list[i]] = coarseNormalsP

                # finalFineNormals = fineNormalsP
                # finalMidNormals = midNormalsP
                # finalCoarseNormals = coarseNormalsP

                finalFinePos = finePosP
                finalMidPos = midPosP
                finalCoarsePos = coarsePosP

                # finalFinePos[new_to_old_f_list[i]] = finePosP
                # finalMidPos[new_to_old_f_list[i]] = midPosP
                # finalCoarsePos[new_to_old_f_list[i]] = coarsePosP

            else:
                finalOutPoints = outPoints
                finalOutPointsMid = outPointsMid
                finalOutPointsCoarse = outPointsCoarse
                pointsWeights +=1

                # fineNormalsP = np.squeeze(fineNormals)
                # midNormalsP = np.squeeze(midNormals)
                # coarseNormalsP = np.squeeze(coarseNormals)

                fineNormalsP = np.squeeze(fineNormals)[adjPerm_list[i]]
                fineNormalsP = fineNormalsP[:real_nodes_num_list[i],:]
                midNormalsP = np.squeeze(midNormals)[adjPerm_list[i]]
                midNormalsP = midNormalsP[:real_nodes_num_list[i],:]
                coarseNormalsP = np.squeeze(coarseNormals)[adjPerm_list[i]]
                coarseNormalsP = coarseNormalsP[:real_nodes_num_list[i],:]

                # coarseNormalsP2 = np.squeeze(coarseNormals2)[adjPerm_list[i]]
                # coarseNormalsP2 = coarseNormalsP2[:real_nodes_num_list[i],:]
                # coarseNormalsP3 = np.squeeze(coarseNormals3)[adjPerm_list[i]]
                # coarseNormalsP3 = coarseNormalsP3[:real_nodes_num_list[i],:]

                # finePosP = np.squeeze(finePos)
                # midPosP = np.squeeze(midPos)
                # coarsePosP = np.squeeze(coarsePos)

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
                # finalCoarseNormals2 = coarseNormalsP2
                # finalCoarseNormals3 = coarseNormalsP3
            print("Mesh update: check")
        sess.close()

        finalOutPoints = np.true_divide(finalOutPoints,np.maximum(pointsWeights,1))
        finalOutPointsMid = np.true_divide(finalOutPointsMid,np.maximum(pointsWeights,1))
        finalOutPointsCoarse = np.true_divide(finalOutPointsCoarse,np.maximum(pointsWeights,1))

        # return finalOutPoints, finalFineNormals, finalMidNormals, finalCoarseNormals, finalCoarseNormals, finalCoarseNormals, finalFinePos, finalMidPos, finalCoarsePos
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
    # random_seed = 0
    # np.random.seed(random_seed)

    # sess = tf.InteractiveSession(config=tf.ConfigProto( allow_soft_placement=True, log_device_placement=False))
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
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
    # print("NUM_IN_CHANNELS/3 = "+str(NUM_IN_CHANNELS/3))
    # print("int(NUM_IN_CHANNELS/3) = "+str(int(NUM_IN_CHANNELS/3)))
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
        # n_conv = get_model_reg(fn_, fadj0, ARCHITECTURE, keep_prob)
        # n_conv,_,_ = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)
        n_conv = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)


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
                print("Ongoing training: architecture "+str(ARCHITECTURE)+", net path = "+NETWORK_PATH)
                if sess.run(isFullNanNConv, feed_dict=train_fd):
                    break

            # Get random sample from training dictionary
            batch_num = random.randint(0,len(f_normals_list)-1)

            num_p = f_normals_list[batch_num].shape[1]
            # print("num_p = ",num_p)
            random_ind = np.random.randint(num_p,size=costSamplesNum)
            # random_ind = np.random.randint(1,size=10000)

            random_R = rand_rotation_matrix()
            tens_random_R = np.reshape(random_R,(1,1,3,3))
            tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

            # train_fd = {fn_: f_normals_list[batch_num], fadj: f_adj_list[batch_num], tfn_: GTfn_list[batch_num],
            #               sample_ind: random_ind, keep_prob:1}

            # train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], tfn_: GTfn_list[batch_num],
            #               sample_ind: random_ind, keep_prob:1}

            
            


            train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], tfn_: GTfn_list[batch_num], rot_mat:tens_random_R2,
                            sample_ind: random_ind, keep_prob:1}

            if len(f_adj_list[0])>1:
                train_fd[fadj1]=f_adj_list[batch_num][1]
                train_fd[fadj2]=f_adj_list[batch_num][2]
            # print("OK?")
            if len(f_adj_list[0])>3:
                train_fd[fadj3]=f_adj_list[batch_num][3]
            #i = train_shuffle[iter%(len(train_data))]
            #in_points = train_data[i]

            #sess.run(customLoss,feed_dict=train_fd)

            # print("OK")
            # train_loss += customLoss.eval(feed_dict=train_fd)
            train_loss += sess.run(customLoss,feed_dict=train_fd)
            train_samp+=1
            # print("Still OK!")
            # Show smoothed training loss
            if (iter%evalStepNum == 0):
                train_loss = train_loss/train_samp
                # sess.run(customLoss2,feed_dict=my_feed_dict)
                # train_loss2 = customLoss2.eval(feed_dict=my_feed_dict)
                # sess.run(customLoss3,feed_dict=my_feed_dict)
                # train_loss3 = customLoss3.eval(feed_dict=my_feed_dict)

                print("Iteration %d, training loss %g"%(iter, train_loss))
                # print("Iteration %d, training loss2 %g"%(iter, train_loss2))
                # print("Iteration %d, training loss3 %g"%(iter, train_loss3))

                lossArray[int(iter/evalStepNum),0]=train_loss
                train_loss=0
                train_samp=0

            # Compute validation loss
            if (iter%(evalStepNum*2) ==0):
                valid_loss = 0
                valid_samp = len(valid_f_normals_list)
                # print("valid num_p = ",num_p)
                valid_random_ind = np.arange(costSamplesNum)
                # valid_random_ind = np.random.randint(1,size=10000)
                for vbm in range(valid_samp):
                    # valid_fd = {fn_: valid_f_normals_list[vbm], fadj: valid_f_adj_list[vbm], tfn_: valid_GTfn_list[vbm],
                    #       sample_ind: valid_random_ind, keep_prob:1}

                    # valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], tfn_: valid_GTfn_list[vbm],
                    #       sample_ind: valid_random_ind, keep_prob:1}
                    num_p = valid_f_normals_list[vbm].shape[1]
                    # valid_random_ind = np.random.randint(num_p,size=10000)

                    tens_random_R2 = np.tile(tens_random_R,(BATCH_SIZE,num_p,1,1))

                    valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], tfn_: valid_GTfn_list[vbm], rot_mat:tens_random_R2,
                            sample_ind: valid_random_ind, keep_prob:1}

                    if len(f_adj_list[0])>1:
                        valid_fd[fadj1]=valid_f_adj_list[vbm][1]
                        valid_fd[fadj2]=valid_f_adj_list[vbm][2]

                    if len(f_adj_list[0])>3:
                        valid_fd[fadj3]=valid_f_adj_list[vbm][3]

                    # valid_loss += customLoss.eval(feed_dict=valid_fd)
                    valid_loss += sess.run(customLoss,feed_dict=valid_fd)
                valid_loss/=valid_samp
                print("Iteration %d, validation loss %g"%(iter, valid_loss))
                lossArray[int(iter/evalStepNum),1]=valid_loss
                if iter>0:
                    lossArray[int(iter/evalStepNum)-1,1] = (valid_loss+last_loss)/2
                    last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            # sess.run(train_step2,feed_dict=my_feed_dict)
            # sess.run(train_step3,feed_dict=my_feed_dict)
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


def trainAccuracyNet(in_points_list, GT_points_list, faces_list, f_normals_list, f_adj_list, v_faces_list, valid_in_points_list, valid_GT_points_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list):
    
    # random_seed = 0
    # np.random.seed(random_seed)
    SAMP_NUM = 500
    # keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
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
        n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)
        # n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)


    n_conv0 = normalizeTensor(n_conv0)
    # n_conv1 = normalizeTensor(n_conv1)
    # n_conv2 = normalizeTensor(n_conv2)

    n_conv_list = [n_conv0, n_conv1, n_conv2]
    # isNanNConv = tf.reduce_any(tf.is_nan(n_conv), name="isNanNConv")
    # isFullNanNConv = tf.reduce_all(tf.is_nan(n_conv), name="isNanNConv")

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




def trainDoubleLossNet(in_points_list, GT_points_list, faces_list, f_normals_list, GTf_normals_list, f_adj_list, v_faces_list, valid_in_points_list, valid_GT_points_list, valid_faces_list, valid_f_normals_list, valid_GTf_normals_list, valid_f_adj_list, valid_v_faces_list):
    
    # random_seed = 0
    # np.random.seed(random_seed)
    SAMP_NUM = 500
    # keep_rot_inv=True

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
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
        n_conv0, n_conv1, n_conv2 = get_model_reg_multi_scale(fn_rot, fadjs, ARCHITECTURE, keep_prob)
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
            print("Ongoing training: architecture "+str(ARCHITECTURE)+", net path = "+NETWORK_PATH)
            if sess.run(isFullNanNConv, feed_dict=train_fd):
                break
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




def mainFunction():

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    if not os.path.exists(BINARY_DUMP_PATH):
        os.makedirs(BINARY_DUMP_PATH)

    maxSize = MAX_PATCH_SIZE
    patchSize = MAX_PATCH_SIZE

    training_meshes_num = [0]
    valid_meshes_num = [0]


    # Coarsening parameters
    # coarseningLvlNum = 3
    coarseningStepNum = COARSENING_STEPS
    coarseningLvlNum = COARSENING_LVLS


    binDumpPath = BINARY_DUMP_PATH


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
        t0 = time.clock()
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
        t1 = time.clock()
        print("mesh loaded ("+str(1000*(t1-t0))+"ms)")

        # print("faces0 shape: "+str(faces0.shape))
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        t2 = time.clock()
        print("normals computed ("+str(1000*(t2-t1))+"ms)")
        
        # Get adjacency
        # f_adj0 = getFacesLargeAdj(faces0,K_faces)
        f_adj0 = getFacesOrderedAdj(faces0,K_faces)
        t3 = time.clock()
        print("Adj computed ("+str(1000*(t3-t2))+"ms)")
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0)
        t4 = time.clock()
        print("faces barycenters computed ("+str(1000*(t4-t3))+"ms)")
        f_area0 = np.expand_dims(getTrianglesArea(V0,faces0, normalize=True), axis=1)
        t5 = time.clock()
        print("Areas computed ("+str(1000*(t5-t4))+"ms)")
        f_borderCh0 = np.expand_dims(getBorderFaces(faces0),axis=1)
        
        t6 = time.clock()
        print("Borders computed ("+str(1000*(t6-t5))+"ms)")
        # print("WARNING!!! hard-coded change in data loading: FND instead of normals")
        # with open("/morpheo-nas2/marmando/DeepMeshRefinement/Synthetic/BinaryDump/Dump2_FND_el/"+filename+"FND", 'rb') as fp:
        #     f_FND = pickle.load(fp)
        # f_normals_pos = np.concatenate((f_FND, f_pos0), axis=1)
        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)
        # print("WARNING!!! Added binary channel for border faces")
        # f_normals_pos = np.concatenate((f_normals0, f_borderCh0, f_pos0), axis=1)

        # print("WARNING!!! Added area channel and binary channel for border faces")
        # f_normals_pos = np.concatenate((f_normals0, f_area0, f_borderCh0, f_pos0), axis=1)

        # f_area0 = np.reshape(f_area0, (-1,1))
        # f_normals0 = np.concatenate((f_normals0, f_area0), axis=1)

        t7 = time.clock()

        # Load GT
        GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)
        GTf_normals0 = computeFacesNormals(GT0, faces0)

        t8 = time.clock()
        print("GT loaded + normals computed ("+str(1000*(t8-t7))+"ms)")

        # Get patches if mesh is too big
        facesNum = faces0.shape[0]

        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>MAX_PATCH_SIZE:
            print("Dividing mesh into patches: %i faces (%i max allowed)"%(facesNum,MAX_PATCH_SIZE))
            patchNum = 0
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]
                tp0 = time.clock()
                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)
                tp1 = time.clock()
                print("Mesh patch extracted ("+str(1000*(tp1-tp0))+"ms)")
                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = GTf_normals0[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue

                if coarseningLvlNum>1:

                    # Convert to sparse matrix and coarsen graph
                    coo_adj = listToSparseWNormals(testPatchAdj, patchFNormals[:,-3:], patchFNormals[:,:3])

                    has_sat = True

                    while has_sat:
                        print("Coarsening...")
                        adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                        has_sat = False
                        # Change adj format
                        fAdjs = []
                        for lvl in range(coarseningLvlNum):
                            fadj, has_sat_temp = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                            fadj = np.expand_dims(fadj, axis=0)
                            fAdjs.append(fadj)
                            has_sat = has_sat or has_sat_temp



                    # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                    new_N = len(newToOld)
                    
                    padding6 =np.zeros((new_N-old_N,patchFNormals.shape[1]))
                    # padding6 =np.zeros((new_N-old_N,33))
                    padding3 =np.zeros((new_N-old_N,3))
                    patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                    patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                    # Reorder nodes
                    patchFNormals = patchFNormals[newToOld]
                    patchGTFNormals = patchGTFNormals[newToOld]
                else:
                    fAdjs = []
                    fAdjs.append(testPatchAdj[np.newaxis,:,:])

                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)
                if coarseningLvlNum>1:
                    new_to_old_permutations_list.append(inv_perm(newToOld))
                #####################################################################

                

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
            old_N = facesNum
            # Convert to sparse matrix and coarsen graph
            # print("f_adj0 shape: "+str(f_adj0.shape))
            # print("f_pos0 shape: "+str(f_pos0.shape))
            
            
            if coarseningLvlNum>1:
                coo_adj = listToSparseWNormals(f_adj0, f_pos0, f_normals0)
                has_sat = True

                while has_sat:
                    print("Coarsening...")
                    adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)
                    has_sat = False

                    # Change adj format
                    fAdjs = []
                    for lvl in range(coarseningLvlNum):
                        fadj, has_sat_temp = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                        fadj = np.expand_dims(fadj, axis=0)
                        fAdjs.append(fadj)
                        has_sat = has_sat or has_sat_temp

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,f_normals_pos.shape[1]))
                padding3 =np.zeros((new_N-old_N,3))
                f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
                GTf_normals0 = np.concatenate((GTf_normals0, padding3),axis=0)
                # Reorder nodes
                f_normals_pos = f_normals_pos[newToOld]
                GTf_normals0 = GTf_normals0[newToOld]

            else:
                fAdjs = []
                fAdjs.append(f_adj0[np.newaxis,:,:])

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            if coarseningLvlNum>1:
                new_to_old_permutations_list.append(inv_perm(newToOld)) # Nothing to append here, faces are already correctly ordered
            #####################################################################

            

            

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            #f_adj = np.expand_dims(f_adj0, axis=0)
            GTf_normals = np.expand_dims(GTf_normals0, axis=0)

            in_list.append(f_normals)
            adj_list.append(fAdjs)
            gt_list.append(GTf_normals)
        
            # print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return num_faces, patch_indices, new_to_old_permutations_list

    #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensroflow graph, with the right format
    def addMeshWithVertices(inputFilePath,filename, gtFilePath, gtfilename, v_list, gtv_list, faces_list, n_list, gtn_list, adj_list, v_faces_list, mesh_count_list):
        patch_indices = []
        new_to_old_permutations_list = []
        num_faces = []
        vOldInd_list = []
        fOldInd_list = []

        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)

        vNum = V0.shape[0]
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        # f_adj0,_,_ = getFacesAdj(faces0)
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        # print("WARNING: temp change to face position normalization!! TO BE REMOVED!!!")
        f_pos0 = getTrianglesBarycenter(V0, faces0, normalize=True)
        # f_pos0 = np.reshape(f_pos0,(-1,3))

        f_area0 = np.expand_dims(getTrianglesArea(V0,faces0, normalize=True), axis=1)

        f_borderCh0 = np.expand_dims(getBorderFaces(faces0),axis=1)

        # print("WARNING!!! Added area channel and binary channel for border faces")
        # f_normals_pos = np.concatenate((f_normals0, f_area0, f_borderCh0, f_pos0), axis=1)
        # print("WARNING!!! Added binary channel for border faces")
        # f_normals_pos = np.concatenate((f_normals0, f_borderCh0, f_pos0), axis=1)
        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)

        # Load GT
        GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)

        gtf_normals0 = computeFacesNormals(GT0, faces0)

        # Normalize vertices
        V0, GT0 = normalizePointSets(V0,GT0)


        # Get patches if mesh is too big
        facesNum = faces0.shape[0]
        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        print("faces num = %i"%facesNum)
        print("maxSize = %i"%maxSize)
        if facesNum>maxSize:
            patchNum = 0
            # while((np.any(faceCheck==0))and(patchNum<3)):
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)
                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = gtf_normals0[fOldInd]

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
                coo_adj = listToSparseWNormals(testPatchAdj, patchFNormals[:,-3:], patchFNormals[:,:3])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                # padding6 =np.zeros((new_N-old_N,6))
                padding6 =np.zeros((new_N-old_N,patchFNormals.shape[1]))
                padding3 =np.zeros((new_N-old_N,3))
                minusPadding3 = padding3-1
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                testPatchF = np.concatenate((testPatchF,minusPadding3),axis=0)
                patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                testPatchF = testPatchF[newToOld]
                patchGTFNormals = patchGTFNormals[newToOld]

                oldToNew = inv_perm(newToOld)


                ##### Save number of triangles and patch new_to_old permutation #####
                num_faces.append(old_N)
                patch_indices.append(fOldInd)

                new_to_old_permutations_list.append(oldToNew)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj, _ = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)
                        # fAdjs = []
                        # f_adj = np.expand_dims(testPatchAdj, axis=0)
                        # fAdjs.append(f_adj)

                v_faces = getVerticesFaces(testPatchF,25,testPatchV.shape[0])

                print("mesh size:")
                print("faces: %i"%patchFNormals.shape[0])
                print("vertices: %i"%testPatchV.shape[0])
                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                v_pos = np.expand_dims(testPatchV,axis=0)
                faces = np.expand_dims(testPatchF, axis=0)
                gtv_pos = np.expand_dims(patchGTV,axis=0)
                v_faces = np.expand_dims(v_faces,axis=0)
                gtf_normals = np.expand_dims(patchGTFNormals, axis=0)

                v_list.append(v_pos)
                gtv_list.append(gtv_pos)
                n_list.append(f_normals)
                adj_list.append(fAdjs)
                faces_list.append(faces)
                v_faces_list.append(v_faces)
                gtn_list.append(gtf_normals)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            coo_adj = listToSparseWNormals(f_adj0, f_pos0, f_normals0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)
            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            # padding6 =np.zeros((new_N-old_N,6))
            padding6 =np.zeros((new_N-old_N,f_normals_pos.shape[1]))
            padding3 =np.zeros((new_N-old_N,3))
            minusPadding3 = padding3-1
            minusPadding3 = minusPadding3.astype(int)

            faces0 = np.concatenate((faces0,minusPadding3),axis=0)

            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            gtf_normals = np.concatenate((gtf_normals0, padding3),axis=0)

            oldToNew = inv_perm(newToOld)

            ##### Save number of triangles and patch new_to_old permutation #####
            num_faces.append(old_N) # Keep track of fake nodes
            patch_indices.append([])
            new_to_old_permutations_list.append(oldToNew)
            fOldInd_list.append([])
            vOldInd_list.append([])
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            faces0 = faces0[newToOld]
            gtf_normals = gtf_normals[newToOld]
            

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj, _ = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
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
            gtf_normals = np.expand_dims(gtf_normals,axis=0)

            v_list.append(v_pos)
            gtv_list.append(gtv_pos)
            n_list.append(f_normals)
            adj_list.append(fAdjs)
            faces_list.append(faces)
            v_faces_list.append(v_faces)
            gtn_list.append(gtf_normals)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

        return vOldInd_list, fOldInd_list, vNum, facesNum, new_to_old_permutations_list, num_faces, patch_indices

    
    if RUNNING_MODE == -5: # PICKLEEEEEEEEEEE
        
        myTS = TrainingSet(maxSize, coarseningStepNum, coarseningLvlNum)
        # Training set
        for filename in os.listdir(TRAINING_DATA_PATH):
            if myTS.mesh_count>10:
                break

            if (filename.endswith(".obj")):
                print("Adding " + filename + " (" + str(myTS.mesh_count) + ")")
                gtfilename = getGTFilename(filename)
                for it in range(TRAINING_DATA_REDUNDANCY):
                    myTS.addMeshWithGT(TRAINING_DATA_PATH,filename,GT_DATA_PATH,gtfilename)

        with open(binDumpPath+'trainingSet.pkl','wb') as fp:
            pickle.dump(myTS,fp)

        # with open(binDumpPath+'f_normals_list', 'wb') as fp:
        #     pickle.dump(myTS.in_list, fp)
        # with open(binDumpPath+'GTfn_list', 'wb') as fp:
        #     pickle.dump(myTS.gt_list, fp)
        # with open(binDumpPath+'f_adj_list', 'wb') as fp:
        #     pickle.dump(myTS.adj_list, fp)


        myValidTS = TrainingSet(maxSize, coarseningStepNum, coarseningLvlNum)
        # Validation set
        for filename in os.listdir(VALID_DATA_PATH):
            if (filename.endswith(".obj")):
                gtfilename = getGTFilename(filename)
                myValidTS.addMeshWithGT(VALID_DATA_PATH,filename,GT_DATA_PATH,gtfilename)

        with open(binDumpPath+'validSet.pkl','wb') as fp:
            pickle.dump(myValidTS,fp)

        # # Validation
        # with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
        #     pickle.dump(myValidTS.in_list, fp)
        # with open(binDumpPath+'valid_GTfn_list', 'wb') as fp:
        #     pickle.dump(myValidTS.gt_list, fp)
        # with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
        #     pickle.dump(myValidTS.adj_list, fp)

    # Train network
    if running_mode == 0:

        
        # # Training
        # with open(binDumpPath+'f_normals_list', 'rb') as fp:
        #     f_normals_list = pickle.load(fp)
        # with open(binDumpPath+'GTfn_list', 'rb') as fp:
        #     GTfn_list = pickle.load(fp)
        # with open(binDumpPath+'f_adj_list', 'rb') as fp:
        #     f_adj_list = pickle.load(fp)
        # # Validation
        # with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
        #     valid_f_normals_list = pickle.load(fp)
        # with open(binDumpPath+'valid_GTfn_list', 'rb') as fp:
        #     valid_GTfn_list = pickle.load(fp)
        # with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
        #     valid_f_adj_list = pickle.load(fp)


        with open(binDumpPath+'trainingSet.pkl', 'rb') as fp:
            myTS = pickle.load(fp)

        # examplesNum = len(f_normals_list)
        # valid_examplesNum = len(valid_f_normals_list)
        # print("training examples num = ",examplesNum)
        # print("f_normals_list shape = ",f_normals_list[0].shape)
        # print("GTfn_list shape = ",GTfn_list[0].shape)
        # print("valid_f_normals_list shape = ",valid_f_normals_list[0].shape)
        # print("valid_GTfn_list shape = ",valid_GTfn_list[0].shape)

        with open(binDumpPath+'validSet.pkl', 'rb') as fp:
            myVS = pickle.load(fp)
        # for p in range(examplesNum):

        #     # First, filter flipped faces for GT normals:
        #     myN = GTfn_list[p][0]
        #     myN = normalize(myN)
        #     myAdj = f_adj_list[p][0][0]

        #     filteredN = filterFlippedFaces(myN, myAdj, printAdjShape=(p==0))
        #     GTfn_list[p] = filteredN[np.newaxis,:,:]

        #     # # Optional: remove border channel for noisy input
        #     # myN = f_normals_list[p][0]
        #     # myNhead = myN[:,:3]
        #     # myNtail = myN[:,4:]
        #     # newN = np.concatenate((myNhead,myNtail),axis=1)
        #     # f_normals_list[p] = newN[np.newaxis,:,:]

        
        
        # for p in range(valid_examplesNum):

        #     # First, filter flipped faces for GT normals:
        #     myN = valid_GTfn_list[p][0]
        #     myN = normalize(myN)
        #     myAdj = valid_f_adj_list[p][0][0]
            
        #     filteredN = filterFlippedFaces(myN, myAdj)
        #     valid_GTfn_list[p] = filteredN[np.newaxis,:,:]

        #     # # Optional: remove border channel for noisy input
        #     # myN = valid_f_normals_list[p][0]
        #     # myNhead = myN[:,:3]
        #     # myNtail = myN[:,4:]
        #     # newN = np.concatenate((myNhead,myNtail),axis=1)
        #     # valid_f_normals_list[p] = newN[np.newaxis,:,:]
        trainNet(myTS,myVS)
        # trainNet(myTS.in_list, myTS.gt_list, myTS.adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list)
        # trainNet(f_normals_list, GTfn_list, f_adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list)

    # Simple inference, no GT mesh involved
    elif running_mode == 1:
        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
        noisyFolder = VALID_DATA_PATH
        # Get GT mesh
        for noisyFile in os.listdir(noisyFolder):

            if (not noisyFile.endswith(".obj")):
                continue
            print("noisyFile: "+noisyFile)

            denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"

            noisyFilesList = [noisyFile]
            denoizedFilesList = [denoizedFile]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                # noisyFileWInferredColor = noisyFile[:-4]+"_inferred_normals.obj"

                noisyFileWInferredColor0 = noisyFile[:-4]+"_fine_normals_s.obj"
                noisyFileWInferredColor1 = noisyFile[:-4]+"_mid_normals_s.obj"
                noisyFileWInferredColor2 = noisyFile[:-4]+"_coarse_normals_s.obj"
                noisyFileWInferredColor3 = noisyFile[:-4]+"_coarse_normals2_s.obj"
                noisyFileWInferredColor4 = noisyFile[:-4]+"_coarse_normals3_s.obj"

                noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
                denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"
                faceMeshFile = noisyFile[:-4]+"_face_mesh.obj"
                faceMeshFile1 = noisyFile[:-4]+"_face_mesh1.obj"
                faceMeshFile2 = noisyFile[:-4]+"_face_mesh2.obj"


                if os.path.isfile(RESULTS_PATH+denoizedFile):
                    if B_OVERWRITE_RESULT:
                        print("Warning: %s will be overwritten. (To deactivate overwriting, change parameter in settings.py)"%denoizedFile)
                    else:
                        print("Skipping %s. File already exists. (For automatic overwriting, change parameter in settings.py)"%denoizedFile)
                        continue


                print("Adding mesh "+noisyFile+"...")
                t0 = time.time()
                inputMesh = InferenceMesh(maxSize, coarseningStepNum, coarseningLvlNum)
                inputMesh.addMeshWithVertices(noisyFolder, noisyFile)
                print("mesh added ("+str(1000*(time.time()-t0))+"ms)")

                faces = inputMesh.faces

                print("Inference ...")
                t0 = time.time()
                upV0, upV0mid, upV0coarse, upN0, upN1, upN2, upP0, upP1, upP2 = inferNet(inputMesh)
            
                # upV0, upN0 = inferNet6D(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
                print("Inference complete ("+str(1000*(time.time()-t0))+"ms)")

                # write_mesh(np.concatenate((upV0,np.zeros_like(upV0)),axis=-1), faces[0,:,:], RESULTS_PATH+denoizedFile)
                write_mesh(upV0, faces, RESULTS_PATH+denoizedFile)
                write_mesh(upV0mid, faces, RESULTS_PATH+noisyFile[:-4]+"_d_mid.obj")
                write_mesh(upV0coarse, faces, RESULTS_PATH+noisyFile[:-4]+"_d_coarse.obj")
                
                # testP = upP0
                # testN = upN0
                # # testP = f_normals_list[0][:,:,3:]
                # # testP = np.squeeze(testP)
                # testAdj = f_adj_list[0][0]
                # testAdj = np.squeeze(testAdj)
                # # testN = f_normals_list[0][:,:,:3]
                # # testN = np.squeeze(testN)
                # faceMeshV, faceMeshF = makeFacesMesh(testAdj,testP,testN)

                # write_mesh(faceMeshV, faceMeshF, RESULTS_PATH+faceMeshFile)
                
                # testP1 = upP1
                # testN1 = upN1
                # testAdj1 = f_adj_list[0][1]
                # testAdj1 = np.squeeze(testAdj1)
                # faceMeshV, faceMeshF = makeFacesMesh(testAdj1,testP1,testN1)

                # write_mesh(faceMeshV, faceMeshF, RESULTS_PATH+faceMeshFile1)

                # testP2 = upP2
                # testN2 = upN2
                # testAdj2 = f_adj_list[0][2]
                # testAdj2 = np.squeeze(testAdj2)
                # faceMeshV, faceMeshF = makeFacesMesh(testAdj2,testP2,testN2)

                # write_mesh(faceMeshV, faceMeshF, RESULTS_PATH+faceMeshFile2)

                V0 = inputMesh.vertices
                faces_noisy = inputMesh.faces
                f_normals0 = inputMesh.normals
                angColor0 = (upN0+1)/2
                angColor1 = (upN1+1)/2
                angColor2 = (upN2+1)/2

                angColorNoisy = (f_normals0+1)/2
                
                print("faces_noisy shape: "+str(faces_noisy.shape))

                print("angColor0 shape: "+str(angColor0.shape))
                print("angColor1 shape: "+str(angColor1.shape))
                print("angColor2 shape: "+str(angColor2.shape))
                print("V0 shape: "+str(V0.shape))
                # newV, newF = getColoredMesh(upV0, faces_gt, angColor)
                newVn0, newFn0 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor0)
                newVn1, newFn1 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor1)
                newVn2, newFn2 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor2)
                # newVn3, newFn3 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor3)
                # newVn4, newFn4 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor4)
                

                # write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
                write_mesh(newVn0, newFn0, RESULTS_PATH+noisyFileWInferredColor0)
                write_mesh(newVn1, newFn1, RESULTS_PATH+noisyFileWInferredColor1)
                write_mesh(newVn2, newFn2, RESULTS_PATH+noisyFileWInferredColor2)
                # write_mesh(newVn3, newFn3, RESULTS_PATH+noisyFileWInferredColor3)
                # write_mesh(newVn4, newFn4, RESULTS_PATH+noisyFileWInferredColor4)

                print("angColorNoisy shape: "+str(angColorNoisy.shape))
                newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_noisy, angColorNoisy)
                write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFileWColor)

    # master branch inference (old school, w/o multi-scale vertex update)
    elif running_mode == 12:
        
        maxSize = MAX_PATCH_SIZE
        patchSize = MAX_PATCH_SIZE

        noisyFolder = VALID_DATA_PATH
        # Get GT mesh
        for noisyFile in os.listdir(noisyFolder):

            denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"

            noisyFilesList = [noisyFile]
            denoizedFilesList = [denoizedFile]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                noisyFileWInferredColor = noisyFile[:-4]+"_inferred_normals.obj"
                noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
                denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"

                # if not denoizedFile.startswith("bunny"):
                #     continue

                if os.path.isfile(RESULTS_PATH+denoizedFile):
                    if B_OVERWRITE_RESULT:
                        print("Warning: %s will be overwritten. (To deactivate overwriting, change parameter in settings.py)"%denoizedFile)
                    else:
                        print("Skipping %s. File already exists. (For automatic overwriting, change parameter in settings.py)"%denoizedFile)
                        continue

                
                print("Adding mesh "+noisyFile+"...")
                t0 = time.time()
                myTS = InferenceMesh(maxSize, coarseningStepNum, coarseningLvlNum)
                myTS.addMesh(noisyFolder, noisyFile)

                print("mesh added ("+str(1000*(time.time()-t0))+"ms)")
                

                print("Inference ...")
                t0 = time.time()

                upV0, upN0 = inferNetOld(myTS)
                print("Inference complete ("+str(1000*(time.time()-t0))+"ms)")

                write_mesh(upV0, myTS.faces, RESULTS_PATH+denoizedFile)

                angColor = (upN0+1)/2

                angColorNoisy = (myTS.normals+1)/2
                
                faces_noisy = myTS.faces
                V0 = myTS.vertices
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

        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_noisy/"
        gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        # # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
        # # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/Kinect_v1/Results/b22/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"
        # # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/test/original/"

        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/noisy/"
        gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"

        # noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # noisyFolder = "/morpheo-nas/marmando/Data/Anja/look-6-2/Results/RVDs/Static/"
        # gtFolder = "/morpheo-nas/marmando/Data/Anja/look-6-2/Results/RVDs/Static/"

        # noisyFolder = "/morpheo-nas2/marmando/marmando_temp/kick540/"
        # gtFolder = "/morpheo-nas2/marmando/marmando_temp/kick540/"


        # results file name
        csv_filename = RESULTS_PATH+"results.csv"
        
        
        angDict = {}
        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("aaaMerlion")):
                continue
            mesh_count = [0]

            # Get all 3 noisy meshes
            noisyFile0 = gtFileName[:-4]+"_noisy.obj"
            # noisyFile0 = gtFileName[:-4]+"_denoized.obj"
            denoizedFile0 = gtFileName[:-4]+"_denoized.obj"


            # noisyFile0 = gtFileName
            # denoizedFile0 = gtFileName[:-4]+"_denoized_synth.obj"
            # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
            

            # noisyFile0 = gtFileName[:-4]+"_n1.obj"
            # noisyFile1 = gtFileName[:-4]+"_n2.obj"
            # noisyFile2 = gtFileName[:-4]+"_n3.obj"

            # denoizedFile0 = gtFileName[:-4]+"_denoized_gray_1.obj"
            # denoizedFile1 = gtFileName[:-4]+"_denoized_gray_2.obj"
            # denoizedFile2 = gtFileName[:-4]+"_denoized_gray_3.obj"

            noisyFilesList = [noisyFile0]
            denoizedFilesList = [denoizedFile0]

            # noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            # denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            isTreated=True
            for fileNum in range(len(denoizedFilesList)):
                if not os.path.isfile(RESULTS_PATH+denoizedFilesList[fileNum]):
                    isTreated=False
            if isTreated:
                continue
            # if (os.path.isfile(RESULTS_PATH+denoizedFile0)) and (os.path.isfile(RESULTS_PATH+denoizedFile1)) and (os.path.isfile(RESULTS_PATH+denoizedFile2)):
            #     continue

            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            GTf_normals = computeFacesNormals(GT, faces_gt)

            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            

            
            for testRep in range(1):
                for fileNum in range(len(denoizedFilesList)):
                    
                    denoizedFile = denoizedFilesList[fileNum]
                    denoizedHeatmap = denoizedFile[:-4]+"_H.obj"
                    noisyFile = noisyFilesList[fileNum]

                    denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"
                    noisyFileWColor = noisyFile[:-4]+"_nC.obj"
                    noisyFileWGtN = noisyFile[:-4]+"_gtnC.obj"
                    gtFileWGtN = gtFileName[:-4]+"_GT.obj"

                    if not os.path.exists(noisyFolder+noisyFile):
                        continue

                    # if not noisyFile.startswith("boy_15"):
                    #     continue

                    # if True:
                    if not os.path.isfile(RESULTS_PATH+denoizedFile):
                    
                    
                        f_normals_list = []
                        GTfn_list = []
                        f_adj_list = []
                        v_list = []
                        gtv_list = []
                        faces_list = []
                        v_faces_list = []

                        print("Adding mesh "+noisyFile+"...")
                        t0 = time.time()
                        faces_num, patch_indices, permutations = addMesh(noisyFolder, noisyFile, gtFolder, gtFileName, f_normals_list, GTfn_list, f_adj_list, mesh_count)
                        # vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list, = addMeshWithVertices(noisyFolder, noisyFile, gtFolder, gtFileName, v_list, gtv_list, faces_list, f_normals_list, f_adj_list, v_faces_list, mesh_count)
                        
                        print("mesh added ("+str(1000*(time.time()-t0))+"ms)")
                        # Now recover vertices positions and create Edge maps
                        V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)


                        V0exp = np.expand_dims(V0, axis=0)

                        depth_diff = GT-V0
                        depth_dir = normalize(depth_diff)

                        # print("Inference ...")
                        t0 = time.time()
                        #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                        # upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                        # upV0, upV0mid, upV0coarse, upN0, upN1, upN2, upP0, upP1, upP2 = inferNet(v_list, faces_list, f_normals_list, f_adj_list, v_faces_list, vOldInd_list, fOldInd_list, vNum, fNum, adjPerm_list, real_nodes_num_list)
                        upV0, upN0 = inferNetOld(V0exp, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum, depth_dir)
                        print("Inference complete ("+str(1000*(time.time()-t0))+"ms)")

                        # print("computing Hausdorff "+str(fileNum+1)+"...")
                        t0 = time.time()
                        # haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                        denseGT = getDensePC(GT, faces_gt, res=1)
                        
                        if gtFileName.startswith("aaarma") or gtFileName.startswith("Merlion"):
                            haus_dist0 = 0
                            avg_dist0 = 0
                        else:
                            haus_dist0, _, avg_dist0, _ = hausdorffOverSampled(upV0, GT, upV0, denseGT, accuracyOnly=True)
                        
                        # print("Hausdorff complete ("+str(1000*(time.clock()-t0))+"ms)")
                        # print("computing Angular diff "+str(fileNum+1)+"...")
                        t0 = time.time()
                        angDistVec = angularDiffVec(upN0, GTf_normals)

                        borderF = getBorderFaces(faces_gt)

                        angDistIn = angDistVec[borderF==0]
                        angDistOut = angDistVec[borderF==1]

                        angDistIn0 = np.mean(angDistIn)
                        angStdIn0 = np.std(angDistIn)
                        angDistOut0 = np.mean(angDistOut)
                        angStdOut0 = np.std(angDistOut)
                        angDist0 = np.mean(angDistVec)
                        angStd0 = np.std(angDistVec)
                        # print("ang dist, std = (%f, %f)"%(angDist0, angStd0))

                        angDist0, angStd0 = angularDiff(upN0, GTf_normals)
                        # print("ang dist, std = (%f, %f)"%(angDist0, angStd0))
                        # print("Angular diff complete ("+str(1000*(time.clock()-t0))+"ms)")
                        # print("max angle: "+str(np.amax(angDistVec)))

                        angDict[noisyFile[:-4]] = angDistVec
                         # --- heatmap ---
                        angColor = angDistVec / HEATMAP_MAX_ANGLE

                        angColor = 1 - angColor
                        angColor = np.maximum(angColor, np.zeros_like(angColor))

                        # print("getting colormap "+str(fileNum+1)+"...")
                        t0 = time.time()
                        colormap = getHeatMapColor(1-angColor)
                        # print("colormap shape: "+str(colormap.shape))
                        newV, newF = getColoredMesh(upV0, faces_gt, colormap)
                        # print("colormap complete ("+str(1000*(time.clock()-t0))+"ms)")
                        #newV, newF = getHeatMapMesh(upV0, faces_gt, angColor)
                        # print("writing mesh...")
                        t0 = time.time()
                        write_mesh(newV, newF, RESULTS_PATH+denoizedHeatmap)
                        print("mesh written ("+str(1000*(time.time()-t0))+"ms)")
                        

                        write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile)


                        finalNormals = computeFacesNormals(upV0, faces_gt)
                        f_normals0 = computeFacesNormals(V0, faces_gt)
                        angColor = (upN0+1)/2
                        angColorFinal = (finalNormals+1)/2
                        angColorNoisy = (f_normals0+1)/2
                        angColorGt = (GTf_normals+1)/2
                   
                        newV, newF = getColoredMesh(upV0, faces_gt, angColorFinal)
                        newVn, newFn = getColoredMesh(V0, faces_gt, angColor)
                        newVnoisy, newFnoisy = getColoredMesh(V0, faces_gt, angColorNoisy)
                        newVgt, newFgt = getColoredMesh(V0, faces_gt, angColorGt)

                        Vgt, Fgt = getColoredMesh(GT, faces_gt, angColorGt)

                        write_mesh(newV, newF, RESULTS_PATH+denoizedFileWColor)
                        write_mesh(newVn, newFn, RESULTS_PATH+noisyFileWColor)
                        write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFile)
                        write_mesh(newVgt, newFgt, RESULTS_PATH+noisyFileWGtN)
                        write_mesh(Vgt, Fgt, RESULTS_PATH+gtFileWGtN)
                        # angColor0 = (upN0+1)/2
                        # newV, newF = getColoredMesh(V0, faces_gt, angColor0)
                        # write_mesh(newV, newF, RESULTS_PATH+denoizedHeatmap)

                        # angColorRaw = (upN0Raw+1)/2
                        # newV, newF = getColoredMesh(V0, faces_gt, angColorRaw)
                        # write_mesh(newV, newF, RESULTS_PATH+denoizedFile[:-4]+"_nRaw.obj")

                        # Fill arrays
                        nameArray.append(denoizedFile)
                        resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum, angDistIn0, angStdIn0, angDistOut0, angStdOut0])

            if not nameArray:
                continue

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
            scipy.io.savemat(RESULTS_PATH+"angDiffRaw.mat",mdict=angDict)


    # Train network with accuracy loss on point sets (rather than normals angular loss)
    elif running_mode == 4:


        # gtnameoffset = 7
        # f_normals_list = []
        # f_adj_list = []
        # v_pos_list = []
        # gtv_pos_list = []
        # faces_list = []
        # v_faces_list = []
        # gtf_normals_list = []

        # valid_f_normals_list = []
        # valid_f_adj_list = []
        # valid_v_pos_list = []
        # valid_gtv_pos_list = []
        # valid_faces_list = []
        # valid_v_faces_list = []
        # valid_gtf_normals_list = []

        # f_normals_list_temp = []
        # f_adj_list_temp = []
        # v_pos_list_temp = []
        # gtv_pos_list_temp = []
        # faces_list_temp = []
        # v_faces_list_temp = []
        # gtf_normals_list_temp = []

        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_train/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Noisy/decim_valid/"

        # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/FAUST/Data/Ground_Truth/"

        # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/valid/"
        # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/noisy/"
        # # validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
        # # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"

        # # inputFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/noisy/"
        # # validFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/valid/"
        # # gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/train/original/"

        # gtnameoffset = 7 #10

        # Training
        pickleNum=0
        while os.path.isfile(binDumpPath+'f_normals_list'+str(pickleNum)):
            with open(binDumpPath+'f_normals_list'+str(pickleNum), 'rb') as fp:
                # f_normals_list_temp = pickle.load(fp, encoding='latin1')
                f_normals_list_temp = pickle.load(fp)
            with open(binDumpPath+'f_adj_list'+str(pickleNum), 'rb') as fp:
                # f_adj_list_temp = pickle.load(fp, encoding='latin1')
                f_adj_list_temp = pickle.load(fp)
            with open(binDumpPath+'v_pos_list'+str(pickleNum), 'rb') as fp:
                # v_pos_list_temp = pickle.load(fp, encoding='latin1')
                v_pos_list_temp = pickle.load(fp)
            with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'rb') as fp:
                # gtv_pos_list_temp = pickle.load(fp, encoding='latin1')
                gtv_pos_list_temp = pickle.load(fp)
            with open(binDumpPath+'faces_list'+str(pickleNum), 'rb') as fp:
                # faces_list_temp = pickle.load(fp, encoding='latin1')
                faces_list_temp = pickle.load(fp)
            with open(binDumpPath+'v_faces_list'+str(pickleNum), 'rb') as fp:
                # v_faces_list_temp = pickle.load(fp, encoding='latin1')
                v_faces_list_temp = pickle.load(fp)
            with open(binDumpPath+'gtf_normals_list'+str(pickleNum), 'rb') as fp:
                # f_normals_list_temp = pickle.load(fp, encoding='latin1')
                gtf_normals_list_temp = pickle.load(fp)

            if pickleNum==0:
                f_normals_list = f_normals_list_temp
                f_adj_list = f_adj_list_temp
                v_pos_list = v_pos_list_temp
                gtv_pos_list = gtv_pos_list_temp
                faces_list = faces_list_temp
                v_faces_list = v_faces_list_temp
                gtf_normals_list = gtf_normals_list_temp
            else:

                f_normals_list+=f_normals_list_temp
                f_adj_list+=f_adj_list_temp
                v_pos_list+=v_pos_list_temp
                gtv_pos_list+=gtv_pos_list_temp
                faces_list+=faces_list_temp
                v_faces_list+=v_faces_list_temp
                gtf_normals_list+=gtf_normals_list_temp


            print("loaded training pickle "+str(pickleNum))
            pickleNum+=1


        # Validation
        with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
            # valid_f_normals_list = pickle.load(fp, encoding='latin1')
            valid_f_normals_list = pickle.load(fp)
        with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
            # valid_f_adj_list = pickle.load(fp, encoding='latin1')
            valid_f_adj_list = pickle.load(fp)
        with open(binDumpPath+'valid_v_pos_list', 'rb') as fp:
            # valid_v_pos_list = pickle.load(fp, encoding='latin1')
            valid_v_pos_list = pickle.load(fp)
        with open(binDumpPath+'valid_gtv_pos_list', 'rb') as fp:
            # valid_gtv_pos_list = pickle.load(fp, encoding='latin1')
            valid_gtv_pos_list = pickle.load(fp)
        with open(binDumpPath+'valid_faces_list', 'rb') as fp:
            # valid_faces_list = pickle.load(fp, encoding='latin1')
            valid_faces_list = pickle.load(fp)
        with open(binDumpPath+'valid_v_faces_list', 'rb') as fp:
            # valid_v_faces_list = pickle.load(fp, encoding='latin1')
            valid_v_faces_list = pickle.load(fp)
        with open(binDumpPath+'valid_gtf_normals_list', 'rb') as fp:
            # valid_f_normals_list = pickle.load(fp, encoding='latin1')
            valid_gtf_normals_list = pickle.load(fp)

        print("f_normals_list length = ",len(f_normals_list))
        print("gtf_normals_list length = ",len(gtf_normals_list))
        print("valid_f_normals_list length = ",len(valid_f_normals_list))
        print("valid_gtf_normals_list length = ",len(valid_gtf_normals_list))

        print(" f_normals_list shape = ",f_normals_list[0].shape)
        print(" gtf_normals_list shape = ",gtf_normals_list[0].shape)
        print(" valid_f_normals_list shape = ",valid_f_normals_list[0].shape)
        print(" valid_gtf_normals_list shape = ",valid_gtf_normals_list[0].shape)


        # examplesNum = len(f_normals_list)

        # for p in range(examplesNum):
        #     myN = gtf_normals_list[p][0]
        #     myN = normalize(myN)
        #     myAdj = f_adj_list[p][0][0]
        #     filteredN = filterFlippedFaces(myN, myAdj)
        #     gtf_normals_list[p] = filteredN[np.newaxis,:,:]

        # valid_examplesNum = len(valid_f_normals_list)
        # for p in range(valid_examplesNum):
        #     myN = valid_gtf_normals_list[p][0]
        #     myN = normalize(myN)
        #     myAdj = valid_f_adj_list[p][0][0]
        #     filteredN = filterFlippedFaces(myN, myAdj)
        #     valid_gtf_normals_list[p] = filteredN[np.newaxis,:,:]

        for myI in range(len(f_normals_list)):
            nrmls = f_normals_list[myI]
            print("nrmls shape = ",nrmls.shape)
            adj = f_adj_list[myI]
            print("adj[0] shape = ",adj[0].shape)

        # trainAccuracyNet(v_pos_list, gtv_pos_list, faces_list, f_normals_list, f_adj_list, v_faces_list, valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_f_adj_list, valid_v_faces_list)
        trainDoubleLossNet(v_pos_list, gtv_pos_list, faces_list, f_normals_list, gtf_normals_list, f_adj_list, v_faces_list, valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_gtf_normals_list, valid_f_adj_list, valid_v_faces_list)
        # trainAccuracyNet(valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list, valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list)

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

            _, edge_map, v_e_map = getFacesAdj(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
            

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

                oldToNew = inv_perm(newToOld)
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

        gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/rescaled_gt/"
        # gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v2/test/original/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results_heat.csv"
        
        angDict={}
        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (gtFileName.endswith("Merlion.obj")):
                continue

            # denoizedFile0 = gtFileName[:-4]+"_n1-dtree3.obj"
            # denoizedFile1 = gtFileName[:-4]+"_n2-dtree3.obj"
            # denoizedFile2 = gtFileName[:-4]+"_n3-dtree3.obj"

            denoizedFile0 = gtFileName[:-4]+"_n1_nllr_default.obj"
            denoizedFile1 = gtFileName[:-4]+"_n2_nllr_default.obj"
            denoizedFile2 = gtFileName[:-4]+"_n3_nllr_default.obj"

            # denoizedFile0 = gtFileName[:-4]+"_noisy-dtree3.obj"
            # denoizedFile0 = gtFileName[:-4]+"_denoized.obj"
            # heatFile0 = gtFileName[:-4]+"_heatmap.obj"

            # denoizedFile0 = gtFileName[:-4]+"_denoized_gray_1.obj"
            # denoizedFile1 = gtFileName[:-4]+"_denoized_gray_2.obj"
            # denoizedFile2 = gtFileName[:-4]+"_denoized_gray_3.obj"

            # denoizedFile0 = gtFileName[:-4]+"_n1_rescaled.obj"
            # denoizedFile1 = gtFileName[:-4]+"_n2_rescaled.obj"
            # denoizedFile2 = gtFileName[:-4]+"_n3_rescaled.obj"

            # denoizedFile0 = gtFileName[:-4]+"_n1_denoised_gray.obj"
            # denoizedFile1 = gtFileName[:-4]+"_n2_denoised_gray.obj"
            # denoizedFile2 = gtFileName[:-4]+"_n3_denoised_gray.obj"

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
            denseGT = getDensePC(GT, faces_gt, res=1)
            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            #edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            denoizedFilesList = [denoizedFile0]
            heatMapFilesList = [heatFile0]

            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]
            heatMapFilesList = [heatFile0,heatFile1,heatFile2]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                heatFile = heatMapFilesList[fileNum]
                
                if not os.path.isfile(RESULTS_PATH+heatFile):
                    
                    V0,_,_, _, _ = load_mesh(RESULTS_PATH, denoizedFile, 0, False)
                    f_normals0 = computeFacesNormals(V0, faces_gt)

                    print("computing Hausdorff "+ denoizedFile + " " + str(fileNum+1)+"...")
                    # haus_dist0, avg_dist0 = oneSidedHausdorff(V0, GT)

                    
                    haus_dist0, _, avg_dist0, _ = hausdorffOverSampled(V0, GT, V0, denseGT, accuracyOnly=True)

                    angDistVec = angularDiffVec(f_normals0, GTf_normals)

                    borderF = getBorderFaces(faces_gt)

                    angDistIn = angDistVec[borderF==0]
                    angDistOut = angDistVec[borderF==1]

                    angDistIn0 = np.mean(angDistIn)
                    angStdIn0 = np.std(angDistIn)
                    angDistOut0 = np.mean(angDistOut)
                    angStdOut0 = np.std(angDistOut)
                    angDist0 = np.mean(angDistVec)
                    angStd0 = np.std(angDistVec)

                    angDistSquare = np.mean(np.square(angDistVec))
                    angDistVecRad = angDistVec*math.pi/180
                    angDistSquareRad = np.mean(np.square(angDistVecRad))
                    print("angDistSquare " + denoizedFile + " = %f"%angDistSquare)
                    print("angDistSquareRad " + denoizedFile + " = %f"%angDistSquareRad)
                    print("sqrt angDistSquare " + denoizedFile + " = %f"%np.sqrt(angDistSquare))

                    #print("ang dist, std = (%f, %f)"%(angDist0, angStd0))

                    angDist0, angStd0 = angularDiff(f_normals0, GTf_normals)
                    print("max angle: "+str(np.amax(angDistVec)))
                    dictLabel = denoizedFile[:-4]
                    dictLabel = dictLabel.replace('-','_')
                    angDict[dictLabel] = angDistVec
                    # --- Test heatmap ---
                    angColor = angDistVec / HEATMAP_MAX_ANGLE
                    angColor = 1 - angColor
                    angColor = np.maximum(angColor, np.zeros_like(angColor))

                    colormap = getHeatMapColor(1-angColor)
                    newV, newF = getColoredMesh(V0, faces_gt, colormap)

                    # newV, newF = getHeatMapMesh(V0, faces_gt, angColor)

                    write_mesh(newV, newF, RESULTS_PATH+heatFile)
                    
                    # Fill arrays
                    nameArray.append(denoizedFile)
                    resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum, angDistIn0, angStdIn0, angDistOut0, angStdOut0])

            if not nameArray:
                continue
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

            scipy.io.savemat(RESULTS_PATH+"angDiffFinal.mat",mdict=angDict)

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

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        noisyFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/noisy/"
        gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_Fusion/test/original/"

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

            noisyFile0 = gtFileName[:-4]+"_noisy.obj"
            denoizedFile0 = gtFileName[:-4]+"_denoized.obj"
            noisyFileWColor0 = gtFileName[:-4]+"_nC.obj"

            # Get all 3 noisy meshes
            # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
            
            # noisyFile0 = gtFileName[:-4]+"_n1.obj"
            # noisyFile1 = gtFileName[:-4]+"_n2.obj"
            # noisyFile2 = gtFileName[:-4]+"_n3.obj"

            # noisyFileWColor0 = gtFileName[:-4]+"_n1C.obj"
            # noisyFileWColor1 = gtFileName[:-4]+"_n2C.obj"
            # noisyFileWColor2 = gtFileName[:-4]+"_n3C.obj"

            # denoizedFile0 = gtFileName[:-4]+"_denoizedC_1.obj"
            # denoizedFile1 = gtFileName[:-4]+"_denoizedC_2.obj"
            # denoizedFile2 = gtFileName[:-4]+"_denoizedC_3.obj"


            # if (os.path.isfile(RESULTS_PATH+denoizedFile0)):
            if (os.path.isfile(RESULTS_PATH+denoizedFile0)) and (os.path.isfile(RESULTS_PATH+denoizedFile1)) and (os.path.isfile(RESULTS_PATH+denoizedFile2)):
                continue

            # Load GT mesh
            GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
            GTf_normals = computeFacesNormals(GT, faces_gt)

            facesNum = faces_gt.shape[0]
            # We only need to load faces once. Connectivity doesn't change for noisy meshes
            # Same for adjacency matrix

            _, edge_map, v_e_map = getFacesAdj(faces_gt)
            f_adj = getFacesLargeAdj(faces_gt,K_faces)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            
            noisyFilesList = [noisyFile0]
            noisyFilesWColorList = [noisyFileWColor0]
            denoizedFilesList = [denoizedFile0]

            # noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            # noisyFilesWColorList = [noisyFileWColor0,noisyFileWColor1,noisyFileWColor2]
            # denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

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
                    V0exp = np.expand_dims(V0, axis=0)

                    print("running ...")
                    #upV0, upN0 = inferNet(V0, GTfn_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    

                    depth_diff = GT-V0
                    depth_dir = normalize(depth_diff)


                    # upV0, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum)
                    upV0, upN0 = inferNetOld(V0exp, f_normals_list, f_adj_list, edge_map, v_e_map,faces_num, patch_indices, permutations,facesNum, depth_dir)
                    
                    
                    print("computing Hausdorff "+str(fileNum+1)+"...")
                    haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                    angDistVec = angularDiffVec(upN0, GTf_normals)
                    angDist0, angStd0 = angularDiff(upN0, GTf_normals)
                    print("max angle: "+str(np.amax(angDistVec)))

                    finalNormals = computeFacesNormals(upV0, faces_gt)
                    angColor = (upN0+1)/2
                    angColorFinal = (finalNormals+1)/2
                    angColorNoisy = (f_normals0+1)/2
               
                    newV, newF = getColoredMesh(upV0, faces_gt, angColorFinal)
                    newVn, newFn = getColoredMesh(V0, faces_gt, angColor)
                    newVnoisy, newFnoisy = getColoredMesh(V0, faces_gt, angColorNoisy)

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


    # Load and pickle training data.
    elif running_mode == 11:

        myTS = TrainingSet(maxSize, coarseningStepNum, coarseningLvlNum)
        myValidSet = TrainingSet(maxSize,coarseningStepNum, coarseningLvlNum)

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_v_pos_list = []
        valid_gtv_pos_list = []
        valid_faces_list = []
        valid_v_faces_list = []
        valid_gtf_normals_list = []

        f_normals_list_temp = []
        f_adj_list_temp = []
        v_pos_list_temp = []
        gtv_pos_list_temp = []
        faces_list_temp = []
        v_faces_list_temp = []
        gtf_normals_list_temp = []
        
        gtFilePath = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/gt/"

        pickleNum=0

        # Training set
        for filename in os.listdir(TRAINING_DATA_PATH):
            #print("training_meshes_num start_iter " + str(training_meshes_num))
            if training_meshes_num[0]>30:
                break
            if myTS.mesh_count>10:
                break

            if (filename.endswith(".obj")):
                print("Adding %s (%i)"%(filename, training_meshes_num[0]))
                gtfilename = getGTFilename(filename)
                
                myTS.addMeshWithVerticesAndGT(TRAINING_DATA_PATH, filename, GT_DATA_PATH, gtfilename)

                # addMeshWithVertices(TRAINING_DATA_PATH, filename, GT_DATA_PATH, gtfilename, v_pos_list_temp, gtv_pos_list_temp, f_normals_list_temp, f_adj_list_temp, e_map_list_temp, v_emap_list_temp, training_meshes_num)
                # addMeshWithVertices(TRAINING_DATA_PATH, filename, GT_DATA_PATH, gtfilename, v_pos_list_temp, gtv_pos_list_temp, faces_list_temp, f_normals_list_temp, gtf_normals_list_temp, f_adj_list_temp, v_faces_list_temp, training_meshes_num)



                # # Save batches of meshes/patches (for training only)
                # if training_meshes_num[0]>30:
                #     # Training
                #     with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(f_normals_list_temp, fp)
                #     with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(f_adj_list_temp, fp)
                #     with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(v_pos_list_temp, fp)
                #     with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(gtv_pos_list_temp, fp)
                #     with open(binDumpPath+'faces_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(faces_list_temp, fp)
                #     with open(binDumpPath+'v_faces_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(v_faces_list_temp, fp)
                #     with open(binDumpPath+'gtf_normals_list'+str(pickleNum), 'wb') as fp:
                #         pickle.dump(gtf_normals_list_temp, fp)
                #     pickleNum+=1

                #     f_normals_list_temp = []
                #     f_adj_list_temp = []
                #     v_pos_list_temp = []
                #     gtv_pos_list_temp = []
                #     faces_list_temp = []
                #     v_faces_list_temp = []
                #     gtf_normals_list_temp = []
                #     training_meshes_num[0] = 0

        with open(binDumpPath+'trainingSetWithVertices.pkl'+str(pickleNum), 'wb') as fp:
                pickle.dump(myTS, fp)


        # if training_meshes_num[0]>0:
        #     # Training
        #     with open(binDumpPath+'f_normals_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(f_normals_list_temp, fp)
        #     with open(binDumpPath+'f_adj_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(f_adj_list_temp, fp)
        #     with open(binDumpPath+'v_pos_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(v_pos_list_temp, fp)
        #     with open(binDumpPath+'gtv_pos_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(gtv_pos_list_temp, fp)
        #     with open(binDumpPath+'faces_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(faces_list_temp, fp)
        #     with open(binDumpPath+'v_faces_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(v_faces_list_temp, fp)
        #     with open(binDumpPath+'gtf_normals_list'+str(pickleNum), 'wb') as fp:
        #         pickle.dump(gtf_normals_list_temp, fp)



        # Validation set
        for filename in os.listdir(VALID_DATA_PATH):
            if (filename.endswith(".obj")):
                gtfilename = getGTFilename(filename)
                
                myValidSet.addMeshWithVerticesAndGT(VALID_DATA_PATH, filename, GT_DATA_PATH, gtfilename)
                # addMeshWithVertices(VALID_DATA_PATH, filename, GT_DATA_PATH, gtfilename, valid_v_pos_list, valid_gtv_pos_list, valid_f_normals_list, valid_f_adj_list, valid_e_map_list, valid_v_emap_list, valid_meshes_num)
                # addMeshWithVertices(VALID_DATA_PATH, filename, GT_DATA_PATH, gtfilename, valid_v_pos_list, valid_gtv_pos_list, valid_faces_list, valid_f_normals_list, valid_gtf_normals_list, valid_f_adj_list, valid_v_faces_list, valid_meshes_num)
        
        # Validation
        with open(binDumpPath+'validSetWithVertices.pkl'+str(pickleNum), 'wb') as fp:
            pickle.dump(myTS, fp)

        # with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
        #     pickle.dump(valid_f_normals_list, fp)
        # with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
        #     pickle.dump(valid_f_adj_list, fp)
        # with open(binDumpPath+'valid_v_pos_list', 'wb') as fp:
        #     pickle.dump(valid_v_pos_list, fp)
        # with open(binDumpPath+'valid_gtv_pos_list', 'wb') as fp:
        #     pickle.dump(valid_gtv_pos_list, fp)
        # with open(binDumpPath+'valid_faces_list', 'wb') as fp:
        #     pickle.dump(valid_faces_list, fp)
        # with open(binDumpPath+'valid_v_faces_list', 'wb') as fp:
        #     pickle.dump(valid_v_faces_list, fp)
        # with open(binDumpPath+'valid_gtf_normals_list', 'wb') as fp:
        #     pickle.dump(valid_gtf_normals_list, fp)




    print("Complete: mode = "+str(RUNNING_MODE)+", architecture "+str(ARCHITECTURE)+", net path = "+NETWORK_PATH)
    #

if __name__ == "__main__":
    
    print("Tensorflow version = ", tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=int, default=0)
    #parser.add_argument('--dataset_path')
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--num_iterations', type=int, default=20000)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--net_name', type=str, default='net')
    parser.add_argument('--running_mode', type=int, default=0)
    parser.add_argument('--coarsening_steps', type=int, default=2)
    

    #parser.add_argument('--num_classes', type=int)

    FLAGS = parser.parse_args()

    ARCHITECTURE = FLAGS.architecture
    #DATASET_PATH = FLAGS.dataset_path
    
    # Override default results path if specified as command parameter
    if not FLAGS.results_path is None:
        RESULTS_PATH = FLAGS.results_path

    if not FLAGS.network_path is None:
        NETWORK_PATH = FLAGS.network_path

    NUM_ITERATIONS = FLAGS.num_iterations
    DEVICE = FLAGS.device
    NET_NAME = FLAGS.net_name
    RUNNING_MODE = FLAGS.running_mode
    # COARSENING_STEPS = FLAGS.coarsening_steps

    # Experimental value on synthetic dataset:
    AVG_EDGE_LENGTH = 0.005959746586165783

    #NUM_CLASSES = FLAGS.num_classes

    mainFunction()


