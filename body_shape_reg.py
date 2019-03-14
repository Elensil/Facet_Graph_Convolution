from __future__ import division
import numpy as np
import math
import time
#import h5py
import argparse
import os
import pickle

import random
from lib.coarsening import *
import scipy.io
from scipy import ndimage
import shutil

import matplotlib.pyplot as plt

trainMode=True
if trainMode:
    import tensorflow as tf
    # from model_d2 import *
    from model import *
    from utils import *
    from tensorflow.python import debug as tf_debug
else:
    from PIL import Image
    import detectron.utils.densepose_methods as dp_utils

def trainBodyShapeNet(input_list, gtshape_list, graph_adj, perm, face_pos, nodeParamDisp, valid_input_list, valid_gtshape_list, mode='gender'):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


    EXAMPLE_NUM = len(input_list)
    VALID_EXAMPLE_NUM = len(valid_input_list)

    print("EXAMPLE_NUM = "+str(EXAMPLE_NUM))
    
    NUM_IN_CHANNELS = input_list[0].shape[2]
    # NUM_IN_CHANNELS = 29

    BATCH_SIZE = 6

    graph_card = np.zeros(len(graph_adj), dtype=np.int32)
    K_faces = np.zeros(len(graph_adj), dtype=np.int32)
    for g in range(len(graph_adj)):
        print("graph_adj["+str(g)+"] shape = "+str(graph_adj[g].shape))
        graph_card[g] = graph_adj[g].shape[1]
        K_faces[g] = graph_adj[g].shape[2]

    # K_faces = graph_adj[0].shape[2]

    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE,TEMPLATE_CARD, NUM_IN_CHANNELS], name='fn_')
    node_pos = tf.placeholder(tf.float32, shape=[1, TEMPLATE_CARD, 3], name='node_pos')

    fadj0 = tf.placeholder(tf.int32, shape=[1,graph_card[0], K_faces[0]], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[1,graph_card[1], K_faces[1]], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[1,graph_card[2], K_faces[2]], name='fadj2')
    fadj3 = tf.placeholder(tf.int32, shape=[1,graph_card[3], K_faces[3]], name='fadj2')

    perm_ = tf.placeholder(tf.int32, shape=[graph_card[0]], name='perm')
    gtshapegender_ = tf.placeholder('float32', shape=[BATCH_SIZE,11], name='tfn_')

    nodeRawWeights_ = tf.placeholder('float32', shape=[TEMPLATE_CARD, 10], name='nodeRawWeights_')

    node_pos_tiled = tf.tile(node_pos,[BATCH_SIZE,1,1])

    print("graph_card = "+str(graph_card))
    print("TEMPLATE_CARD = "+str(TEMPLATE_CARD))

    empty_and_fake_nodes = tf.equal(fn_[:,:,0],-1)
    print("empty_and_fake_nodes shape = "+str(empty_and_fake_nodes.shape))

    empty_and_fake_nodes = tf.tile(tf.expand_dims(empty_and_fake_nodes,-1),[1,1,10])

    batchNodeRawWeights = tf.tile(tf.expand_dims(nodeRawWeights_,axis=0), [BATCH_SIZE,1,1])

    nodeFilteredWeights = tf.where(empty_and_fake_nodes,tf.zeros_like(batchNodeRawWeights,dtype=tf.float32),batchNodeRawWeights)
    # [batch, N, 10]
    batchWeights = tf.reduce_sum(nodeFilteredWeights, axis=1)
    # [batch, 10]

    new_fn = fn_

    fn_centered = 2 * new_fn - 1

    fullfn = tf.concat((fn_centered,node_pos_tiled), axis=-1)

    padding = tf.zeros([BATCH_SIZE,graph_card[0]-TEMPLATE_CARD,NUM_IN_CHANNELS+3])
    padding = padding-3
    ext_fn = tf.concat((fullfn, padding), axis=1)
    perm_fn = tf.gather(ext_fn,perm_, axis=1)

    gtgender = gtshapegender_[:,0]
    gtshape = gtshapegender_[:,1:]

    # ext_shapegender = tf.concatenate((gtshapegender_, tf.zeros([graph_card[0]-TEMPLATE_CARD,11])), axis=0)
    # perm_shapegender = tf.gather(ext_shapegender,perm_)

    # sample_ind = tf.placeholder(tf.int32, shape=[10000], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)

    fadjs = [fadj0,fadj1,fadj2,fadj3]

    if mode=='gender':
        with tf.variable_scope("model_gender_class"):
            pmale, _ = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode='gender')

        with tf.device(DEVICE):
            print("pmale shape = "+str(pmale.shape))
            print("gtgender shape = "+str(gtgender.shape))
            
            totalLogl = loglikelihood(tf.reshape(pmale,[-1]),gtgender)
            crossEntropyLoss = -totalLogl
            train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
            
            isNanPmale = tf.reduce_any(tf.is_nan(pmale), name="isNanPmale")
            
    elif mode=='shape':

        with tf.variable_scope("model_gender_class"):
            predicted_shape, _ = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob)
            
        with tf.device(DEVICE):
            print("predicted_shape shape = "+str(predicted_shape.shape))
            print("gtshape shape = "+str(gtshape.shape))
            # shapeLoss = mseLoss(predicted_shape,gtshape)
            # shapeLoss = mseLossWeighted(predicted_shape,gtshape,batchWeights)            
            print("WARNING!!!! Hard-coded loss change (1st param)")
            shapeLoss = mseLoss(predicted_shape[:,:1],gtshape[:,:1])
            print("predicted_shape slice shape = "+str((predicted_shape[:,:1]).shape))
            print("shapeLoss shape = "+str(shapeLoss.shape))

            crossEntropyLoss = shapeLoss
            train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
            
            isNanPmale = tf.reduce_any(tf.is_nan(predicted_shape), name="isNanPmale")

    elif mode=='bodyratio':
        tfH0 = tf.constant(H0, shape=[1,1], dtype=tf.float32)
        tfW0 = tf.constant(W0, shape=[1,1], dtype=tf.float32)
        tfD0 = tf.constant(D0, shape=[1,1], dtype=tf.float32)
        tfHMat = tf.constant(HMat, shape=[10,1], dtype=tf.float32)
        tfWMat = tf.constant(WMat, shape=[10,1], dtype=tf.float32)
        tfDMat = tf.constant(DMat, shape=[10,1], dtype=tf.float32)

        bH = tf.matmul(gtshape,tfHMat)
        bW = tf.matmul(gtshape,tfWMat)
        bD = tf.matmul(gtshape,tfDMat)
        print("tfHMat shape = "+str(tfHMat.shape))
        print("bH shape = "+str(bH.shape))
        # [batch, 1]
        bH = bH + tfH0
        bW = bW + tfW0
        bD = bD + tfD0
        print("bH shape = "+str(bH.shape))

        new_param = tf.divide(bD,bH)
        # [batch, 1]
        print("new_param shape = "+str(new_param.shape))
        with tf.variable_scope("model_gender_class"):
            predicted_shape, _ = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode)
        with tf.device(DEVICE):
            print("predicted_shape shape = "+str(predicted_shape.shape))
            print("gtshape shape = "+str(gtshape.shape))            
            print("WARNING!!!! Hard-coded loss change (homemade param)")
            shapeLoss = mseLoss(predicted_shape,new_param)
            print("shapeLoss shape = "+str(shapeLoss.shape))
            crossEntropyLoss = shapeLoss
            train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
            
            isNanPmale = tf.reduce_any(tf.is_nan(predicted_shape), name="isNanPmale")

    batch_loss = tf.reduce_mean(crossEntropyLoss)

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
    

    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    # Training

    train_loss=0
    train_samp=0

    SAVEITER = 1000

    evalStepNum=20

    with tf.device(DEVICE):
        # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        lossArray = np.zeros([int(SAVEITER/evalStepNum),2])
        last_loss = 0
        lossArrayIter = 0
        for iter in range(NUM_ITERATIONS):

            # Batch version...
            inbatchlist = []
            gtbatchlist = []
            for b in range(BATCH_SIZE):
                batch_num = random.randint(0,EXAMPLE_NUM-1)

                if mode=='shape' or mode=='bodyratio':
                    # LIMIT TO MALE EXAMPLES
                    while gtshape_list[batch_num][0,0]==0:
                        batch_num = random.randint(0,EXAMPLE_NUM-1)

                inbatchlist.append(input_list[batch_num])
                gtbatchlist.append(gtshape_list[batch_num])


            in_batch = np.concatenate(inbatchlist, axis=0)
            shape_batch = np.concatenate(gtbatchlist, axis=0)

            # train_fd = {fn_: input_list[batch_num], gtshapegender_: gtshape_list[batch_num], keep_prob:1}
            train_fd = {fn_: in_batch, gtshapegender_: shape_batch, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, nodeRawWeights_:nodeParamDisp, keep_prob:0.8}


            # One-example version
                    # # Get random sample from training dictionary
                    # batch_num = random.randint(0,EXAMPLE_NUM-1)

                    # train_fd = {fn_: input_list[batch_num], fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], gtshapegender_: gtshape_list[batch_num], perm_: perm, node_pos:face_pos, keep_prob:1}

            # Show smoothed training loss
            if ((iter%evalStepNum == 0) and (iter>0)):
                train_loss = train_loss/train_samp
                

                print("Iteration %d, training loss %g"%(iter, train_loss))

                # if mode=='gender':
                #     pred_pmale = sess.run(pmale, feed_dict=train_fd)
                #     print("Predicted male probability = "+str(pred_pmale) + ", Ground truth = " + str(shape_batch[:,0])) 

                lossArray[lossArrayIter,0]=train_loss
                lossArrayIter+=1
                train_loss=0
                train_samp=0
                # pred_pmale, valGender,valYp, valYgt, valLogYp = sess.run([pmale, gtgender, yp, ygt, testLogYp], feed_dict=train_fd)
                # print("pred_pmale shape = "+str(pred_pmale.shape))
                # print("pred_pmale[0] = "+str(pred_pmale[0]))
                # pred_pmale_val = pred_pmale[0]
                # print("Predicted male probability = %g, Ground truth = %g"%(pred_pmale_val, gtshape_list[batch_num][0,0]))
                # print("invalues: (yp = %g, ygt = %g). log(yp) = %g"%(valYp,valYgt, valLogYp))

            
            # marg_train_loss = crossEntropyLoss.eval(feed_dict=train_fd)

            marg_train_loss = sess.run(batch_loss, feed_dict=train_fd)
            train_loss += marg_train_loss
            train_samp+=1

            if iter==0:
                myinput = sess.run(perm_fn,feed_dict=train_fd)
                myinputsamp = myinput[0,:,:]
                mynodepos = myinputsamp[:,-3:]
                mynodecolor=myinputsamp[:,:2]
                mynodecolor=np.tile(mynodecolor,[1,2])
                mynodecolor=(mynodecolor[:,:3]+1)/2
                # print("node colors min & max: (%f, %f)"%(np.amin(mynodecolor),np.amax(mynodecolor)))
                mynodecolor = np.where(mynodecolor<0,0,mynodecolor)
                # print("node colors min & max: (%f, %f)"%(np.amin(mynodecolor),np.amax(mynodecolor)))
                mynodecolor = mynodecolor * 255
                # print("node colors min & max: (%f, %f)"%(np.amin(mynodecolor),np.amax(mynodecolor)))
                mynodecolor = mynodecolor.astype(int)
                mynodecolor = mynodecolor.astype(float)
                xyzarray = np.concatenate((mynodepos,mynodecolor),axis=1)
                write_coff(xyzarray,"/morpheo-nas2/marmando/ShapeRegression/Test/input_test.off")


            # Compute validation loss
            if True:
                if (iter%(evalStepNum*2) ==0) and (iter>0):
                    valid_loss = 0
                    for vbm in range(VALID_EXAMPLE_NUM):
                        
                        valid_inbatchlist = []
                        valid_gtbatchlist = []
                        for b in range(BATCH_SIZE):
                            batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                            if mode=='shape' or mode=='bodyratio':
                                # LIMIT TO MALE EXAMPLES
                                while valid_gtshape_list[batch_num][0,0]==0:
                                    batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                            valid_inbatchlist.append(valid_input_list[batch_num])
                            valid_gtbatchlist.append(valid_gtshape_list[batch_num])
                            # valid_inbatchlist.append(valid_input_list[b])
                            # valid_gtbatchlist.append(valid_gtshape_list[b])



                        valid_in_batch = np.concatenate(valid_inbatchlist, axis=0)
                        valid_shape_batch = np.concatenate(valid_gtbatchlist, axis=0)
                        
                        valid_fd = {fn_: valid_in_batch, gtshapegender_: valid_shape_batch, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, nodeRawWeights_:nodeParamDisp, keep_prob:1}


                        # valid_fd = {fn_: valid_input_list[vbm], fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], gtshapegender_: valid_gtshape_list[vbm], perm_: perm, node_pos:face_pos, keep_prob:1}

                        # valid_loss += crossEntropyLoss.eval(feed_dict=valid_fd)
                        valid_loss += sess.run(batch_loss,feed_dict=valid_fd)
                    valid_loss/=VALID_EXAMPLE_NUM
                    print("Iteration %d, validation loss %g\n"%(iter, valid_loss))
                    lossArray[lossArrayIter-1,1]=valid_loss
                    if iter>0:
                        lossArray[lossArrayIter-2,1] = (valid_loss+last_loss)/2
                        last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            
            if sess.run(isNanPmale,feed_dict=train_fd):
                    hasNan = True
                    print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(input_list)))
                    return
            
            if (iter%SAVEITER == 0) and (iter>0):
                saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
                f = open(csv_filename,'ab')
                np.savetxt(f,lossArray, delimiter=",")
                f.close()
                lossArray = np.zeros([int(SAVEITER/evalStepNum),2]) 
                lossArrayIter=0
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()



def trainMeshEncoder(input_list, gtshape_list, graph_adj, perm, face_pos, nodeParamDisp, valid_input_list, valid_gtshape_list, mode='gender'):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


    EXAMPLE_NUM = len(input_list)
    VALID_EXAMPLE_NUM = len(valid_input_list)

    print("EXAMPLE_NUM = "+str(EXAMPLE_NUM))
    
    NUM_IN_CHANNELS = input_list[0].shape[2]
    # NUM_IN_CHANNELS = 29

    BATCH_SIZE = 6

    graph_card = np.zeros(len(graph_adj), dtype=np.int32)
    K_faces = np.zeros(len(graph_adj), dtype=np.int32)
    for g in range(len(graph_adj)):
        print("graph_adj["+str(g)+"] shape = "+str(graph_adj[g].shape))
        graph_card[g] = graph_adj[g].shape[1]
        K_faces[g] = graph_adj[g].shape[2]

    # K_faces = graph_adj[0].shape[2]

    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE,TEMPLATE_CARD, NUM_IN_CHANNELS], name='fn_')
    node_pos = tf.placeholder(tf.float32, shape=[1, TEMPLATE_CARD, 3], name='node_pos')

    fadj0 = tf.placeholder(tf.int32, shape=[1,graph_card[0], K_faces[0]], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[1,graph_card[1], K_faces[1]], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[1,graph_card[2], K_faces[2]], name='fadj2')
    fadj3 = tf.placeholder(tf.int32, shape=[1,graph_card[3], K_faces[3]], name='fadj2')

    perm_ = tf.placeholder(tf.int32, shape=[graph_card[0]], name='perm')
    gtshapegender_ = tf.placeholder('float32', shape=[BATCH_SIZE,11], name='tfn_')

    nodeRawWeights_ = tf.placeholder('float32', shape=[TEMPLATE_CARD, 10], name='nodeRawWeights_')

    node_pos_tiled = tf.tile(node_pos,[BATCH_SIZE,1,1])

    print("graph_card = "+str(graph_card))
    print("TEMPLATE_CARD = "+str(TEMPLATE_CARD))

    empty_and_fake_nodes = tf.equal(fn_[:,:,0],-1)
    print("empty_and_fake_nodes shape = "+str(empty_and_fake_nodes.shape))

    empty_and_fake_nodes = tf.tile(tf.expand_dims(empty_and_fake_nodes,-1),[1,1,10])

    batchNodeRawWeights = tf.tile(tf.expand_dims(nodeRawWeights_,axis=0), [BATCH_SIZE,1,1])

    nodeFilteredWeights = tf.where(empty_and_fake_nodes,tf.zeros_like(batchNodeRawWeights,dtype=tf.float32),batchNodeRawWeights)
    nodeMask = tf.where(empty_and_fake_nodes,tf.zeros_like(batchNodeRawWeights,dtype=tf.float32),tf.ones_like(batchNodeRawWeights,dtype=tf.float32))

    # [batch, N, 10]
    batchWeights = tf.reduce_sum(nodeFilteredWeights, axis=1)
    # [batch, 10]

    new_fn = fn_

    fn_centered = 2 * new_fn - 1

    fullfn = tf.concat((fn_centered,node_pos_tiled), axis=-1)

    padding = tf.zeros([BATCH_SIZE,graph_card[0]-TEMPLATE_CARD,NUM_IN_CHANNELS+3])
    padding = padding-3
    ext_fn = tf.concat((fullfn, padding), axis=1)
    perm_fn = tf.gather(ext_fn,perm_, axis=1)


    perm_empty_and_fake_nodes = tf.equal(perm_fn[:,:,0],-3)
    print("perm_empty_and_fake_nodes shape = "+str(perm_empty_and_fake_nodes.shape))

    # perm_empty_and_fake_nodes = tf.tile(tf.expand_dims(perm_empty_and_fake_nodes,-1),[1,1,10])

    perm_node_mask = tf.where(perm_empty_and_fake_nodes,tf.zeros_like(perm_empty_and_fake_nodes,dtype=tf.float32),tf.ones_like(perm_empty_and_fake_nodes,dtype=tf.float32))

    gtgender = gtshapegender_[:,0]
    gtshape = gtshapegender_[:,1:]

    # ext_shapegender = tf.concatenate((gtshapegender_, tf.zeros([graph_card[0]-TEMPLATE_CARD,11])), axis=0)
    # perm_shapegender = tf.gather(ext_shapegender,perm_)

    # sample_ind = tf.placeholder(tf.int32, shape=[10000], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)

    fadjs = [fadj0,fadj1,fadj2,fadj3]


    with tf.variable_scope("model_gender_class"):
        encodedMesh, _ = get_mesh_encoder(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode='gender')
        encodedMesh_centered = encodedMesh * 2 - 1
    with tf.device(DEVICE):
        
        fullGraphLoss = mseLoss(encodedMesh_centered, perm_fn[:,:,:2])

        filteredLoss = tf.multiply(fullGraphLoss, perm_node_mask)

        train_step = tf.train.AdamOptimizer().minimize(filteredLoss, global_step=batch)
        
        isNanPmale = tf.reduce_any(tf.is_nan(encodedMesh), name="isNanPmale")
            
    

    batch_loss = tf.reduce_mean(filteredLoss)

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
    

    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    # Training

    train_loss=0
    train_samp=0

    SAVEITER = 1000

    evalStepNum=20

    with tf.device(DEVICE):
        # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        lossArray = np.zeros([int(SAVEITER/evalStepNum),2])
        last_loss = 0
        lossArrayIter = 0
        for iter in range(NUM_ITERATIONS):

            # Batch version...
            inbatchlist = []
            gtbatchlist = []
            for b in range(BATCH_SIZE):
                batch_num = random.randint(0,EXAMPLE_NUM-1)

                if mode=='shape' or mode=='bodyratio':
                    # LIMIT TO MALE EXAMPLES
                    while gtshape_list[batch_num][0,0]==0:
                        batch_num = random.randint(0,EXAMPLE_NUM-1)

                inbatchlist.append(input_list[batch_num])
                gtbatchlist.append(gtshape_list[batch_num])


            in_batch = np.concatenate(inbatchlist, axis=0)
            shape_batch = np.concatenate(gtbatchlist, axis=0)

            # train_fd = {fn_: input_list[batch_num], gtshapegender_: gtshape_list[batch_num], keep_prob:1}
            train_fd = {fn_: in_batch, gtshapegender_: shape_batch, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, nodeRawWeights_:nodeParamDisp, keep_prob:0.8}


            # One-example version
                    # # Get random sample from training dictionary
                    # batch_num = random.randint(0,EXAMPLE_NUM-1)

                    # train_fd = {fn_: input_list[batch_num], fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], gtshapegender_: gtshape_list[batch_num], perm_: perm, node_pos:face_pos, keep_prob:1}

            # Show smoothed training loss
            if ((iter%evalStepNum == 0) and (iter>0)):
                train_loss = train_loss/train_samp
                

                print("Iteration %d, training loss %g"%(iter, train_loss))

                # if mode=='gender':
                #     pred_pmale = sess.run(pmale, feed_dict=train_fd)
                #     print("Predicted male probability = "+str(pred_pmale) + ", Ground truth = " + str(shape_batch[:,0])) 

                lossArray[lossArrayIter,0]=train_loss
                lossArrayIter+=1
                train_loss=0
                train_samp=0
                # pred_pmale, valGender,valYp, valYgt, valLogYp = sess.run([pmale, gtgender, yp, ygt, testLogYp], feed_dict=train_fd)
                # print("pred_pmale shape = "+str(pred_pmale.shape))
                # print("pred_pmale[0] = "+str(pred_pmale[0]))
                # pred_pmale_val = pred_pmale[0]
                # print("Predicted male probability = %g, Ground truth = %g"%(pred_pmale_val, gtshape_list[batch_num][0,0]))
                # print("invalues: (yp = %g, ygt = %g). log(yp) = %g"%(valYp,valYgt, valLogYp))

            
            # marg_train_loss = crossEntropyLoss.eval(feed_dict=train_fd)

            marg_train_loss = sess.run(batch_loss, feed_dict=train_fd)
            train_loss += marg_train_loss
            train_samp+=1

            if iter==0:
                myinput, myoutput = sess.run([perm_fn, encodedMesh], feed_dict=train_fd)
                myinputsamp = myinput[1,:,:]
                myoutputsamp = myoutput[1,:,:]
                mynodepos = myinputsamp[:,-3:]
                mynodecolor=myinputsamp[:,:2]
                mynodecolor=np.tile(mynodecolor,[1,2])
                mynodecolor=(mynodecolor[:,:3]+1)/2

                # print("node colors min & max: (%f, %f)"%(np.amin(mynodecolor),np.amax(mynodecolor)))
                mynodecolor = np.where(mynodecolor<0,0,mynodecolor)
                # print("node colors min & max: (%f, %f)"%(np.amin(mynodecolor),np.amax(mynodecolor)))
                mynodecolor = mynodecolor * 255
                # print("node colors min & max: (%f, %f)"%(np.amin(mynodecolor),np.amax(mynodecolor)))
                mynodecolor = mynodecolor.astype(int)
                mynodecolor = mynodecolor.astype(float)
                xyzarray = np.concatenate((mynodepos,mynodecolor),axis=1)
                write_coff(xyzarray,"/morpheo-nas2/marmando/ShapeRegression/Test/meshEncoder/input_test.off")


                outcolor = np.tile(myoutputsamp,[1,2])
                outcolor=(outcolor[:,:3])
                outcolor = outcolor * 255
                outcolor = outcolor.astype(int)
                outcolor = outcolor.astype(float)
                xyzarray = np.concatenate((mynodepos,outcolor),axis=1)
                write_coff(xyzarray,"/morpheo-nas2/marmando/ShapeRegression/Test/meshEncoder/output_test.off")

            # Compute validation loss
            if True:
                if (iter%(evalStepNum*2) ==0) and (iter>0):
                    valid_loss = 0
                    for vbm in range(VALID_EXAMPLE_NUM):
                        
                        valid_inbatchlist = []
                        valid_gtbatchlist = []
                        for b in range(BATCH_SIZE):
                            batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                            if mode=='shape' or mode=='bodyratio':
                                # LIMIT TO MALE EXAMPLES
                                while valid_gtshape_list[batch_num][0,0]==0:
                                    batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                            valid_inbatchlist.append(valid_input_list[batch_num])
                            valid_gtbatchlist.append(valid_gtshape_list[batch_num])
                            # valid_inbatchlist.append(valid_input_list[b])
                            # valid_gtbatchlist.append(valid_gtshape_list[b])



                        valid_in_batch = np.concatenate(valid_inbatchlist, axis=0)
                        valid_shape_batch = np.concatenate(valid_gtbatchlist, axis=0)
                        
                        valid_fd = {fn_: valid_in_batch, gtshapegender_: valid_shape_batch, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, nodeRawWeights_:nodeParamDisp, keep_prob:1}


                        # valid_fd = {fn_: valid_input_list[vbm], fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], gtshapegender_: valid_gtshape_list[vbm], perm_: perm, node_pos:face_pos, keep_prob:1}

                        # valid_loss += crossEntropyLoss.eval(feed_dict=valid_fd)
                        valid_loss += sess.run(batch_loss,feed_dict=valid_fd)
                    valid_loss/=VALID_EXAMPLE_NUM
                    print("Iteration %d, validation loss %g\n"%(iter, valid_loss))
                    lossArray[lossArrayIter-1,1]=valid_loss
                    if iter>0:
                        lossArray[lossArrayIter-2,1] = (valid_loss+last_loss)/2
                        last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            
            if sess.run(isNanPmale,feed_dict=train_fd):
                    hasNan = True
                    print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(input_list)))
                    return
            
            if (iter%SAVEITER == 0) and (iter>0):
                saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
                f = open(csv_filename,'ab')
                np.savetxt(f,lossArray, delimiter=",")
                f.close()
                lossArray = np.zeros([int(SAVEITER/evalStepNum),2]) 
                lossArrayIter=0
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()





def trainImageNet(input_list, gtshape_list, valid_input_list, valid_gtshape_list, mode='gender'):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


    EXAMPLE_NUM = len(input_list)
    VALID_EXAMPLE_NUM = len(valid_input_list)

    print("EXAMPLE_NUM = "+str(EXAMPLE_NUM))
    
    NUM_IN_CHANNELS = input_list[0].shape[3]
    NUM_IN_CHANNELS = 3
    WIDTH = 320     # input_list[0].shape[1]
    HEIGHT = 240    # input_list[0].shape[2]
    
    BATCH_SIZE = 10

    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, HEIGHT, WIDTH, NUM_IN_CHANNELS], name='fn_')
    gtshapegender_ = tf.placeholder('float32', shape=[BATCH_SIZE,11], name='tfn_')


    gtgender = gtshapegender_[:,0]
    gtshape = gtshapegender_[:,1:]



    # ext_shapegender = tf.concatenate((gtshapegender_, tf.zeros([graph_card[0]-TEMPLATE_CARD,11])), axis=0)
    # perm_shapegender = tf.gather(ext_shapegender,perm_)



    # sample_ind = tf.placeholder(tf.int32, shape=[10000], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)
    fn_centered = fn_ * 2 - 1

    if mode=='gender':
        with tf.variable_scope("model_gender_class"):
            pmale, testFeatures = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode='gender')
        
        with tf.device(DEVICE):

            # print("WARNING!!! Hard-coded constant output")
            # pmale = tf.constant([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
            # initial = tf.random_normal([1], stddev=0.01)
            # testVar =  tf.Variable(initial, name="weight")
            # pmale = pmale + tf.minimum(tf.maximum(testVar, tf.constant([-0.5])),tf.constant([0.5]))
            totalLogl = loglikelihood(tf.reshape(pmale,[-1]),gtgender)
            crossEntropyLoss = -totalLogl
            
    elif mode=='shape':
        with tf.variable_scope("model_gender_class"):
            predicted_shape, _ = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode='shape')
            pmale = predicted_shape
        with tf.device(DEVICE):
            # shapeLoss = mseLoss(predicted_shape,gtshape)
            print("WARNING!!!! Hard-coded loss change (1st param)")
            shapeLoss = mseLoss(predicted_shape[:,:1],gtshape[:,:1])
            crossEntropyLoss = shapeLoss

    elif mode=='bodyratio':
        tfH0 = tf.constant(H0, shape=[1,1], dtype=tf.float32)
        tfW0 = tf.constant(W0, shape=[1,1], dtype=tf.float32)
        tfD0 = tf.constant(D0, shape=[1,1], dtype=tf.float32)
        tfHMat = tf.constant(HMat, shape=[10,1], dtype=tf.float32)
        tfWMat = tf.constant(WMat, shape=[10,1], dtype=tf.float32)
        tfDMat = tf.constant(DMat, shape=[10,1], dtype=tf.float32)

        bH = tf.matmul(gtshape,tfHMat)
        bW = tf.matmul(gtshape,tfWMat)
        bD = tf.matmul(gtshape,tfDMat)
        print("tfHMat shape = "+str(tfHMat.shape))
        print("bH shape = "+str(bH.shape))
        # [batch, 1]
        bH = bH + tfH0
        bW = bW + tfW0
        bD = bD + tfD0
        print("bH shape = "+str(bH.shape))

        new_param = tf.divide(bD,bH)
        # [batch, 1]
        print("new_param shape = "+str(new_param.shape))
        with tf.variable_scope("model_gender_class"):
            predicted_shape, _ = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode)
            pmale = predicted_shape
        with tf.device(DEVICE):
            print("predicted_shape shape = "+str(predicted_shape.shape))
            print("gtshape shape = "+str(gtshape.shape))            
            print("WARNING!!!! Hard-coded loss change (homemade param)")
            shapeLoss = mseLoss(predicted_shape,new_param)
            print("shapeLoss shape = "+str(shapeLoss.shape))
            crossEntropyLoss = shapeLoss


    lossSum = tf.reduce_mean(crossEntropyLoss)
    train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
    isNanPmale = tf.reduce_any(tf.is_nan(pmale), name="isNanPmale")

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
    

    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    # Training

    train_loss=0
    train_samp=0

    SAVEITER = 2000
    evalStepNum=50

    with tf.device(DEVICE):
        # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        lossArray = np.zeros([int(SAVEITER/evalStepNum),2])
        last_loss = 0
        lossArrayIter = 0
        for iter in range(NUM_ITERATIONS):


            # Get random sample from training dictionary
            # Hard-coded batch size
            inbatchlist = []
            gtbatchlist = []
            for b in range(BATCH_SIZE):
                batch_num = random.randint(0,EXAMPLE_NUM-1)
                if mode=='shape' or mode=='bodyratio':
                    # LIMIT TO MALE EXAMPLES
                    while gtshape_list[batch_num][0,0]==0:
                        batch_num = random.randint(0,EXAMPLE_NUM-1)
                inbatchlist.append(input_list[batch_num])
                gtbatchlist.append(gtshape_list[batch_num])


            in_batch = np.concatenate(inbatchlist, axis=0)
            shape_batch = np.concatenate(gtbatchlist, axis=0)

            # train_fd = {fn_: input_list[batch_num], gtshapegender_: gtshape_list[batch_num], keep_prob:1}
            train_fd = {fn_: in_batch, gtshapegender_: shape_batch, keep_prob:1}
            
            #sess.run(customLoss,feed_dict=train_fd)
            

            # marg_train_loss, posCost, negCost = sess.run([crossEntropyLoss, posC, negC], feed_dict=train_fd)

            # Show smoothed training loss
            # if ((iter%10 == 0) and (iter>0)) or (iter>0 and iter<10):
            if ((iter%evalStepNum == 0) and (iter>0)):
                train_loss = train_loss/train_samp
                

                print("Iteration %d, training loss %g"%(iter, train_loss))

                lossArray[lossArrayIter,0]=train_loss
                lossArrayIter+=1
                train_loss=0
                train_samp=0
                # pred_pmale, loglklhd, testF = sess.run([pmale, totalLogl, testFeatures], feed_dict=train_fd)
                # pred_pmale = sess.run(pmale, feed_dict=train_fd)
                # # print("pred_pmale shape = "+str(pred_pmale.shape))
                # # print("pred_pmale[0] = "+str(pred_pmale[0]))
                # pred_pmale_val = pred_pmale[0]
                # print("Predicted male probability = "+str(pred_pmale) + ", Ground truth = " + str(shape_batch[:,0]))
                # print("loglikelihood = "+str(loglklhd))
                # print("test Features = "+str(testF))
                # # print("invalues: (yp = %g, ygt = %g). log(yp) = %g"%(valYp,valYgt, valLogYp))

            
            # marg_train_loss = crossEntropyLoss.eval(feed_dict=train_fd)

            marg_train_loss = sess.run(lossSum, feed_dict=train_fd)
            train_loss += marg_train_loss
            train_samp+=1


            # Compute validation loss
            if (iter%(evalStepNum*2) ==0) and (iter>0):
                valid_loss = 0
                
                valid_inbatchlist = []
                valid_gtbatchlist = []
                for b in range(BATCH_SIZE):
                    batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                    if mode=='shape' or mode=='bodyratio':
                        # LIMIT TO MALE EXAMPLES
                        while valid_gtshape_list[batch_num][0,0]==0:
                            batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                    valid_inbatchlist.append(valid_input_list[batch_num])
                    valid_gtbatchlist.append(valid_gtshape_list[batch_num])


                valid_in_batch = np.concatenate(valid_inbatchlist, axis=0)
                valid_shape_batch = np.concatenate(valid_gtbatchlist, axis=0)
                
                valid_fd = {fn_: valid_in_batch, gtshapegender_: valid_shape_batch, keep_prob:1}
                # valid_loss = crossEntropyLoss.eval(feed_dict=valid_fd)
                valid_loss = sess.run(lossSum,feed_dict=valid_fd)
                
                # for vbm in range(VALID_EXAMPLE_NUM):
                    # valid_fd = {fn_: valid_input_list[vbm], gtshapegender_: valid_gtshape_list[vbm], keep_prob:1}

                #     valid_loss += crossEntropyLoss.eval(feed_dict=valid_fd)
                # valid_loss/=VALID_EXAMPLE_NUM
                print("Iteration %d, validation loss %g\n"%(iter, valid_loss))
                lossArray[lossArrayIter-1,1]=valid_loss
                if iter>0:
                    lossArray[lossArrayIter-2,1] = (valid_loss+last_loss)/2
                    last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            
            if sess.run(isNanPmale,feed_dict=train_fd):
                    hasNan = True
                    print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(input_list)))
                    return
            
            if (iter%SAVEITER == 0) and (iter>0):
                saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
                f = open(csv_filename,'ab')
                np.savetxt(f,lossArray, delimiter=",")
                f.close()
                lossArray = np.zeros([int(SAVEITER/evalStepNum),2]) 
                lossArrayIter=0
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()



def trainImageEncoder(input_list, valid_input_list, gtshape_list, valid_gtshape_list, mode='encoder'):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)


    EXAMPLE_NUM = len(input_list)
    VALID_EXAMPLE_NUM = len(valid_input_list)

    print("EXAMPLE_NUM = "+str(EXAMPLE_NUM))
    
    NUM_IN_CHANNELS = input_list[0].shape[3]
    NUM_IN_CHANNELS = 3
    WIDTH = 320     # input_list[0].shape[1]
    HEIGHT = 240    # input_list[0].shape[2]
    
    BATCH_SIZE = 20

    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, HEIGHT, WIDTH, NUM_IN_CHANNELS], name='fn_')
    
    noise_ = tf.placeholder('float32', shape=[BATCH_SIZE, HEIGHT, WIDTH, NUM_IN_CHANNELS], name='fn_')

    keep_prob = tf.placeholder(tf.float32)

    gtshapegender_ = tf.placeholder('float32', shape=[BATCH_SIZE,11], name='tfn_')


    gtgender = gtshapegender_[:,0]
    gtshape = gtshapegender_[:,1:]
    
    batch = tf.Variable(0, trainable=False)
    fn_centered = fn_ * 2 - 1

    noisy_fn = fn_centered + noise_

    mask0_bool = tf.reduce_any(tf.not_equal(fn_centered,-1), axis=-1)
    # [batch, width, height]
    mask0 = tf.where(mask0_bool, tf.ones([BATCH_SIZE, HEIGHT, WIDTH], dtype=tf.float32), tf.zeros([BATCH_SIZE, HEIGHT, WIDTH], dtype=tf.float32))
    # mask0 = tf.expand_dims(mask0,axis=-1)


    with tf.variable_scope("model_gender_class"):
        out_gen, code = get_image_encoder(noisy_fn, ARCHITECTURE, keep_prob)
    

    if mode=='encoder':    
        with tf.device(DEVICE):
        
            fullImageLoss = mseLoss(out_gen, fn_)

        lossT = tf.multiply(mask0, fullImageLoss)
        lossF = fullImageLoss - lossT
        finalLoss = 100 * lossT + lossF
        train_step = tf.train.AdamOptimizer().minimize(finalLoss, global_step=batch)
        
        lossSum = tf.reduce_mean(finalLoss)
    
    


    

    # saver0 = tf.train.Saver()

    # sess.run(tf.global_variables_initializer())

    # globalStep = 0

    # ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
    # if ckpt and ckpt.model_checkpoint_path:
    #     splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
    #     if splitCkpt[0] == NET_NAME:
    #         saver0.restore(sess, ckpt.model_checkpoint_path)
    #         #Extract from checkpoint filename
    #         globalStep = int(splitCkpt[1])
    



    if mode=='bodyratio':

        code_ = tf.placeholder('float32', shape=[BATCH_SIZE, 1, 512], name='code_')

        tfH0 = tf.constant(H0, shape=[1,1], dtype=tf.float32)
        tfW0 = tf.constant(W0, shape=[1,1], dtype=tf.float32)
        tfD0 = tf.constant(D0, shape=[1,1], dtype=tf.float32)
        tfHMat = tf.constant(HMat, shape=[10,1], dtype=tf.float32)
        tfWMat = tf.constant(WMat, shape=[10,1], dtype=tf.float32)
        tfDMat = tf.constant(DMat, shape=[10,1], dtype=tf.float32)

        bH = tf.matmul(gtshape,tfHMat)
        bW = tf.matmul(gtshape,tfWMat)
        bD = tf.matmul(gtshape,tfDMat)
        print("tfHMat shape = "+str(tfHMat.shape))
        print("bH shape = "+str(bH.shape))
        # [batch, 1]
        bH = bH + tfH0
        bW = bW + tfW0
        bD = bD + tfD0
        print("bH shape = "+str(bH.shape))

        new_param = tf.divide(bD,bH)
        # [batch, 1]
        print("new_param shape = "+str(new_param.shape))
        print("code shape = "+str(code.shape))
        predicted_br = get_encoded_reg(code_, 0)

        with tf.device(DEVICE):
            shapeLoss = mseLoss(predicted_br,new_param)
            print("shapeLoss shape = "+str(shapeLoss.shape))
            train_step = tf.train.AdamOptimizer().minimize(shapeLoss, global_step=batch)
            lossSum = tf.reduce_mean(shapeLoss)

    isNanPmale = tf.reduce_any(tf.is_nan(lossSum), name="isNanPmale")
    saver1 = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
    if ckpt and ckpt.model_checkpoint_path:
        splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
        if splitCkpt[0] == NET_NAME:
            saver1.restore(sess, ckpt.model_checkpoint_path)
            #Extract from checkpoint filename
            globalStep = int(splitCkpt[1])

    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    # Training

    train_loss=0
    train_samp=0

    SAVEITER = 500
    evalStepNum=50

    with tf.device(DEVICE):
        # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        lossArray = np.zeros([int(SAVEITER/evalStepNum),2])
        last_loss = 0
        lossArrayIter = 0
        for iter in range(NUM_ITERATIONS):


            # Get random sample from training dictionary
            # Hard-coded batch size
            inbatchlist = []
            gtbatchlist = []
            for b in range(BATCH_SIZE):
                batch_num = random.randint(0,EXAMPLE_NUM-1)

                # LIMIT TO MALE EXAMPLES
                while gtshape_list[batch_num][0,0]==0:
                    batch_num = random.randint(0,EXAMPLE_NUM-1)
                inbatchlist.append(input_list[batch_num])
                gtbatchlist.append(gtshape_list[batch_num])

            in_batch = np.concatenate(inbatchlist, axis=0)
            shape_batch = np.concatenate(gtbatchlist, axis=0)
            noisy_im = np.random.randn(BATCH_SIZE,HEIGHT,WIDTH,NUM_IN_CHANNELS)/10

            if mode=='bodyratio':
                code_fd = {fn_: in_batch, noise_: noisy_im, keep_prob:1}
                myCode = sess.run(code, feed_dict=code_fd)
                train_fd = {fn_: in_batch, code_: myCode, gtshapegender_: shape_batch, keep_prob:1}
            elif mode=='encoder':
                train_fd = {fn_: in_batch, noise_: noisy_im, keep_prob:1}

            # Show smoothed training loss
            # if ((iter%10 == 0) and (iter>0)) or (iter>0 and iter<10):
            if ((iter%evalStepNum == 0) and (iter>0)):
                train_loss = train_loss/train_samp
                

                print("Iteration %d, training loss %g"%(iter, train_loss))

                lossArray[lossArrayIter,0]=train_loss
                lossArrayIter+=1
                train_loss=0
                train_samp=0
                # pred_pmale, loglklhd, testF = sess.run([pmale, totalLogl, testFeatures], feed_dict=train_fd)
                # pred_pmale = sess.run(pmale, feed_dict=train_fd)
                # # print("pred_pmale shape = "+str(pred_pmale.shape))
                # # print("pred_pmale[0] = "+str(pred_pmale[0]))
                # pred_pmale_val = pred_pmale[0]
                # print("Predicted male probability = "+str(pred_pmale) + ", Ground truth = " + str(shape_batch[:,0]))
                # print("loglikelihood = "+str(loglklhd))
                # print("test Features = "+str(testF))
                # # print("invalues: (yp = %g, ygt = %g). log(yp) = %g"%(valYp,valYgt, valLogYp))

            
            # marg_train_loss = crossEntropyLoss.eval(feed_dict=train_fd)

            marg_train_loss = sess.run(lossSum, feed_dict=train_fd)
            train_loss += marg_train_loss
            train_samp+=1


            # Compute validation loss
            if (iter%(evalStepNum*2) ==0) and (iter>0):
                valid_loss = 0
                
                valid_inbatchlist = []
                valid_gtbatchlist = []
                for b in range(BATCH_SIZE):
                    batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                    # LIMIT TO MALE EXAMPLES
                    while valid_gtshape_list[batch_num][0,0]==0:
                        batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                    valid_inbatchlist.append(valid_input_list[batch_num])
                    valid_gtbatchlist.append(valid_gtshape_list[batch_num])

                valid_in_batch = np.concatenate(valid_inbatchlist, axis=0)
                valid_shape_batch = np.concatenate(valid_gtbatchlist, axis=0)
                valid_noisy_im = np.random.randn(BATCH_SIZE,HEIGHT,WIDTH,NUM_IN_CHANNELS)/10

                if mode=='bodyratio':
                    valid_code_fd = {fn_: valid_in_batch, noise_: valid_noisy_im, keep_prob:1}
                    valid_myCode = sess.run(code, feed_dict=valid_code_fd)
                    valid_fd = {fn_: valid_in_batch, code_: valid_myCode, gtshapegender_: valid_shape_batch, keep_prob:1}
                elif mode=='encoder':
                    valid_fd = {fn_: valid_in_batch, noise_: valid_noisy_im, keep_prob:1}


                valid_loss = sess.run(lossSum,feed_dict=valid_fd)
                
                # for vbm in range(VALID_EXAMPLE_NUM):
                    # valid_fd = {fn_: valid_input_list[vbm], gtshapegender_: valid_gtshape_list[vbm], keep_prob:1}

                #     valid_loss += crossEntropyLoss.eval(feed_dict=valid_fd)
                # valid_loss/=VALID_EXAMPLE_NUM
                print("Iteration %d, validation loss %g\n"%(iter, valid_loss))
                lossArray[lossArrayIter-1,1]=valid_loss
                if iter>0:
                    lossArray[lossArrayIter-2,1] = (valid_loss+last_loss)/2
                    last_loss=valid_loss

            sess.run(train_step,feed_dict=train_fd)
            
            if sess.run(isNanPmale,feed_dict=train_fd):
                    hasNan = True
                    print("WARNING! NAN FOUND AFTER TRAINING!!!! training example "+str(batch_num)+"/"+str(len(input_list)))
                    return
            
            if (iter%SAVEITER == 0) and (iter>0):
                saver1.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
                f = open(csv_filename,'ab')
                np.savetxt(f,lossArray, delimiter=",")
                f.close()
                lossArray = np.zeros([int(SAVEITER/evalStepNum),2]) 
                lossArrayIter=0
    
    saver1.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()





def testBodyShapeNet(inGraph, graph_adj, perm, face_pos, mode='shape'):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

    
    NUM_IN_CHANNELS = inGraph.shape[2]
    # NUM_IN_CHANNELS = 29

    BATCH_SIZE = inGraph.shape[0]           # Number of input to process
    
    graph_card = np.zeros(len(graph_adj), dtype=np.int32)
    K_faces = np.zeros(len(graph_adj), dtype=np.int32)
    for g in range(len(graph_adj)):
        print("graph_adj["+str(g)+"] shape = "+str(graph_adj[g].shape))
        graph_card[g] = graph_adj[g].shape[1]
        K_faces[g] = graph_adj[g].shape[2]

    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE,TEMPLATE_CARD, NUM_IN_CHANNELS], name='fn_')
    node_pos = tf.placeholder(tf.float32, shape=[1, TEMPLATE_CARD, 3], name='node_pos')

    fadj0 = tf.placeholder(tf.int32, shape=[1,graph_card[0], K_faces[0]], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[1,graph_card[1], K_faces[1]], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[1,graph_card[2], K_faces[2]], name='fadj2')
    fadj3 = tf.placeholder(tf.int32, shape=[1,graph_card[3], K_faces[3]], name='fadj2')

    perm_ = tf.placeholder(tf.int32, shape=[graph_card[0]], name='perm')

    new_fn = fn_

    fn_centered = 2 * new_fn - 1

    print("graph_card = "+str(graph_card))
    print("TEMPLATE_CARD = "+str(TEMPLATE_CARD))
    node_pos_tiled = tf.tile(node_pos,[BATCH_SIZE,1,1])
    fullfn = tf.concat((fn_centered,node_pos_tiled), axis=-1)

    padding = tf.zeros([BATCH_SIZE,graph_card[0]-TEMPLATE_CARD,NUM_IN_CHANNELS+3])
    padding = padding-3
    ext_fn = tf.concat((fullfn, padding), axis=1)
    perm_fn = tf.gather(ext_fn,perm_, axis=1)



    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)

    fadjs = [fadj0,fadj1,fadj2,fadj3]

    print("perm_fn shape = "+str(perm_fn.shape))
    with tf.device(DEVICE):
        with tf.variable_scope("model_gender_class"):
            predicted_shape, testFeatures = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode)
            # if mode='gender':
            #     pmale, testFeatures = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode='gender')
            # elif mode='shape':
            #     predicted_shape, testFeatures = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode='shape')
        

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    globalStep = 0


    # test = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # varnum=0
    # for t in test:
    #     print(t)
    #     varnum+=1
    #     if varnum>20:
    #         break

    # return

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
    if ckpt and ckpt.model_checkpoint_path:
        splitCkpt = os.path.basename(ckpt.model_checkpoint_path).split('-')
        if splitCkpt[0] == NET_NAME:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #Extract from checkpoint filename
            globalStep = int(splitCkpt[1])
    

    with tf.device(DEVICE):
        infer_fd = {fn_: inGraph, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, keep_prob:1}

        myShape, myFeatures = sess.run([predicted_shape, tf.squeeze(testFeatures)], feed_dict=infer_fd)

    
        

    sess.close()
    
    invPerm = inv_perm(perm)

    # print("myFeatures shape = "+str(myFeatures.shape))
    # print("perm shape = "+str(perm.shape))
    # print("invPerm shape = "+str(invPerm.shape))
    myFeatures = np.transpose(myFeatures,[1,0,2])
    myFeatures = myFeatures[invPerm]
    myFeatures = myFeatures[:TEMPLATE_CARD,:]
    myFeatures = np.transpose(myFeatures,[1,0,2])


    return myShape, myFeatures

def testImageNet(inputImage, mode='gender'):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

    NUM_IN_CHANNELS = inputImage.shape[3]
    NUM_IN_CHANNELS = 3
    WIDTH = 320     # input_list[0].shape[1]
    HEIGHT = 240    # input_list[0].shape[2]

    BATCH_SIZE = inputImage.shape[0]    # Number of input images to process

    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, HEIGHT, WIDTH, NUM_IN_CHANNELS], name='fn_')

    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)
    fn_centered = fn_ * 2 - 1

    with tf.variable_scope("model_gender_class"):
            pmale, testFeatures = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode)
    # if mode=='gender':
    #     with tf.variable_scope("model_gender_class"):
    #         pmale, testFeatures = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode='gender')
    # elif mode=='bodyratio':
    #     with tf.variable_scope("model_gender_class"):
    #         pmale, testFeatures = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode=mode)
    # elif mode=='shape':
    #     with tf.variable_scope("model_gender_class"):
    #         predicted_shape, testFeatures = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob, mode='shape')
    #         pmale = predicted_shape

    
    isNanPmale = tf.reduce_any(tf.is_nan(pmale), name="isNanPmale")

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

    with tf.device(DEVICE):
        train_fd = {fn_: inputImage, keep_prob:1}

        mypmale, myFeatures = sess.run([pmale, tf.squeeze(testFeatures)], feed_dict=train_fd)
    
    sess.close()
    
    return mypmale, myFeatures


def testImageEncoder(inputImage):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    # if not os.path.exists(RESULTS_PATH):
    #         os.makedirs(RESULTS_PATH)
    
    NUM_IN_CHANNELS = inputImage.shape[3]
    NUM_IN_CHANNELS = 3
    WIDTH = 320     # inputImage.shape[1]
    HEIGHT = 240    # inputImage.shape[2]
    
    BATCH_SIZE = inputImage.shape[0]    # Number of input images to process

    # training data
    fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, HEIGHT, WIDTH, NUM_IN_CHANNELS], name='fn_')
    
    # noise_ = tf.placeholder('float32', shape=[BATCH_SIZE, HEIGHT, WIDTH, NUM_IN_CHANNELS], name='fn_')

    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)
    fn_centered = fn_ * 2 - 1

    with tf.variable_scope("model_gender_class"):
        out_gen, code = get_image_encoder(fn_centered, ARCHITECTURE, keep_prob)


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
    

    with tf.device(DEVICE):
        train_fd = {fn_: inputImage, keep_prob:1}
        myGenImage, myEncodedImage = sess.run([out_gen, code],feed_dict=train_fd)

            
    sess.close()
    
    return myGenImage, myEncodedImage


def malahanobisLoss(prediction, gt, invCovarMat):
    diffVec = tf.subtract(gt,prediction)
    diffVecCol = tf.reshape(diffVec,[-1,1])
    diffVecRow = tf.reshape(diffVec,[1,-1])
    
    squareDist = tf.matmul(diffVecRow,tf.matmul(invCovarMat,diffVecCol))

    return squareDist


def mseLoss(prediction, gt):

    loss = tf.reduce_sum(tf.square(tf.subtract(gt,prediction)),axis=-1)
    return loss


def mseLossWeighted(prediction, gt, paramWeights):

    loss = tf.reduce_sum(tf.square(tf.subtract(gt,prediction)),axis=-1)
    
    sloss = tf.square(tf.subtract(gt, prediction))

    weightedsloss = tf.multiply(sloss, paramWeights)

    batchloss = tf.reduce_sum(weightedsloss, axis=-1)

    totalWeight = tf.reduce_sum(paramWeights)

    loss = tf.divide(batchloss,totalWeight)

    return loss



def loglikelihood(yp,ygt, name='loglikelihood'):

    epsilon=10e-9
    with tf.variable_scope('accuracyLoss'):
        posC = ygt*tf.log(yp+epsilon)
        negC = tf.subtract(tf.constant(1.0),ygt)*tf.log(tf.subtract(tf.constant(1.0),yp)+epsilon)
        logl = tf.add(posC,negC, name='logl')
        # logl = tf.reduce_sum(logl,name='logl')

    print("yp shape = "+str(yp.shape))
    print("ygt shape = "+str(ygt.shape))
    print("posC shape = "+str(posC.shape))
    print("negC shape = "+str(negC.shape))
    print("logl shape = "+str(logl.shape))
    return logl



def pickleImages(inputPath, destPath):

    inputList = []
    gtShapeList = []
    pos3dList = []
    fileNum=0

    pathList = inputPath.split('/')
    inputFolder=pathList[-2]
    imageDumpPath = "/morpheo-nas2/marmando/ShapeRegression/ImagesDump/"

    imageDumpFolder = imageDumpPath+inputFolder+"/"

    if not os.path.exists(imageDumpFolder):
        os.makedirs(imageDumpFolder)

    # Now read the smpl model.
    with open('/morpheo-nas2/marmando/densepose/DensePoseData/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    # with open('./DensePoseData/basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
        data = pickle.load(f)
        Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)

    # # Now read the smpl model.
    # with open('/morpheo-nas2/marmando/densepose/DensePoseData/basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    # # with open('./DensePoseData/basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     VerticesF = data['v_template']  ##  Loaded vertices of size (6890, 3)

    vxmax = np.amax(Vertices[:,0])
    vxmin = np.amin(Vertices[:,0])
    vymax = np.amax(Vertices[:,1])
    vymin = np.amin(Vertices[:,1])
    vzmax = np.amax(Vertices[:,2])
    vzmin = np.amin(Vertices[:,2])
    Vertices[:,0] = Vertices[:,0] - vxmin
    Vertices[:,1] = Vertices[:,1] - vymin
    Vertices[:,2] = Vertices[:,2] - vzmin

    bigmax = max((vxmax-vxmin),(vymax-vymin),(vzmax-vzmin))
    Vertices = Vertices/bigmax




    print("BB: (%f,%f,%f) to (%f,%f,%f)"%(vxmin,vymin,vzmin,vxmax,vymax,vzmax))
    for filename in os.listdir(inputPath):

        # if fileNum>=10:
        #     break

        # do stuff
        if not (filename.endswith(".jpg")):
            continue
        iuvFilename = filename[:-4]+"_IUV.png"

        infoFilename = filename[:-10]+"info.mat"

        if not os.path.isfile(inputPath + iuvFilename):
            print("WARNING: file with no infered densepose: "+filename)
            continue

        print("processing "+filename+"...")

        # Load Images
        # densepose image: on hold for now
        iuvIm = Image.open(inputPath+iuvFilename)
        iuvMat = np.array(iuvIm,dtype=np.float32)

        inIm = Image.open(inputPath+filename)
        inMat = np.array(inIm, dtype=np.float32)
        inMat = inMat/255

        # Add array to list
        inputList.append(inMat)

        # Get body shape
        infoMat = scipy.io.loadmat(inputPath+infoFilename)
        bodyShape = infoMat['shape'][:,0]
        gender = infoMat['gender'][0]               # array (1,)
        genBodyShape = np.concatenate((gender,bodyShape))
        # Add to list
        gtShapeList.append(genBodyShape)

        # Densepose part
        iMat = iuvMat[:,:,2]
        sil = np.nonzero(iMat)
        iuvpix = iuvMat[sil]

        points_num = iuvpix.shape[0]
        
        collected_3dpos = np.zeros((points_num,3),dtype=np.float32)
        # baryCoords = np.zeros((points_num,3))
        faceInd = np.zeros(points_num)
        im3dpos = np.zeros_like(inMat)
        # print("sil shape = "+str(sil.shape))
        # For every non null pixel:
        for i in range(points_num):
            # uu,vv,ii = iuvpix[i,:]
            ii = iuvpix[i,:][2]
            vv = iuvpix[i,:][0]/256
            uu = iuvpix[i,:][1]/256
            # Get face ind + bary coordinates
            FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)

            # Use FBC to get 3D coordinates on the surface.
            p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
            collected_3dpos[i,:] = p
            # print("sil[i] = "+str(sil[i]))
            im3dpos[sil[0][i],sil[1][i]]=p

        pos3dList.append(im3dpos)
        resIm = Image.fromarray((im3dpos*255).astype(np.uint8))
        resIm.save(imageDumpFolder + filename[:-4] + "_3dpos.png")

        fileNum+=1

    
    # pickle list
    with open(destPath+inputFolder+'_raw_image_input.pkl', 'wb') as fp:
        pickle.dump(inputList, fp)   
    # pickle list
    with open(destPath+inputFolder+'_gt_shape.pkl', 'wb') as fp:
        pickle.dump(gtShapeList, fp)
    # pickle list
    with open(destPath+inputFolder+'_3dpos.pkl', 'wb') as fp:
        pickle.dump(pos3dList, fp)

def pickleFolder(inputPath, destPath):

    inputList = []
    gtShapeList = []
    fileNum=0
    for filename in os.listdir(inputPath):

        # if fileNum>=10:
        #     break

        if not (filename.endswith(".jpg")):
            continue
        # if not (filename.startswith("01_03_c0003_img017")):
        #     continue
        iuvFilename = filename[:-4]+"_IUV.png"

        infoFilename = filename[:-10]+"info.mat"

        if not os.path.isfile(inputPath + iuvFilename):
            print("WARNING: file with no infered densepose: "+filename)
            continue

        print("processing "+filename+"...")

        # Load Images
        iuvIm = Image.open(inputPath+iuvFilename)
        iuvMat = np.array(iuvIm,dtype=np.float32)
        inIm = Image.open(inputPath+filename)
        inMat = np.array(inIm, dtype=np.float32)
        inMat = inMat/255

        padInMat = np.pad(inMat,((1,1),(1,1),(0,0)),mode='constant',constant_values=0)


        iMat = iuvMat[:,:,2]
        sil = np.nonzero(iMat)
        iuvpix = iuvMat[sil]

        # Get (image) UV coordinates
        rawUvec = sil[0]
        rawVvec = sil[1]

        # Normalize them (using bounding box)
        umin = np.amin(rawUvec)
        vmin = np.amin(rawVvec)

        uvec = rawUvec-umin
        vvec = rawVvec-vmin

        uvmax = max(np.amax(uvec),np.amax(vvec))
        
        uvec = uvec / uvmax
        vvec = vvec / uvmax



        points_num = iuvpix.shape[0]
        
        baryCoords = np.zeros((points_num,3))
        faceInd = np.zeros(points_num)

        # array of input values on the template graph
        netInput = np.zeros((TEMPLATE_CARD,29),dtype=np.float32)
        netInput = netInput-1

        # For every non null pixel:
        for i in range(points_num):
            # uu,vv,ii = iuvpix[i,:]
            ii = iuvpix[i,:][2]
            vv = iuvpix[i,:][0]/256
            uu = iuvpix[i,:][1]/256
            # Get face ind + bary coordinates
            FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)

            # add bary coords and pix coords to face
            faceInd[i] = FaceIndex
            baryCoords[i,:] = [bc1,bc2,bc3]
            
            imU = rawUvec[i]
            imV = rawVvec[i]
            writtenU = uvec[i]
            writtenV = vvec[i]
            imPatch = padInMat[imU:imU+3,imV:imV+3,:]
            imPatch = np.reshape(imPatch,-1)
            # print("imPatch = "+str(imPatch))

            # Simple strategy for now
            if netInput[FaceIndex,0]==-1:   # If no value on this node yet
                netInput[FaceIndex,0] = writtenU
                netInput[FaceIndex,1] = writtenV
                netInput[FaceIndex,2:] = imPatch

        # Add array to list
        inputList.append(netInput)

        # Get body shape
        infoMat = scipy.io.loadmat(inputPath+infoFilename)
        bodyShape = infoMat['shape'][:,0]
        gender = infoMat['gender'][0]               # array (1,)
        genBodyShape = np.concatenate((gender,bodyShape))
        # Add to list
        gtShapeList.append(genBodyShape)

        fileNum+=1


    pathList = inputPath.split('/')
    inputFolder=pathList[-2]
    # pickle list
    with open(destPath+inputFolder+'_standard_input.pkl', 'wb') as fp:
        pickle.dump(inputList, fp)   
    # pickle list
    with open(destPath+inputFolder+'_gt_shape.pkl', 'wb') as fp:
        pickle.dump(gtShapeList, fp)


def doIVUMatStuff(iuvMat):

    iMat = iuvMat[:,:,2]
    sil = np.nonzero(iMat)
    iuvpix = iuvMat[sil]

    # Get (image) UV coordinates
    rawUvec = sil[0]
    rawVvec = sil[1]

    # Normalize them (using bounding box)
    umin = np.amin(rawUvec)
    vmin = np.amin(rawVvec)

    uvec = rawUvec-umin
    vvec = rawVvec-vmin

    uvmax = max(np.amax(uvec),np.amax(vvec))
    
    uvec = uvec / uvmax
    vvec = vvec / uvmax

    return iuvpix, uvec, vvec

def pickleFolder2Images(inputPath, folder, destPath):

    inputList = []
    gtShapeList = []
    fileNum=0
    pklNum=0

    # for folder in sorted(os.listdir(inputPath)):        # For each subfodler
    print("Processing folder "+folder+"...")
    folderPath = inputPath + folder + "/"

    # Make list of all jpg files (i.e. original files)
    fileList = []
    for filename in sorted(os.listdir(folderPath)):
        if (filename.endswith(".jpg")):
            fileList.append(filename)
    folderNameLength = len(folder)
    frame = 0
    while frame < len(fileList):    # Loop over files. Supposed to do one loop per sequence

        filename = fileList[frame]  # first file of the sequence

        seqName = filename[6:11]        # e.g. "c0091"

        print("processing sequence "+seqName+"...")

        # Get number of files in sequence
        seqLength = 1
        while fileList[frame+seqLength][folderNameLength+1:folderNameLength+6]==seqName:
            seqLength+=1
            if frame+seqLength==len(fileList):
                break

        print("sequence length = "+str(seqLength))

        # Choose two different frames at random
        f1 = np.random.randint(seqLength)
        f2 = np.random.randint(seqLength)
        while(f1==f2):
            f1 = np.random.randint(seqLength)
            f2 = np.random.randint(seqLength)

        filename1 = fileList[frame+f1]
        filename2 = fileList[frame+f2]
        print("Chosen files: "+filename1+" ("+str(f1)+") and "+filename2+" ("+str(f2)+")")

        iuvFilename1 = filename1[:-4]+"_IUV.png"
        iuvFilename2 = filename2[:-4]+"_IUV.png"

        infoFilename = filename[:-10]+"info.mat"

        if not (os.path.isfile(folderPath + iuvFilename1) and os.path.isfile(folderPath + iuvFilename2)):
            print("WARNING: file with no infered densepose: "+filename1+" or "+filename2)
            frame+=seqLength
            continue

        

        # Load Images
        iuvIm1 = Image.open(folderPath+iuvFilename1)
        iuvMat1 = np.array(iuvIm1,dtype=np.float32)
        iuvIm2 = Image.open(folderPath+iuvFilename2)
        iuvMat2 = np.array(iuvIm2,dtype=np.float32)
                        # inIm = Image.open(inputPath+filename)
                        # inMat = np.array(inIm, dtype=np.float32)
                        # inMat = inMat/255

                        # padInMat = np.pad(inMat,((1,1),(1,1),(0,0)),mode='constant',constant_values=0)

        iuvpix1, uvec1, vvec1 = doIVUMatStuff(iuvMat1)
        iuvpix2, uvec2, vvec2 = doIVUMatStuff(iuvMat2)
        


        points_num1 = iuvpix1.shape[0]
        points_num2 = iuvpix2.shape[0]

        # array of input values on the template graph
        netInput = np.zeros((TEMPLATE_CARD,4),dtype=np.float32)
        netInput = netInput-1
        print("processing image 1...")
        # For every non null pixel:
        for i in range(points_num1):
            ii = iuvpix1[i,:][2]
            vv = iuvpix1[i,:][0]/256
            uu = iuvpix1[i,:][1]/256
            # Get face ind + bary coordinates
            FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
            writtenU = uvec1[i]
            writtenV = vvec1[i]
                            # imU = rawUvec[i]
                            # imV = rawVvec[i]
                            # imPatch = padInMat[imU:imU+3,imV:imV+3,:]
                            # imPatch = np.reshape(imPatch,-1)
                            # # print("imPatch = "+str(imPatch))

            # Simple strategy for now
            if netInput[FaceIndex,0]==-1:   # If no value on this node yet
                netInput[FaceIndex,0] = writtenU
                netInput[FaceIndex,1] = writtenV
                                # netInput[FaceIndex,2:] = imPatch
        print("processing image 2...")
        for i in range(points_num2):
            ii = iuvpix2[i,:][2]
            vv = iuvpix2[i,:][0]/256
            uu = iuvpix2[i,:][1]/256
            # Get face ind + bary coordinates
            FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
            writtenU = uvec2[i]
            writtenV = vvec2[i]
                            # imU = rawUvec[i]
                            # imV = rawVvec[i]
                            # imPatch = padInMat[imU:imU+3,imV:imV+3,:]
                            # imPatch = np.reshape(imPatch,-1)
                            # # print("imPatch = "+str(imPatch))

            # Simple strategy for now
            if netInput[FaceIndex,2]==-1:   # If no value on this node yet
                netInput[FaceIndex,2] = writtenU
                netInput[FaceIndex,3] = writtenV
                                # netInput[FaceIndex,2:] = imPatch
        # Add array to list
        inputList.append(netInput)

        frame+=seqLength

        # Get body shape
        infoMat = scipy.io.loadmat(folderPath+infoFilename)
        bodyShape = infoMat['shape'][:,0]
        gender = infoMat['gender'][0]               # array (1,)
        genBodyShape = np.concatenate((gender,bodyShape))
        # Add to list
        gtShapeList.append(genBodyShape)

        fileNum+=1

        # if fileNum==20:
        #     fileNum=0

        #     # pickle list
        #     with open(destPath+str(pklNum)+'_standard_input.pkl', 'wb') as fp:
        #         pickle.dump(inputList, fp)   
        #     # pickle list
        #     with open(destPath+str(pklNum)+'_gt_shape.pkl', 'wb') as fp:
        #         pickle.dump(gtShapeList, fp)
        #     pklNum+=1
        #     inputList=[]
        #     gtShapeList=[]
        #     print("Saved pickle "+str(pklNum))

        print("frame = "+str(frame)+" on "+str(len(fileList)))
    # pickle list
    with open(destPath+folder+'_standard_input.pkl', 'wb') as fp:
        pickle.dump(inputList, fp)   
    # pickle list
    with open(destPath+folder+'_gt_shape.pkl', 'wb') as fp:
        pickle.dump(gtShapeList, fp)


def getInput(imPath, iuvFilePath):
    # Load Images
    # iuvIm = Image.open(iuvFilePath)
    # iuvMat = np.array(iuvIm,dtype=np.float32)
    iuvMat = plt.imread(iuvFilePath)
    # inIm = Image.open(imPath)
    # inMat = np.array(inIm, dtype=np.float32)
    # inMat = plt.imread(imPath)
    inMat = plt.imread(iuvFilePath)
    inMat = inMat/255

    padInMat = np.pad(inMat,((1,1),(1,1),(0,0)),mode='constant',constant_values=0)

    iMat = iuvMat[:,:,2]
    sil = np.nonzero(iMat)
    iuvpix = iuvMat[sil]

    # Get (image) UV coordinates
    rawUvec = sil[0]
    rawVvec = sil[1]

    # Normalize them (using bounding box)
    umin = np.amin(rawUvec)
    vmin = np.amin(rawVvec)

    uvec = rawUvec-umin
    vvec = rawVvec-vmin

    uvmax = max(np.amax(uvec),np.amax(vvec))
    
    uvec = uvec / uvmax
    vvec = vvec / uvmax



    points_num = iuvpix.shape[0]
    
    baryCoords = np.zeros((points_num,3))
    faceInd = np.zeros(points_num)

    # array of input values on the template graph
    netInput = np.zeros((TEMPLATE_CARD,29),dtype=np.float32)
    netInput = netInput-1

    # For every non null pixel:
    for i in range(points_num):
        # uu,vv,ii = iuvpix[i,:]
        ii = iuvpix[i,:][2]
        vv = iuvpix[i,:][0]/256
        uu = iuvpix[i,:][1]/256
        # Get face ind + bary coordinates
        FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)

        # add bary coords and pix coords to face
        faceInd[i] = FaceIndex
        baryCoords[i,:] = [bc1,bc2,bc3]
        
        imU = rawUvec[i]
        imV = rawVvec[i]
        writtenU = uvec[i]
        writtenV = vvec[i]
        imPatch = padInMat[imU:imU+3,imV:imV+3,:]
        imPatch = np.reshape(imPatch,-1)
        # print("imPatch = "+str(imPatch))

        # Simple strategy for now
        if netInput[FaceIndex,0]==-1:   # If no value on this node yet
            netInput[FaceIndex,0] = writtenU
            netInput[FaceIndex,1] = writtenV
            netInput[FaceIndex,2:] = imPatch

    return netInput

def testPerm():
    binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/goodUV/"
    with open(binDumpPath+'adj.pkl', 'rb') as fp:
        adj_list = pickle.load(fp, encoding='latin1')
    with open(binDumpPath+'01_02_standard_input.pkl', 'rb') as fp:
        in_list = pickle.load(fp, encoding='latin1')
    with open(binDumpPath+'01_02_gt_shape.pkl', 'rb') as fp:
        gtshape_list = pickle.load(fp, encoding='latin1')
    with open(binDumpPath+'perm.pkl', 'rb') as fp:
        perm = pickle.load(fp, encoding='latin1')


    noisyFolder = "/morpheo-nas2/marmando/densepose/Test/"
    noisyFile = "male_template_dp.obj"
    V0,_,_, faces0, _ = load_mesh(noisyFolder, noisyFile, 0, False)

    # f_pos0 = getTrianglesBarycenter(V0, faces0, normalize=True)

    # destPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/"
    # with open(destPath+'face_pos.pkl', 'wb') as fp:
    #     pickle.dump(f_pos0, fp) 

    central_color = in_list[2][:,14:17]
    # central_color = in_list[0][:,0:3]
    # central_color[:,2]=0.0

    ds = gtshape_list[0]
    print("body shape = "+str(ds))

    newV, newF = getColoredMesh(V0, faces0, central_color)
    write_mesh(newV, newF, "/morpheo-nas2/marmando/ShapeRegression/Test/"+"centralColor2.obj")

    # nt = perm.shape[0]
    # print("perm shape = "+str(perm.shape))
    # n0 = faces0.shape[0]
    # print("n0 = %g, nt = %g"%(n0,nt))
    # print("perm slice: "+str(perm[:20]))
    # padding = np.zeros((nt-n0,3),dtype=np.int32)
    # padding = padding -1
    # faces = np.concatenate((faces0,padding),axis=0)
    # facesNew = faces[perm]
    # print("faces shape = "+str(faces.shape))
    # print("facesNew shape = "+str(facesNew.shape))
    # write_mesh(V0,facesNew,"/morpheo-nas2/marmando/ShapeRegression/Test/perm/test.obj")

def checkINDSPic(imMat):
    val = np.unique(imMat)
    if val.shape[0]>2:
        return False
    else:
        return True
    # return (np.sum(imMat)==0)


def countConnectedComponents(im):

    binIm = (im[:,:,0]>0)
    dilIm = ndimage.binary_dilation(binIm, iterations=4)
    _, nr_objects = ndimage.label(dilIm > 0)

    return nr_objects


def getRandominput(binPath, totalSampNum, mode='mesh'):

    in_list = []
    gtshape_list = []

    binFolderList = []
    binFolderCardList = []

    if mode=='mesh':
        extInputStr = "_standard_input.pkl"
    elif mode=='image':
        extInputStr = "_3dpos.pkl"

    for filename in os.listdir(binPath):

        if not filename.endswith(extInputStr):
            continue

        if filename.startswith("01_03"):
            print("WARNING, hardcoded input filtering")
            continue

        binFolder = filename[:-len(extInputStr)]


        binFolderList.append(binFolder)
        inFileName = filename
        shapeFileName = binFolder + "_gt_shape.pkl"

        with open(binPath+shapeFileName, 'rb') as fp:
            gtshape_list = pickle.load(fp, encoding='latin1')

        binFolderCardList.append(len(gtshape_list))

        # print(binFolder+": "+str(len(gtshape_list))+" examples")

    # Generate array to sample from
    binNumList = []
    for i in range(len(binFolderList)):
        binNum = np.full((binFolderCardList[i]),i,dtype=np.int32)
        sampNum = np.arange(binFolderCardList[i],dtype = np.int32)
        binArr = np.stack((binNum,sampNum),axis=1)
        # print("binArr shape = "+str(binArr.shape))
        binNumList.append(binArr)

    finalArr = np.concatenate(binNumList,axis=0)
    print("finalArr shape = "+str(finalArr.shape))

    randPerm = np.random.permutation(finalArr.shape[0])

    if totalSampNum>finalArr.shape[0]:
        totalSampNum = finalArr.shape[0]

    randPermSamp = randPerm[:totalSampNum]

    randPermSamp = np.sort(randPermSamp)
    # print("randPerSamp = "+str(randPermSamp))
    samples = finalArr[randPermSamp,:]
    # print("samples = "+str(samples))
    neededFolders = np.unique(samples[:,0])
    binFoldersArr = np.array(binFolderList)
    # print("neededFolders = "+str(neededFolders))
    neededFoldersName = binFoldersArr[neededFolders]

    curSampInd = 0
    curFolderInd = neededFolders[0]
    print("Loading "+str(neededFoldersName.shape[0])+" folders on "+str(len(binFolderList)))
    for f in range(neededFoldersName.shape[0]):
        curFolder = neededFoldersName[f]
        print("Loading "+curFolder+"...")
        curFolderInd = neededFolders[f]
        with open(binPath+curFolder+extInputStr, 'rb') as fp:
            cur_in_list = pickle.load(fp, encoding='latin1')
        with open(binPath+curFolder+'_gt_shape.pkl', 'rb') as fp:
            cur_gtshape_list = pickle.load(fp, encoding='latin1')

        while (curSampInd<totalSampNum) and (samples[curSampInd,0]==curFolderInd):
            in_list.append(cur_in_list[samples[curSampInd,1]])
            gtshape_list.append(cur_gtshape_list[samples[curSampInd,1]])
            curSampInd+=1

    return in_list, gtshape_list


def mainFunction():

    

    if not trainMode:
        # inputPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/17_07/"
        # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
        # pickleFolder(inputPath, binDumpPath)

        inputPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/17_06/"
        binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/processed_images/"
        pickleImages(inputPath, binDumpPath)

        # inputPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/"
        # folder = "32_18"
        # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/twoImages/"
        # pickleFolder2Images(inputPath, folder, binDumpPath)
    else:

        if RUNNING_MODE==0 or RUNNING_MODE==1 or RUNNING_MODE==10:     # Train network on template (0: shape, 1: gender, 10: bodyratio)

            # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/twoImages/"
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"

            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'perm.pkl', 'rb') as fp:
                perm = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
                face_pos = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'covariance.pkl', 'rb') as fp:
                covariance = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'male_node_dist.pkl', 'rb') as fp:
                nodeParamDisp = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            
            in_list, gtshape_list = getRandominput(binDumpPath, 3000, mode='mesh')


            face_pos = np.expand_dims(face_pos,axis=0)

            # Trim valid list
            valid_in_list = valid_in_list[:30]
            valid_gtshape_list = valid_gtshape_list[:30]

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            
            if RUNNING_MODE==0:
                trainBodyShapeNet(in_list, gtshape_list, adj_list, perm, face_pos, nodeParamDisp, valid_in_list, valid_gtshape_list, mode='shape')
            elif RUNNING_MODE==1:
                trainBodyShapeNet(in_list, gtshape_list, adj_list, perm, face_pos, nodeParamDisp, valid_in_list, valid_gtshape_list, mode='gender')
            elif RUNNING_MODE==10:
                trainBodyShapeNet(in_list, gtshape_list, adj_list, perm, face_pos, nodeParamDisp, valid_in_list, valid_gtshape_list, mode='bodyratio')

            print("Complete: mode = "+str(RUNNING_MODE)+", architecture "+str(ARCHITECTURE)+", net path = "+RESULTS_PATH)

        elif RUNNING_MODE==2 or RUNNING_MODE==3 or RUNNING_MODE==12:      # Train Network on images alone (2: shape, 3: gender, 12: bodyratio)

            # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/rawImages/"
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/processed_images/"

            in_list, gtshape_list = getRandominput(binDumpPath, 5000, mode='image')
            
            with open(binDumpPath+'01_03_3dpos.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            

            if RUNNING_MODE==2:
                trainImageNet(in_list, gtshape_list, valid_in_list, valid_gtshape_list, mode='shape')
            elif RUNNING_MODE==3:
                trainImageNet(in_list, gtshape_list, valid_in_list, valid_gtshape_list, mode='gender')
            elif RUNNING_MODE==12:
                trainImageNet(in_list, gtshape_list, valid_in_list, valid_gtshape_list, mode='bodyratio')

            print("Complete: mode = "+str(RUNNING_MODE)+", architecture "+str(ARCHITECTURE)+", net path = "+RESULTS_PATH)

        elif RUNNING_MODE==4:       # Test network
            # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/twoImages/"
            # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/wololo/"
            # with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
            #     in_list = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
                in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                gtshape_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'perm.pkl', 'rb') as fp:
                perm = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
                face_pos = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
            face_pos = np.expand_dims(face_pos,axis=0)

            inMat = np.concatenate(in_list,axis=0)
            inMat = inMat[:20,:,:]
            # myIn = getInput("/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/01_03/01_03_c0003_img017.jpg", "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/01_03/01_03_c0003_img017_IUV.png")
            # myFeatures = testBodyShapeNet(myIn, adj_list, perm, face_pos)

            # mode='shape'
            mode='bodyratio'
            myShape, myFeatures = testBodyShapeNet(inMat, adj_list, perm, face_pos, mode)
            print("inMat shape = "+str(inMat.shape))
            print("myFeatures shape = "+str(myFeatures.shape))
            # print("myShape =  "+str(myShape))
            print("myShape shape =  "+str(myShape.shape))

            gtshape = np.stack(gtshape_list,axis=0)
            # --- Get body ratio
            tfH0 = np.reshape(H0, [1,1])
            tfW0 = np.reshape(W0, [1,1])
            tfD0 = np.reshape(D0, [1,1])
            tfHMat = np.reshape(HMat, [10,1])
            tfWMat = np.reshape(WMat, [10,1])
            tfDMat = np.reshape(DMat, [10,1])

            bH = np.matmul(gtshape[:,1:],tfHMat)
            bW = np.matmul(gtshape[:,1:],tfWMat)
            bD = np.matmul(gtshape[:,1:],tfDMat)

            bH = bH + tfH0
            bD = bD + tfD0
            bW = bW + tfW0
            gtbodyratio = np.divide(bD,bH)

            meanRatio = np.mean(gtbodyratio)

            for inInd in range(inMat.shape[0]):
                
                # print("inInd = "+str(inInd))

                if gtshape[inInd,0]==0:
                    continue

                # print("myShape =  "+str(myShape[inInd]))
                
                print("predicted BR = %f, GT BR = %f"%(myShape[inInd]/meanRatio, gtbodyratio[inInd]/meanRatio))
            return

            feature_color = myFeatures
            # feature_color[:,1:]=0.0
            # feature_color = np.concatenate((myFeatures,np.zeros((myFeatures.shape[0],1),dtype=np.float32)),axis=1)
            
            # for dim in range(3):
            #     minDim = np.amin(feature_color[:,dim])
            #     maxDim = np.amax(feature_color[:,dim])
            #     # print("maxDim = "+str(maxDim))
            #     feature_color[:,dim] = feature_color[:,dim]-minDim
            #     feature_color[:,dim] = feature_color[:,dim]/(maxDim-minDim)
            #     maxDim = np.amax(feature_color[:,dim])
            #     # print("maxDim 2 = "+str(maxDim))
            # # feature_color[:,0] = 0.0
            
            totalMax = np.amax(feature_color)
            totalMin = np.amin(feature_color)
            print("totalMin = %f, totalMax = %f"%(totalMin, totalMax))
            feature_color = feature_color-totalMin
            feature_color = feature_color/(totalMax-totalMin)

            # feature_color = feature_color[:,:3]
            # feature_color = feature_color[:,3:6]
            # feature_color = feature_color[:,6:9]
            # feature_color = feature_color[:,9:12]
            # feature_color = feature_color[:,12:15]
            # feature_color = feature_color[:,15:18]
            # feature_color = feature_color[:,18:21]
            # feature_color = feature_color[:,21:24]
            # feature_color = feature_color[:,24:27]
            feature_color = feature_color[:,27:30]

            noisyFolder = "/morpheo-nas2/marmando/densepose/Test/"
            noisyFile = "male_template_dp.obj"
            V0,_,_, faces0, _ = load_mesh(noisyFolder, noisyFile, 0, False)
            newV, newF = getColoredMesh(V0, faces0, feature_color)
            write_mesh(newV, newF, "/morpheo-nas2/marmando/ShapeRegression/Test/"+"test.obj")

        elif RUNNING_MODE==5:       # Check body shape sampling
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            gtshape_list = []
            

            # # --- "loading bin dump" version ---
            extInputStr = "_gt_shape.pkl"
            for filename in sorted(os.listdir(binDumpPath)):

                if not filename.endswith(extInputStr):
                    continue
                with open(binDumpPath+filename, 'rb') as fp:
                    gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_01_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_02_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_04_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_05_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_06_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_08_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'02_01_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'02_03_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'02_04_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')

            
            # # --- "Loading input" version ---
            # inputPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/"
            # for folder in os.listdir(inputPath):

            #     folderPath = inputPath + folder + "/"

            #     for filename in os.listdir(folderPath):

            #         if filename.endswith("info.mat"):
            #             infoMat = scipy.io.loadmat(folderPath+filename)
            #             bodyShape = infoMat['shape'][:,0]
            #             gender = infoMat['gender'][0]               # array (1,)
            #             genBodyShape = np.concatenate((gender,bodyShape))

            #             gtshape_list.append(genBodyShape)

            #     # print("List size = %g"%(len(gtshape_list)))

            #     # print(folderPath)

            shapestack = np.stack(gtshape_list,axis=0)
            print("shapestack shape = "+str(shapestack.shape))


            mean_shape = np.mean(shapestack, axis=0)
            std_shape = np.var(shapestack, axis=0)
            print("mean shape = "+str(mean_shape))
            print("var shape = "+str(std_shape))

            shapeNum=0
            uniqueShape=[]
            maleNum=0
            femaleNum=0

            for i in range(shapestack.shape[0]):
                # print("shape %g... (%g)"%(i, shapeNum))
                curShape = shapestack[i,:]
                if shapeNum==0:
                    uniqueShape.append(curShape)
                    shapeNum += 1

                else:
                    for k in range(len(uniqueShape)):
                        if (curShape==uniqueShape[k]).all():
                            # uniqueShape.append(curShape)
                            # shapeNum += 1
                            break
                        if k==(len(uniqueShape)-1):
                            uniqueShape.append(curShape)
                            shapeNum += 1

                if curShape[0]==0:
                    femaleNum += 1
                else:
                    maleNum += 1


            print(str(shapeNum) + " unique shapes for %g examples"%(shapestack.shape[0]))
            print("%g males, %g females"%(maleNum,femaleNum))


            # uniqueShape covariance:
            uniqueShapeMat = np.stack(uniqueShape,axis=1)
            print("uniqueShapeMat shape = "+str(uniqueShapeMat.shape))

            unCov = np.cov(uniqueShapeMat)

            # with open(binDumpPath+'covariance.pkl', 'wb') as fp:
            #     pickle.dump(unCov, fp)

            unCov = 1000 * unCov

            # print("covariance matrix = "+str(unCov))

            print("unique shapes covariance matrix: \n")
            prtStr = ""
            for i in range(unCov.shape[0]):
                prtStr+="[ "
                # print("[ ")
                for j in range(unCov.shape[1]):
                    prtStr+="%g \t"%(int(unCov[i,j]))
                    # print("%g "%(unCov[i,j]))
                prtStr+="]\n"
                # print("]\n")
            print(prtStr)

            maleShape = uniqueShapeMat[uniqueShapeMat[:,0]==1]




            # totCov = np.cov(np.transpose(shapestack))
            # totCov = 1000 * totCov

            # print("total shapes covariance matrix: \n")
            # prtStr = ""
            # for i in range(totCov.shape[0]):
            #     prtStr+="[ "
            #     # print("[ ")
            #     for j in range(totCov.shape[1]):
            #         prtStr+="%g \t"%(int(totCov[i,j]))
            #         # print("%g "%(unCov[i,j]))
            #     prtStr+="]\n"
            #     # print("]\n")
            # print(prtStr)


        elif RUNNING_MODE==6:       # Filter data

            imagePathRoot = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run2/"
            discPathRoot = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/discarded_train_images/run2/"
            

            for foldName in os.listdir(imagePathRoot):

                # if not foldName=="15_02":
                #     continue
                # foldName = "05_08"
                print("processing "+foldName+"...")
                imagePath = imagePathRoot + foldName + "/"
                discPath = discPathRoot + foldName + "/"

                if not os.path.exists(discPath):
                    os.makedirs(discPath)

                totalFiles=0
                triggeredFiles=0
                for filename in sorted(os.listdir(imagePath)):

                    # if fileNum>=10:
                    #     break

                    if not (filename.endswith("IUV.png")):
                        continue

                    iuvMat = plt.imread(imagePath+filename)

                    # iuvIm = Image.open(imagePath+filename)
                    # iuvMat = np.array(iuvIm,dtype=np.float32)
                    indsFile = filename[:-7] + "INDS.png"


                    # If INDS file is present, use it
                    if os.path.exists(imagePath+indsFile):

                        indsMat = plt.imread(imagePath+indsFile)
                        if (not checkINDSPic(indsMat)):
                            # print("File "+filename+": "+str(numComp)+" components")
                            print("File "+filename+": removed")
                            triggeredFiles+=1
                            shutil.move(imagePath+filename, discPath+filename)

                    # Else, use connected components
                    else:

                        numComp = countConnectedComponents(iuvMat)
                        if (numComp>1):
                            print("File "+filename+": "+str(numComp)+" components")
                            triggeredFiles+=1
                            shutil.move(imagePath+filename, discPath+filename)

                    # Also check that at least something is detected
                    if (np.sum(iuvMat)==0):
                        print("File "+filename+": empty densepose file: removed")
                        triggeredFiles+=1
                        shutil.move(imagePath+filename, discPath+filename)

                    totalFiles+=1
                print("%g out of %g"%(triggeredFiles,totalFiles))

        elif RUNNING_MODE==7:       # Check adj dim

            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')

            coarsNum=0
            new_adj_list = []
            for adj in adj_list:
                print("adj shape = "+str(adj.shape))
                initK = adj.shape[2]
                print("adj num %g, initial K = %g"%(coarsNum,initK))

                curCol = initK-1

                adjSlice = adj[:,:,curCol]

                test = np.all(adjSlice==0)
                while test:
                    curCol -= 1    
                    adjSlice = adj[:,:,curCol]
                    test = np.all(adjSlice==0)

                print("new K = %g"%(curCol+1))

                newAdj = adj[:,:,:curCol+1]
                trash = adj[:,:,curCol+1:]
                print("trash sum = "+str(np.sum(trash)))

                new_adj_list.append(newAdj)

                coarsNum +=1

            with open(binDumpPath+'_adj2.pkl', 'wb') as fp:
                pickle.dump(new_adj_list, fp)

        elif RUNNING_MODE==8:       # Play with binary data

            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            in_list = []
            gtshape_list = []

            binFolderList = []
            binFolderCardList = []
            for filename in os.listdir(binDumpPath):

                if not filename.endswith("standard_input.pkl"):
                    continue

                binFolder = filename[:-19]

                binFolderList.append(binFolder)
                inFileName = filename
                shapeFileName = binFolder + "_gt_shape.pkl"

                with open(binDumpPath+shapeFileName, 'rb') as fp:
                    gtshape_list = pickle.load(fp, encoding='latin1')

                binFolderCardList.append(len(gtshape_list))

                print(binFolder+": "+str(len(gtshape_list))+" examples")

            # Generate array to sample from
            binNumList = []
            for i in range(len(binFolderList)):
                binNum = np.full((binFolderCardList[i]),i,dtype=np.int32)
                sampNum = np.arange(binFolderCardList[i],dtype = np.int32)
                binArr = np.stack((binNum,sampNum),axis=1)
                print("binArr shape = "+str(binArr.shape))
                binNumList.append(binArr)

            finalArr = np.concatenate(binNumList,axis=0)
            print("finalArr shape = "+str(finalArr.shape))

            randPerm = np.random.permutation(finalArr.shape[0])
            randPermSamp = randPerm[:10]

            randPermSamp = np.sort(randPermSamp)

            samples = finalArr[randPermSamp,:]

            neededFolders = np.unique(samples[:,0])

            print(samples)

            neededFolders = np.unique(samples[:,0])
            print("neededFolders = "+str(neededFolders))

            binFoldersArr = np.array(binFolderList)
            print("binFoldersArr = "+str(binFoldersArr))

            neededFoldersName = binFoldersArr[neededFolders]

            print("neededFoldersName = "+str(neededFoldersName))






            # with open(binDumpPath+'adj.pkl', 'rb') as fp:
            #     adj_list = pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_05_standard_input.pkl', 'rb') as fp:
            #     in_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_05_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_06_standard_input.pkl', 'rb') as fp:
            #     in_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_06_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_08_standard_input.pkl', 'rb') as fp:
            #     in_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_08_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'05_01_standard_input.pkl', 'rb') as fp:
            #     in_list += pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'05_01_gt_shape.pkl', 'rb') as fp:
            #     gtshape_list += pickle.load(fp, encoding='latin1')

            # with open(binDumpPath+'perm.pkl', 'rb') as fp:
            #     perm = pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
            #     face_pos = pickle.load(fp, encoding='latin1')

            # with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
            #     valid_in_list = pickle.load(fp, encoding='latin1')
            # with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
            #     valid_gtshape_list = pickle.load(fp, encoding='latin1')

        elif RUNNING_MODE==9:       # Temp: measure loading time: bin dump vs png

            startTime = time.time()
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/processed_images/"
            with open(binDumpPath+'02_01_3dpos.pkl', 'rb') as fp:
                list3dPos = pickle.load(fp, encoding='latin1')

            elapsedTime = time.time() - startTime
            print('[{}] finished in {} ms'.format("bin dump", int(elapsedTime * 1000)))

            startTime = time.time()
            imList = []
            imDumpPath = "/morpheo-nas2/marmando/ShapeRegression/ImagesDump/"
            for filename in os.listdir(imDumpPath+"02_01/"):
                myIm = plt.imread(imDumpPath+"02_01/"+filename)
                imList.append(myIm)
            elapsedTime = time.time() - startTime
            print('[{}] finished in {} ms'.format("png loading", int(elapsedTime * 1000)))

        elif RUNNING_MODE==11:      # Test image network
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/processed_images/"
            # with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
            #     in_list = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_03_3dpos.pkl', 'rb') as fp:
                in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                gtshape_list = pickle.load(fp, encoding='latin1')

            plt.imsave("/morpheo-nas2/marmando/ShapeRegression/Test/imageNetInput.png",in_list[0])
            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)

            inMat = np.concatenate(in_list,axis=0)
            inMat = inMat[:10,:]
            myShape, myFeatures = testImageNet(inMat, mode='bodyratio')

            gtshape = np.stack(gtshape_list,axis=0)
            print("myFeatures shape = "+str(myFeatures.shape))

            # --- Get body ratio
            tfH0 = np.reshape(H0, [1,1])
            tfW0 = np.reshape(W0, [1,1])
            tfD0 = np.reshape(D0, [1,1])
            tfHMat = np.reshape(HMat, [10,1])
            tfWMat = np.reshape(WMat, [10,1])
            tfDMat = np.reshape(DMat, [10,1])

            bH = np.matmul(gtshape[:,1:],tfHMat)
            bW = np.matmul(gtshape[:,1:],tfWMat)
            bD = np.matmul(gtshape[:,1:],tfDMat)

            bH = bH + tfH0
            bD = bD + tfD0
            bW = bW + tfW0
            gtbodyratio = np.divide(bD,bH)

            meanRatio = np.mean(gtbodyratio)
            print("mean body ratio = %f"%(meanRatio))

            for inInd in range(inMat.shape[0]):
                
                # print("inInd = "+str(inInd))

                if gtshape[inInd,0]==0:
                    continue

                # print("myShape =  "+str(myShape[inInd]))
                
                print("predicted BR = %f, GT BR = %f"%(myShape[inInd]/meanRatio, gtbodyratio[inInd]/meanRatio))

                

                # print("gt ratio = "+str(gtbodyratio))


            # # print("Filter = "+str(myFeatures[0,0,:,:]))
            # feature_color = myFeatures
            # # feature_color[:,1:]=0.0
            # # feature_color = np.concatenate((myFeatures,np.zeros((myFeatures.shape[0],1),dtype=np.float32)),axis=1)
            
            # # for dim in range(3):
            # #     minDim = np.amin(feature_color[:,dim])
            # #     maxDim = np.amax(feature_color[:,dim])
            # #     # print("maxDim = "+str(maxDim))
            # #     feature_color[:,dim] = feature_color[:,dim]-minDim
            # #     feature_color[:,dim] = feature_color[:,dim]/(maxDim-minDim)
            # #     maxDim = np.amax(feature_color[:,dim])
            # #     # print("maxDim 2 = "+str(maxDim))
            # # # feature_color[:,0] = 0.0
            
            # totalMax = np.amax(feature_color)
            # totalMin = np.amin(feature_color)
            # print("totalMin = %f, totalMax = %f"%(totalMin, totalMax))
            # feature_color = feature_color-totalMin
            # feature_color = feature_color/(totalMax-totalMin)

            # feature_color = feature_color[:,:,:3]
            # # feature_color = feature_color[:,:,3:6]
            # # feature_color = feature_color[:,6:9]
            # # feature_color = feature_color[:,9:12]
            # # feature_color = feature_color[:,12:15]
            # # feature_color = feature_color[:,15:18]
            # # feature_color = feature_color[:,18:21]
            # # feature_color = feature_color[:,21:24]
            # # feature_color = feature_color[:,24:27]
            # # feature_color = feature_color[:,27:30]
            # print("feature_color shape = "+str(feature_color.shape))
            # plt.imsave("/morpheo-nas2/marmando/ShapeRegression/Test/imageNetTest.png",feature_color)

        elif RUNNING_MODE==13 or RUNNING_MODE==15:      # Train image encoder

            # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/rawImages/"
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/processed_images/"

            in_list, gtshape_list = getRandominput(binDumpPath, 5000, mode='image')
            
            with open(binDumpPath+'01_03_3dpos.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            
            if RUNNING_MODE==13:
                trainImageEncoder(in_list, valid_in_list, gtshape_list, valid_gtshape_list, mode='encoder')
            elif RUNNING_MODE==15:
                trainImageEncoder(in_list, valid_in_list, gtshape_list, valid_gtshape_list, mode='bodyratio')

            print("Complete: mode = "+str(RUNNING_MODE)+", architecture "+str(ARCHITECTURE)+", net path = "+RESULTS_PATH)

        elif RUNNING_MODE==14:      # Test image encoder
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/processed_images/"
            # with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
            #     in_list = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_03_3dpos.pkl', 'rb') as fp:
                in_list = pickle.load(fp, encoding='latin1')

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)

            inMat = np.concatenate(in_list,axis=0)
            inMat = inMat[:3,:]

            outIm, outCode = testImageEncoder(inMat)

            for inInd in range(inMat.shape[0]):
                
                plt.imsave("/morpheo-nas2/marmando/ShapeRegression/Test/encoder/gen_"+str(inInd)+".png",outIm[inInd,:,:,:])
                # plt.imsave("/morpheo-nas2/marmando/ShapeRegression/Test/encoder/in_"+str(inInd)+".png",inMat[inInd,:,:,:])
                
        if RUNNING_MODE==16:     # Train mesh encoder

            # binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/twoImages/"
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"

            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'perm.pkl', 'rb') as fp:
                perm = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
                face_pos = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'covariance.pkl', 'rb') as fp:
                covariance = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'male_node_dist.pkl', 'rb') as fp:
                nodeParamDisp = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            
            with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
                in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                gtshape_list = pickle.load(fp, encoding='latin1')
            # in_list, gtshape_list = getRandominput(binDumpPath, 5000, mode='mesh')


            face_pos = np.expand_dims(face_pos,axis=0)

            # Trim valid list
            valid_in_list = valid_in_list[:30]
            valid_gtshape_list = valid_gtshape_list[:30]

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            
            if RUNNING_MODE==16:
                trainMeshEncoder(in_list, gtshape_list, adj_list, perm, face_pos, nodeParamDisp, valid_in_list, valid_gtshape_list, mode='shape')
            

            print("Complete: mode = "+str(RUNNING_MODE)+", architecture "+str(ARCHITECTURE)+", net path = "+RESULTS_PATH)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=int, default=0)
    #parser.add_argument('--dataset_path')
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--net_name', type=str, default='net')
    parser.add_argument('--running_mode', type=int, default=0)
    parser.add_argument('--coarsening_steps', type=int, default=3)
    
    if not trainMode:
        DP = dp_utils.DensePoseMethods()
    TEMPLATE_CARD = 13774

    vTOP = 414
    vBOT = 3353
    vL = 800
    vR = 4288
    vFRONT = 3500
    vBACK = 3021
    H0 = 1.78314602
    W0 = 0.324824
    D0 = 0.24939099
    HMat = np.array([-0.07496882,  0.01156511,  0.01216906, -0.00113598, -0.00349156,
        0.00107003, -0.00032398,  0.00224208, -0.00145152,  0.00017297])
    WMat = np.array([-0.01917022, -0.02803352,  0.00578245, -0.0047634 ,  0.00466483,
        0.0039943 ,  0.00313493,  0.002968  ,  0.00140102, -0.00100837])
    DMat = np.array([-0.01555526, -0.02619525,  0.005661  , -0.006553  ,  0.00015785,
        0.00201796,  0.00635979,  0.00187502,  0.00196823, -0.00238279])


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
    # testPerm()



