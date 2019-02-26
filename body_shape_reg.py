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

def trainBodyShapeNet(input_list, gtshape_list, graph_adj, perm, face_pos, valid_input_list, valid_gtshape_list, mode='gender'):

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
    
    NUM_IN_CHANNELS = input_list[0].shape[1]
    NUM_IN_CHANNELS = 29

    BATCH_SIZE = 5

    # Version without extra dim
    if False:
        graph_card = np.zeros(len(graph_adj), dtype=np.int32)
        for g in range(len(graph_adj)):
            graph_adj[g] = np.squeeze(graph_adj[g])
            print("graph_adj["+str(g)+"] shape = "+str(graph_adj[g].shape))
            graph_card[g] = graph_adj[g].shape[0]

        K_faces = graph_adj[0].shape[1]

        # training data
        fn_ = tf.placeholder('float32', shape=[TEMPLATE_CARD, NUM_IN_CHANNELS], name='fn_')
        node_pos = tf.placeholder(tf.float32, shape=[TEMPLATE_CARD, 3], name='node_pos')

        fadj0 = tf.placeholder(tf.int32, shape=[graph_card[0], K_faces], name='fadj0')
        fadj1 = tf.placeholder(tf.int32, shape=[graph_card[1], K_faces], name='fadj1')
        fadj2 = tf.placeholder(tf.int32, shape=[graph_card[2], K_faces], name='fadj2')
        fadj3 = tf.placeholder(tf.int32, shape=[graph_card[3], K_faces], name='fadj2')

        perm_ = tf.placeholder(tf.int32, shape=[graph_card[0]], name='perm')
        gtshapegender_ = tf.placeholder('float32', shape=[11], name='tfn_')

        print("graph_card = "+str(graph_card))
        print("TEMPLATE_CARD = "+str(TEMPLATE_CARD))

        fullfn = tf.concat((fn_,node_pos), axis=-1)

        padding = tf.zeros([graph_card[0]-TEMPLATE_CARD,NUM_IN_CHANNELS+3])
        ext_fn = tf.concat((fullfn, padding), axis=0)
        perm_fn = tf.gather(ext_fn,perm_)

        gtgender = gtshapegender_[0]
        gtshape = gtshapegender_[1:]

    else:
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

        node_pos_tiled = tf.tile(node_pos,[BATCH_SIZE,1,1])

        print("graph_card = "+str(graph_card))
        print("TEMPLATE_CARD = "+str(TEMPLATE_CARD))

        empty_and_fake_nodes = tf.equal(fn_[:,:,0],-1)
        print("empty_and_fake_nodes shape = "+str(empty_and_fake_nodes.shape))
        empty_and_fake_nodes = tf.tile(tf.expand_dims(empty_and_fake_nodes,-1),[1,1,NUM_IN_CHANNELS])

        # new_fn = tf.where(empty_and_fake_nodes,1000*fn_,fn_)

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

    
    # fadj0_t = tf.tile(fadj0,[BATCH_SIZE,1,1])
    # fadj1_t = tf.tile(fadj1,[BATCH_SIZE,1,1])
    # fadj2_t = tf.tile(fadj2,[BATCH_SIZE,1,1])
    # fadj3_t = tf.tile(fadj3,[BATCH_SIZE,1,1])

    # fadjs = [fadj0_t,fadj1_t,fadj2_t,fadj3_t]
    fadjs = [fadj0,fadj1,fadj2,fadj3]
    



    



    if mode=='gender':
        with tf.variable_scope("model_gender_class"):
            pmale = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob, mode='gender')

        with tf.device(DEVICE):
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
            print("WARNING!!!! Hard-coded loss change (1st param)")
            shapeLoss = mseLoss(predicted_shape[:,:2],gtshape[:,:2])

            print("shapeLoss shape = "+str(shapeLoss.shape))

            crossEntropyLoss = shapeLoss
            train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
            
            isNanPmale = tf.reduce_any(tf.is_nan(predicted_shape), name="isNanPmale")

    batch_loss = tf.reduce_sum(crossEntropyLoss)

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

    SAVEITER = 500

    with tf.device(DEVICE):
        # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        lossArray = np.zeros([int(SAVEITER/10),2])
        last_loss = 0
        lossArrayIter = 0
        for iter in range(NUM_ITERATIONS):

            # Batch version...
            inbatchlist = []
            gtbatchlist = []
            for b in range(BATCH_SIZE):
                batch_num = random.randint(0,EXAMPLE_NUM-1)

                # LIMIT TO MALE EXAMPLES
                # print("gtshape_list el shape = "+str(gtshape_list[batch_num].shape))
                while gtshape_list[batch_num][0,0]==0:
                    batch_num = random.randint(0,EXAMPLE_NUM-1)

                inbatchlist.append(input_list[batch_num])
                gtbatchlist.append(gtshape_list[batch_num])


            in_batch = np.concatenate(inbatchlist, axis=0)
            shape_batch = np.concatenate(gtbatchlist, axis=0)

            # train_fd = {fn_: input_list[batch_num], gtshapegender_: gtshape_list[batch_num], keep_prob:1}
            train_fd = {fn_: in_batch, gtshapegender_: shape_batch, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, keep_prob:1}


            # One-example version
                    # # Get random sample from training dictionary
                    # batch_num = random.randint(0,EXAMPLE_NUM-1)

                    # train_fd = {fn_: input_list[batch_num], fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], gtshapegender_: gtshape_list[batch_num], perm_: perm, node_pos:face_pos, keep_prob:1}

            # Show smoothed training loss
            if ((iter%10 == 0) and (iter>0)):
                train_loss = train_loss/train_samp
                

                print("Iteration %d, training loss %g"%(iter, train_loss))

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


            # Compute validation loss
            if True:
                if (iter%20 ==0) and (iter>0):
                    valid_loss = 0
                    for vbm in range(VALID_EXAMPLE_NUM):
                        
                        valid_inbatchlist = []
                        valid_gtbatchlist = []
                        for b in range(BATCH_SIZE):
                            # batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                            # valid_inbatchlist.append(valid_input_list[batch_num])
                            # valid_gtbatchlist.append(valid_gtshape_list[batch_num])
                            valid_inbatchlist.append(valid_input_list[b])
                            valid_gtbatchlist.append(valid_gtshape_list[b])


                        valid_in_batch = np.concatenate(valid_inbatchlist, axis=0)
                        valid_shape_batch = np.concatenate(valid_gtbatchlist, axis=0)
                        
                        valid_fd = {fn_: valid_in_batch, gtshapegender_: valid_shape_batch, fadj0: graph_adj[0], fadj1: graph_adj[1], fadj2: graph_adj[2], fadj3: graph_adj[3], perm_: perm, node_pos:face_pos, keep_prob:1}


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
                lossArray = np.zeros([int(SAVEITER/10),2]) 
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

    if mode=='gender':
        with tf.variable_scope("model_gender_class"):
            fn_centered = fn_ * 2 - 1
            pmale, testFeatures = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob)
        
        with tf.device(DEVICE):
            totalLogl = loglikelihood(tf.reshape(pmale,[-1]),gtgender)
            crossEntropyLoss = -totalLogl
            train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
            isNanPmale = tf.reduce_any(tf.is_nan(pmale), name="isNanPmale")
        
    elif mode=='shape':
        with tf.variable_scope("model_gender_class"):
            fn_centered = fn_ * 2 - 1
            predicted_shape = get_image_conv_model_gender_class(fn_centered, ARCHITECTURE, keep_prob)

        with tf.device(DEVICE):
            shapeLoss = mseLoss(predicted_shape,gtshape)
            crossEntropyLoss = shapeLoss
            train_step = tf.train.AdamOptimizer().minimize(crossEntropyLoss, global_step=batch)
            isNanPmale = tf.reduce_any(tf.is_nan(predicted_shape), name="isNanPmale")


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

    with tf.device(DEVICE):
        # lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        lossArray = np.zeros([int(SAVEITER/10),2])
        last_loss = 0
        lossArrayIter = 0
        for iter in range(NUM_ITERATIONS):


            # Get random sample from training dictionary
            # Hard-coded batch size
            inbatchlist = []
            gtbatchlist = []
            for b in range(BATCH_SIZE):
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
            if ((iter%10 == 0) and (iter>0)) or (iter<10 and iter>0):
                train_loss = train_loss/train_samp
                

                print("Iteration %d, training loss %g"%(iter, train_loss))

                lossArray[lossArrayIter,0]=train_loss
                lossArrayIter+=1
                train_loss=0
                train_samp=0
                pred_pmale, loglklhd, testF = sess.run([pmale, totalLogl, testFeatures], feed_dict=train_fd)
                # # print("pred_pmale shape = "+str(pred_pmale.shape))
                # # print("pred_pmale[0] = "+str(pred_pmale[0]))
                # pred_pmale_val = pred_pmale[0]
                print("Predicted male probability = "+str(pred_pmale) + ", Ground truth = " + str(shape_batch[:,0]))
                # print("loglikelihood = "+str(loglklhd))
                print("test Features = "+str(testF))
                # # print("invalues: (yp = %g, ygt = %g). log(yp) = %g"%(valYp,valYgt, valLogYp))

            
            # marg_train_loss = crossEntropyLoss.eval(feed_dict=train_fd)

            marg_train_loss = sess.run(tf.reduce_sum(crossEntropyLoss), feed_dict=train_fd)
            train_loss += marg_train_loss
            train_samp+=1


            # Compute validation loss
            if (iter%20 ==0) and (iter>0):
                valid_loss = 0
                
                valid_inbatchlist = []
                valid_gtbatchlist = []
                for b in range(BATCH_SIZE):
                    batch_num = random.randint(0,VALID_EXAMPLE_NUM-1)
                    valid_inbatchlist.append(valid_input_list[batch_num])
                    valid_gtbatchlist.append(valid_gtshape_list[batch_num])


                valid_in_batch = np.concatenate(valid_inbatchlist, axis=0)
                valid_shape_batch = np.concatenate(valid_gtbatchlist, axis=0)
                
                valid_fd = {fn_: valid_in_batch, gtshapegender_: valid_shape_batch, keep_prob:1}
                # valid_loss = crossEntropyLoss.eval(feed_dict=valid_fd)
                valid_loss = sess.run(tf.reduce_sum(crossEntropyLoss),feed_dict=valid_fd)
                
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
                lossArray = np.zeros([int(SAVEITER/10),2]) 
                lossArrayIter=0
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")
    f.close()



def testBodyShapeNet(inGraph, graph_adj, perm, face_pos):

    random_seed = 0
    np.random.seed(random_seed)

    # sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if(FLAGS.debug):    #launches debugger at every sess.run() call
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/disk/marmando/tmp/")

    if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

    
    NUM_IN_CHANNELS = inGraph.shape[1]
    NUM_IN_CHANNELS = 29

    # Version without extra dim
    if False:
        graph_card = np.zeros(len(graph_adj), dtype=np.int32)
        for g in range(len(graph_adj)):
            graph_adj[g] = np.squeeze(graph_adj[g])
            print("graph_adj["+str(g)+"] shape = "+str(graph_adj[g].shape))
            graph_card[g] = graph_adj[g].shape[0]

        K_faces = graph_adj[0].shape[1]

        # training data
        fn_ = tf.placeholder('float32', shape=[TEMPLATE_CARD, NUM_IN_CHANNELS], name='fn_')
        node_pos = tf.placeholder(tf.float32, shape=[TEMPLATE_CARD, 3], name='node_pos')

        fadj0 = tf.placeholder(tf.int32, shape=[graph_card[0], K_faces], name='fadj0')
        fadj1 = tf.placeholder(tf.int32, shape=[graph_card[1], K_faces], name='fadj1')
        fadj2 = tf.placeholder(tf.int32, shape=[graph_card[2], K_faces], name='fadj2')
        fadj3 = tf.placeholder(tf.int32, shape=[graph_card[3], K_faces], name='fadj2')

        perm_ = tf.placeholder(tf.int32, shape=[graph_card[0]], name='perm')
        gtshapegender_ = tf.placeholder('float32', shape=[11], name='tfn_')

        print("graph_card = "+str(graph_card))
        print("TEMPLATE_CARD = "+str(TEMPLATE_CARD))

        fullfn = tf.concat((fn_,node_pos), axis=-1)

        padding = tf.zeros([graph_card[0]-TEMPLATE_CARD,NUM_IN_CHANNELS+3])
        ext_fn = tf.concat((fullfn, padding), axis=0)
        perm_fn = tf.gather(ext_fn,perm_)

        gtgender = gtshapegender_[0]
        gtshape = gtshapegender_[1:]

    else:
        graph_card = np.zeros(len(graph_adj), dtype=np.int32)
        K_faces = np.zeros(len(graph_adj), dtype=np.int32)
        for g in range(len(graph_adj)):
            print("graph_adj["+str(g)+"] shape = "+str(graph_adj[g].shape))
            graph_card[g] = graph_adj[g].shape[1]
            K_faces[g] = graph_adj[g].shape[2]

        # K_faces = graph_adj[0].shape[2]

        # training data
        fn_ = tf.placeholder('float32', shape=[1,TEMPLATE_CARD, NUM_IN_CHANNELS], name='fn_')
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
        fullfn = tf.concat((fn_centered,node_pos), axis=-1)

        padding = tf.zeros([1,graph_card[0]-TEMPLATE_CARD,NUM_IN_CHANNELS+3])
        padding = padding-3
        ext_fn = tf.concat((fullfn, padding), axis=1)
        perm_fn = tf.gather(ext_fn,perm_, axis=1)



    keep_prob = tf.placeholder(tf.float32)
    
    batch = tf.Variable(0, trainable=False)

    fadjs = [fadj0,fadj1,fadj2,fadj3]

    with tf.device(DEVICE):
        with tf.variable_scope("model_gender_class"):
            # pmale, testFeatures = get_model_gender_class(perm_fn, fadjs, ARCHITECTURE, keep_prob)
            predicted_shape, testFeatures = get_model_shape_reg(perm_fn, fadjs, ARCHITECTURE, keep_prob)
        

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

        myFeatures = sess.run(tf.squeeze(testFeatures), feed_dict=infer_fd)

    
        

    sess.close()
    
    invPerm = inv_perm(perm)

    # print("myFeatures shape = "+str(myFeatures.shape))
    # print("perm shape = "+str(perm.shape))
    # print("invPerm shape = "+str(invPerm.shape))
    myFeatures = myFeatures[invPerm]

    myFeatures = myFeatures[:TEMPLATE_CARD,:]

    return myFeatures



def mseLoss(prediction, gt):

    loss = tf.reduce_sum(tf.square(tf.subtract(gt,prediction)),axis=-1)
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
    fileNum=0
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
        # iuvIm = Image.open(inputPath+iuvFilename)
        # iuvMat = np.array(iuvIm,dtype=np.float32)
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

        fileNum+=1

    pathList = inputPath.split('/')
    inputFolder=pathList[-2]
    # pickle list
    with open(destPath+inputFolder+'_raw_image_input.pkl', 'wb') as fp:
        pickle.dump(inputList, fp)   
    # pickle list
    with open(destPath+inputFolder+'_gt_shape.pkl', 'wb') as fp:
        pickle.dump(gtShapeList, fp)

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
            
        rgb_vec = np.zeros((points_num,3),dtype=np.float32)
        collected_x = np.zeros(points_num)
        collected_y = np.zeros(points_num)
        collected_z = np.zeros(points_num)
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
        
    rgb_vec = np.zeros((points_num,3),dtype=np.float32)
    collected_x = np.zeros(points_num)
    collected_y = np.zeros(points_num)
    collected_z = np.zeros(points_num)
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
    dilIm = ndimage.binary_dilation(binIm, iterations=7)
    _, nr_objects = ndimage.label(dilIm > 0)

    return nr_objects


def mainFunction():

    

    if not trainMode:
        inputPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/05_01/"
        binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
        pickleFolder(inputPath, binDumpPath)
        # pickleImages(inputPath, binDumpPath)
    else:

        if RUNNING_MODE==0:     # Train network on template

            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            in_list = []
            gtshape_list = []
            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_01_standard_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_01_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'perm.pkl', 'rb') as fp:
                perm = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
                face_pos = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_02_standard_input.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_02_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            
            face_pos = np.expand_dims(face_pos,axis=0)

            # Trim valid list
            valid_in_list = valid_in_list[:5]
            valid_gtshape_list = valid_gtshape_list[:5]

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            
            trainBodyShapeNet(in_list, gtshape_list, adj_list, perm, face_pos, valid_in_list, valid_gtshape_list)

        elif RUNNING_MODE==1:      # Train Network on images alone

            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/rawImages/"
            in_list = []
            gtshape_list = []
            
            with open(binDumpPath+'01_01_raw_image_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_01_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_02_raw_image_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_02_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_raw_image_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            
            with open(binDumpPath+'01_02_raw_image_input.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_02_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            


            trainImageNet(in_list, gtshape_list, valid_in_list, valid_gtshape_list)

        elif RUNNING_MODE==2:     # Train network on template for SHAPE

            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            in_list = []
            gtshape_list = []
            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_01_standard_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_01_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_02_standard_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_02_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_04_standard_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_04_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'02_01_standard_input.pkl', 'rb') as fp:
                in_list += pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'02_01_gt_shape.pkl', 'rb') as fp:
                gtshape_list += pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'perm.pkl', 'rb') as fp:
                perm = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
                face_pos = pickle.load(fp, encoding='latin1')

            with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
                valid_in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'01_03_gt_shape.pkl', 'rb') as fp:
                valid_gtshape_list = pickle.load(fp, encoding='latin1')
            
            # Trim valid list
            valid_in_list = valid_in_list[:5]
            valid_gtshape_list = valid_gtshape_list[:5]


            # Trim input list for testing
            # in_list = in_list[:5]
            # gtshape_list = gtshape_list[:5]


            face_pos = np.expand_dims(face_pos,axis=0)

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
                gtshape_list[el] = np.expand_dims(gtshape_list[el], axis=0)
            for el in range(len(valid_in_list)):
                valid_in_list[el] = np.expand_dims(valid_in_list[el],axis=0)
                valid_gtshape_list[el] = np.expand_dims(valid_gtshape_list[el], axis=0)
            


            trainBodyShapeNet(in_list, gtshape_list, adj_list, perm, face_pos, valid_in_list, valid_gtshape_list, mode='shape')

        
        elif RUNNING_MODE==3:       # Test network
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            # with open(binDumpPath+'01_03_standard_input.pkl', 'rb') as fp:
            #     in_list = pickle.load(fp, encoding='latin1')

            with open("/morpheo-nas2/marmando/ShapeRegression/BinaryDump/wololo/"+'01_03_standard_input.pkl', 'rb') as fp:
                in_list = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'perm.pkl', 'rb') as fp:
                perm = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'face_pos.pkl', 'rb') as fp:
                face_pos = pickle.load(fp, encoding='latin1')
            with open(binDumpPath+'adj.pkl', 'rb') as fp:
                adj_list = pickle.load(fp, encoding='latin1')

            for el in range(len(in_list)):
                in_list[el] = np.expand_dims(in_list[el],axis=0)
            face_pos = np.expand_dims(face_pos,axis=0)

            
            # myIn = getInput("/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/01_03/01_03_c0003_img017.jpg", "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/01_03/01_03_c0003_img017_IUV.png")
            # myFeatures = testBodyShapeNet(myIn, adj_list, perm, face_pos)


            myFeatures = testBodyShapeNet(in_list[0], adj_list, perm, face_pos)
            
            print("myFeatures shape = "+str(myFeatures.shape))
            
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

            feature_color = feature_color[:,:3]
            # feature_color = feature_color[:,3:6]
            # feature_color = feature_color[:,6:9]
            # feature_color = feature_color[:,9:12]
            # feature_color = feature_color[:,12:15]
            # feature_color = feature_color[:,15:18]

            noisyFolder = "/morpheo-nas2/marmando/densepose/Test/"
            noisyFile = "male_template_dp.obj"
            V0,_,_, faces0, _ = load_mesh(noisyFolder, noisyFile, 0, False)
            newV, newF = getColoredMesh(V0, faces0, feature_color)
            write_mesh(newV, newF, "/morpheo-nas2/marmando/ShapeRegression/Test/"+"archi9.obj")

        elif RUNNING_MODE==4:       # Check body shape sampling
            binDumpPath = "/morpheo-nas2/marmando/ShapeRegression/BinaryDump/cleaned/"
            gtshape_list = []
            

            # # --- "loading bin dump" version ---
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

            
            # --- "Loading input" version ---
            inputPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/"
            for folder in os.listdir(inputPath):

                folderPath = inputPath + folder + "/"

                for filename in os.listdir(folderPath):

                    if filename.endswith("info.mat"):
                        infoMat = scipy.io.loadmat(folderPath+filename)
                        bodyShape = infoMat['shape'][:,0]
                        gender = infoMat['gender'][0]               # array (1,)
                        genBodyShape = np.concatenate((gender,bodyShape))

                        gtshape_list.append(genBodyShape)

                # print("List size = %g"%(len(gtshape_list)))

                # print(folderPath)

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

        elif RUNNING_MODE==5:       # Filter data

            imagePath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/train_images/run1/"
            discPath = "/morpheo-nas2/marmando/surreal/Data/SURREAL/data/cmu/discarded_train_images/run1/"
            foldName = "05_10"
            imagePath = imagePath + foldName + "/"
            discPath = discPath + foldName + "/"

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

                totalFiles+=1
            print("%g out of %g"%(triggeredFiles,totalFiles))

        elif RUNNING_MODE==6:       # Check adj dim

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



