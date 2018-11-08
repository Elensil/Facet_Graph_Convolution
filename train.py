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



def inferNet(in_points, f_normals, f_adj, edge_map, v_e_map):

    with tf.Graph().as_default():
        random_seed = 0
        np.random.seed(random_seed)

        sess = tf.InteractiveSession()
        if(FLAGS.debug):    #launches debugger at every sess.run() call
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)


        if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)

        if not os.path.exists(NETWORK_PATH):
                os.makedirs(NETWORK_PATH)


        """
        Load dataset 
        x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
        adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
                                                  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.
        """
        BATCH_SIZE=in_points.shape[0]
        #BATCH_SIZE=1
        NUM_POINTS=in_points.shape[1]
        NUM_FACES = f_normals.shape[1]
        MAX_EDGES = v_e_map.shape[2]
        NUM_EDGES = edge_map.shape[1]
        K_faces = f_adj[0].shape[2]
        NUM_IN_CHANNELS = f_normals.shape[2]

        xp_ = tf.placeholder('float32', shape=(BATCH_SIZE, NUM_POINTS,3),name='xp_')

        fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, NUM_FACES, NUM_IN_CHANNELS], name='fn_')
        #fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_FACES, K_faces], name='fadj')

        fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
        fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
        fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

        e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_EDGES,4], name='e_map_')
        ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_POINTS,MAX_EDGES], name='ve_map_')
        keep_prob = tf.placeholder(tf.float32)

        
        my_feed_dict = {xp_:in_points, fn_: f_normals, fadj0: f_adj[0], fadj1: f_adj[1], fadj2: f_adj[2],
                        e_map_: edge_map, ve_map_: v_e_map, keep_prob:1}
        
        
        batch = tf.Variable(0, trainable=False)
        fadjs = [fadj0,fadj1,fadj2]
        # --- Starting iterative process ---
        #rotTens = getRotationToAxis(fn_)
        with tf.variable_scope("model"):
            n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)

        # n_conv = normalizeTensor(n_conv)
        # n_conv = tf.expand_dims(n_conv,axis=-1)
        # n_conv = tf.matmul(tf.transpose(rotTens,[0,1,3,2]),n_conv)
        # n_conv = tf.reshape(n_conv,[BATCH_SIZE,-1,3])
        #n_conv = tf.slice(fn_,[0,0,0],[-1,-1,3])+n_conv

        n_conv = normalizeTensor(n_conv)


        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(NETWORK_PATH))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        
        refined_x = update_position2(xp_, n_conv, e_map_, ve_map_)

        points = tf.squeeze(refined_x)
        # points shape should now be [NUM_POINTS, 3]

        my_feed_dict[keep_prob]=1
        
        outPoints, outN = sess.run([points,tf.squeeze(n_conv)],feed_dict=my_feed_dict)
        sess.close()
    return outPoints, outN


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
    #fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj')

    fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
    fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
    fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

    tfn_ = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='tfn_')

    # Validation data
    valid_fn = tf.placeholder('float32', shape=(BATCH_SIZE, None,NUM_IN_CHANNELS),name='valid_fn')
    valid_tn = tf.placeholder('float32', shape=(BATCH_SIZE, None,3),name='valid_tn')
    valid_fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='valid_fadj')

    sample_ind = tf.placeholder(tf.int32, shape=[10000], name='sample_ind')

    keep_prob = tf.placeholder(tf.float32)
    
    
    batch = tf.Variable(0, trainable=False)

    # --- Starting iterative process ---


    #rotTens = getRotationToAxis(fn_)

    fadjs = [fadj0,fadj1,fadj2]

    with tf.variable_scope("model"):
        # n_conv = get_model_reg(fn_, fadj0, ARCHITECTURE, keep_prob)
        n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)


    # n_conv = normalizeTensor(n_conv)
    # n_conv = tf.expand_dims(n_conv,axis=-1)
    # n_conv = tf.matmul(tf.transpose(rotTens,[0,1,3,2]),n_conv)
    # n_conv = tf.reshape(n_conv,[BATCH_SIZE,-1,3])
    # n_conv = tf.slice(fn_,[0,0,0],[-1,-1,3])+n_conv
    n_conv = normalizeTensor(n_conv)

    
    with tf.device(DEVICE):
        customLoss = faceNormalsLoss(n_conv, tfn_)
        # customLoss2 = faceNormalsLoss(n_conv2, tfn_)
        # customLoss3 = faceNormalsLoss(n_conv3, tfn_)
        train_step = tf.train.AdamOptimizer().minimize(customLoss, global_step=batch)
        # train_step2 = tf.train.AdamOptimizer().minimize(customLoss2, global_step=batch)
        # train_step3 = tf.train.AdamOptimizer().minimize(customLoss3, global_step=batch)

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

    with tf.device(DEVICE):
        lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
        last_loss = 0
        for iter in range(NUM_ITERATIONS):

            # Get random sample from training dictionary
            batch_num = random.randint(0,len(f_normals_list)-1)
            num_p = f_normals_list[batch_num].shape[1]
            random_ind = np.random.randint(num_p,size=10000)

            # train_fd = {fn_: f_normals_list[batch_num], fadj: f_adj_list[batch_num], tfn_: GTfn_list[batch_num],
            #               sample_ind: random_ind, keep_prob:1}

            # train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], tfn_: GTfn_list[batch_num],
            #               sample_ind: random_ind, keep_prob:1}

            train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0], fadj1: f_adj_list[batch_num][1],
                            fadj2: f_adj_list[batch_num][2], tfn_: GTfn_list[batch_num],
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

                    valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                            fadj2: valid_f_adj_list[vbm][2], tfn_: valid_GTfn_list[vbm],
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

            if (iter%2000 == 0):
                saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+iter)
    
    saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep+NUM_ITERATIONS)

    sess.close()
    csv_filename = RESULTS_PATH+NET_NAME+".csv"
    f = open(csv_filename,'ab')
    np.savetxt(f,lossArray, delimiter=",")


def faceNormalsLoss(fn,gt_fn):

    #version 1
    n_dt = tensorDotProduct(fn,gt_fn)
    #loss = tf.acos(n_dt-1e-5)    # So that it stays differentiable close to 1
    close_to_one = 0.999999999
    loss = tf.acos(tf.maximum(tf.minimum(n_dt,close_to_one),-close_to_one))    # So that it stays differentiable close to 1 and -1
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



# Original update algorithm from Taubin (Linear anisotropic mesh filtering)
# Copied from function above, which was my own adaptation of Taubin's algorithm with vertices normals 
def update_position2(x, face_normals, edge_map, v_edges, iter_num=20):

    lmbd = 1/18

    batch_size, num_points, space_dims = x.get_shape().as_list()
    max_edges = 50
    _, num_edges, _ = edge_map.get_shape().as_list()

    testPoint = 3145
    testEdgeNum = 49
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

        n_edges_vec = tf.reshape(n_edges_vec, [batch_size, num_points, max_edges, 4, 3])


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




def normalizeTensor(x):
    with tf.variable_scope("normalization"):
        #norm = tf.norm(x,axis=-1)
        epsilon = tf.constant(1e-5,name="epsilon")
        square = tf.square(x,name="square")
        square_sum = tf.reduce_sum(square,axis=-1,name="square_sum")
        norm = tf.sqrt(epsilon+square_sum,name="sqrt")

        norm_non_zeros = tf.greater(norm,epsilon)
        inv_norm = tf.where(norm_non_zeros,tf.reciprocal(norm+epsilon,name="norm_division"),tf.zeros_like(norm,name="zeros"))
        newX = tf.multiply(x, tf.expand_dims(inv_norm,axis=-1),name="result")
    return newX



def mainFunction():


    pickleLoad = True
    pickleSave = True

    K_faces = 25

    maxSize = 30000
    patchSize = 25000

    training_meshes_num = [0]
    valid_meshes_num = [0]

    #binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/smallAdj/"
    binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/bigAdj/"
    binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/coarsening4/"

    #binDumpPath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/BinaryDump/kinect_v1/coarsening4/"

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
        # --- Load mesh ---
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        # print("WARNING!!!!! Hardcoded a change in faces adjacency")
        # f_adj0, _, _ = getFacesAdj2(faces0)
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0)


        f_pos0 = np.reshape(f_pos0,(-1,3))
        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)
        # f_area0 = getTrianglesArea(V0,faces0)
        # f_area0 = np.reshape(f_area0, (-1,1))
        # f_normals0 = np.concatenate((f_normals0, f_area0), axis=1)

        # Load GT
        GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)
        GTf_normals0 = computeFacesNormals(GT0, faces0)

        # Get patches if mesh is too big
        facesNum = faces0.shape[0]
        if facesNum>maxSize:
            patchNum = int(facesNum/patchSize)+1
            for p in range(patchNum):
                faceSeed = np.random.randint(facesNum)
                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

                GTPatchV = GT0[vOldInd]
                #patchFNormals = f_normals0[fOldInd]
                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = GTf_normals0[fOldInd]

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                adjs, newToOld = coarsen(coo_adj,4)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                old_N = patchFNormals.shape[0]
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                print("padding6 shape: "+str(padding6.shape))
                print("patchFNormals shape: "+str(patchFNormals.shape))
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                patchGTFNormals = patchGTFNormals[newToOld]

                # Change adj format
                fAdjs = []
                for lvl in range(3):
                    fadj = sparseToList(adjs[2*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)
                        # fAdjs = []
                        # f_adj = np.expand_dims(testPatchAdj, axis=0)
                        # fAdjs.append(f_adj)

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                #f_adj = np.expand_dims(testPatchAdj, axis=0)
                GTf_normals = np.expand_dims(patchGTFNormals, axis=0)

                in_list.append(f_normals)
                adj_list.append(fAdjs)
                gt_list.append(GTf_normals)

                print("Added training patch: mesh " + filename + ", patch " + str(p) + " (" + str(mesh_count_list[0]) + ")")
                mesh_count_list[0]+=1
        else:       #Small mesh case



            # print("f_adj: \n"+str(f_adj0))
            # print("faces0: \n"+str(faces0))

            # print("normals: \n"+str(f_normals0))
            # print("GT normals: \n"+str(GTf_normals0))
            # print("V0: \n"+str(V0))
            # print("GT0: \n"+str(GT0))

            # Convert to sparse matrix and coarsen graph
            coo_adj = listToSparse(f_adj0, f_pos0)
            adjs, newToOld = coarsen(coo_adj,4)

            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            padding6 =np.zeros((new_N-old_N,6))
            padding3 =np.zeros((new_N-old_N,3))
            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            GTf_normals0 = np.concatenate((GTf_normals0, padding3),axis=0)
            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            GTf_normals0 = GTf_normals0[newToOld]

            # Change adj format
            fAdjs = []
            for lvl in range(3):
                fadj = sparseToList(adjs[2*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)

                        # fAdjs = []
                        # f_adj = np.expand_dims(f_adj0, axis=0)
                        # fAdjs.append(f_adj)

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            #f_adj = np.expand_dims(f_adj0, axis=0)
            GTf_normals = np.expand_dims(GTf_normals0, axis=0)


            # print("f_adj: \n"+str(fAdjs[0]))
            # #print("faces0: \n"+str(faces0[0:5,:]))

            # print("normals: \n"+str(f_normals_pos[:,:3]))
            # print("GT normals: \n"+str(GTf_normals0))

            # print("perm: \n"+str(newToOld))


            in_list.append(f_normals)
            adj_list.append(fAdjs)
            gt_list.append(GTf_normals)
        
            print("Added training mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

            mesh_count_list[0]+=1

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

        inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/noisy/"
        validFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/valid/"
        gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/train/original/"
        

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

                    print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")
                    gtfilename = filename[:-gtnameoffset]+".obj"
                    addMesh(inputFilePath, filename, gtFilePath, gtfilename, f_normals_list, GTfn_list, f_adj_list, training_meshes_num)

            # Validation set
            for filename in os.listdir(validFilePath):
                if (filename.endswith(".obj")):
                    gtfilename = filename[:-gtnameoffset]+".obj"
                    addMesh(validFilePath, filename, gtFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_meshes_num)
                    

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

    # Inference: Denoise set, save meshes (colored with heatmap), compute metrics
    elif running_mode == 2:
        
        # Take the opportunity to generate array of metrics on reconstructions
        nameArray = []      # String array, to now which row is what
        resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                            # [Max distance, Mean distance, Mean angle, std angle, face num]

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results.csv"
        
        empiricMax = 20.0

        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("Merlion")) or (gtFileName.startswith("armadillo")) or (gtFileName.startswith("agargoyle")) or \
            (gtFileName.startswith("dragon")):
                continue


            # Get all 3 noisy meshes
            # noisyFile0 = gtFileName[:-4]+"_noisy_1.obj"
            # noisyFile1 = gtFileName[:-4]+"_noisy_2.obj"
            # noisyFile2 = gtFileName[:-4]+"_noisy_3.obj"
            noisyFile0 = gtFileName[:-4]+"_n1.obj"
            noisyFile1 = gtFileName[:-4]+"_n2.obj"
            noisyFile2 = gtFileName[:-4]+"_n3.obj"

            denoizedFile0 = gtFileName[:-4]+"_denoized_1.obj"
            denoizedFile1 = gtFileName[:-4]+"_denoized_2.obj"
            denoizedFile2 = gtFileName[:-4]+"_denoized_3.obj"

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
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            #edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            

            noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                
                if not os.path.isfile(RESULTS_PATH+denoizedFile):
                    
                    V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)
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

                    print("running n"+str(fileNum+1)+"...")
                    upV0, upN0 = inferNet(V0, f_normals_pos0, fAdjs, edge_map0, v_e_map)
                    print("computing Hausdorff "+str(fileNum+1)+"...")
                    haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                    angDistVec = angularDiffVec(upN0, GTf_normals0)
                    angDist0, angStd0 = angularDiff(upN0, GTf_normals0)
                    print("max angle: "+str(np.amax(angDistVec)))

                    angDistVec = angDistVec[oldToNew]
                    # --- Test heatmap ---
                    angColor = angDistVec / empiricMax
                    # print("max color: "+str(np.amax(angColor)))
                    # print("min color: "+str(np.amin(angColor)))
                    # print("mean color: "+str(np.mean(angColor)))
                    angColor = 1 - angColor
                    angColor = np.maximum(angColor, np.zeros_like(angColor))
                    # print("max color: "+str(np.amax(angColor)))
                    # print("min color: "+str(np.amin(angColor)))
                    # print("mean color: "+str(np.mean(angColor)))
                    colormap = getHeatMapColor(1-angColor)
                    print("colormap shape: "+str(colormap.shape))
                    newV, newF = getColoredMesh(upV0, faces_gt, colormap)
                    #newV, newF = getHeatMapMesh(upV0, faces_gt, angColor)

                    write_mesh(newV, newF, RESULTS_PATH+denoizedFile)
                    #write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile0)

                    # Fill arrays
                    nameArray.append(denoizedFile)
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

    elif running_mode == 3:


        inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/"
        
        
        inputFileName = "bunny_iH.obj"
        #inputFileName = "armadillo.obj"
        
        V0,_,_, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)

        f_normals0 = computeFacesNormals(V0, faces0)
        _, edge_map0, v_e_map0 = getFacesAdj2(faces0)
        f_adj0 = getFacesLargeAdj(faces0,K_faces)

        print("barycenters: ")
        f_pos0 = getTrianglesBarycenter(V0, faces0)
        

        print("compute sparse matrix")
        coo_adj = listToSparse(f_adj0, f_pos0)
        print("coarsen graph")
        adjs, newToOld = coarsen(coo_adj,3)
        print("done")
        newAdj0 = sparseToList(adjs[0], K_faces)
        newAdj1 = sparseToList(adjs[1], K_faces)
        print("alright")
        print("sparse row: "+str(adjs[0][0,:]))
        print("list row: "+str(newAdj0[0,:]))
        print("sparse row: "+str(adjs[1][0,:]))
        print("list row: "+str(newAdj1[0,:]))

        old_N = faces0.shape[0]
        new_N = len(newToOld)
        padding3 = np.zeros((new_N-old_N,3))

        fpos0 = np.concatenate((f_pos0,padding3),axis=0)

        fpos0 = fpos0[newToOld]

        # Change pos of fake nodes
        for f in range(new_N):
            if np.array_equal(fpos0[f,:],np.array([0,0,0])):
                if f%2==0:
                    fpos0[f,:] = fpos0[f+1,:]
                else:
                    fpos0[f,:] = fpos0[f-1,:]


        ftest0 = np.arange(new_N)
        ftest0 = np.reshape(ftest0, (int(new_N/2),2))

        #lvl 0
        write_mesh(fpos0, ftest0, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/edges0.obj")
        write_xyz(fpos0, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/fpos0.xyz")

        fpos1 = np.reshape(fpos0,(-1,2,3))
        fpos1 = np.mean(fpos1,axis=1)

        ftest1 = ftest0[:int(new_N/4),:]


        for f in range(int(new_N/2)):
            if np.array_equal(fpos1[f,:],np.array([0,0,0])):
                if f%2==0:
                    fpos1[f,:] = fpos1[f+1,:]
                else:
                    fpos1[f,:] = fpos1[f-1,:]

        #lvl 1
        write_mesh(fpos1, ftest1, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/edges1.obj")
        write_xyz(fpos1, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/fpos1.xyz")


        fpos2 = np.reshape(fpos1,(-1,2,3))
        fpos2 = np.mean(fpos2,axis=1)

        ftest2 = ftest0[:int(new_N/8),:]


        for f in range(int(new_N/4)):
            if np.array_equal(fpos2[f,:],np.array([0,0,0])):
                if f%2==0:
                    fpos2[f,:] = fpos2[f+1,:]
                else:
                    fpos2[f,:] = fpos2[f-1,:]

        #lvl 1
        write_mesh(fpos2, ftest2, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/edges2.obj")
        write_xyz(fpos2, "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/fpos2.xyz")

        # faceSeed = np.random.randint(faces0.shape[0])

        # for i in range(10):
        #   faceSeed = np.random.randint(faces0.shape[0])
        #   testPatchV, testPatchF, testPatchAdj = getMeshPatch(V0, faces0, f_adj0, 10000, faceSeed)

        # print("testPatchV = "+str(testPatchV))
        # print("testPatchF = "+str(testPatchF))
        # print("testPatchAdj = "+str(testPatchAdj))

    elif running_mode == 4:

        gtnameoffset = 7
        f_normals_list = []
        f_adj_list = []
        GTfn_list = []

        valid_f_normals_list = []
        valid_f_adj_list = []
        valid_GTfn_list = []

        inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/"

        gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/TestFolder/"
        
        # Training set
        for filename in os.listdir(inputFilePath):
            #print("training_meshes_num start_iter " + str(training_meshes_num))
            if training_meshes_num[0]>10:
                break
            #if (filename.endswith("noisy.obj")and not(filename.startswith("raptor_f"))and not(filename.startswith("olivier"))and not(filename.startswith("red_box"))and not(filename.startswith("bunny"))):
            #if (filename.endswith(".obj") and not(filename.startswith("buste"))):
            if (filename.endswith(".obj") and (filename.startswith("cubetest"))):
                gtfilename = filename
                print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")
                addMesh(inputFilePath, filename, gtFilePath, gtfilename, f_normals_list, GTfn_list, f_adj_list, training_meshes_num)

        # # Validation set
        # for filename in os.listdir(validFilePath):
        #   if (filename.endswith(".obj")):
        #       gtfilename = filename[:-gtnameoffset]+".obj"
        #       addMesh(validFilePath, filename, gtFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_meshes_num)



        # trainNet(f_normals_list, GTfn_list, f_adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list)

            # write_mesh(testPatchV, testPatchF, "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/testPatch"+str(i)+".obj")

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

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results_heat.csv"
        

        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("Merlion")) or (gtFileName.startswith("aarmadillo")) or (gtFileName.startswith("agargoyle")) or \
            (gtFileName.startswith("dragon")):
                continue

            denoizedFile0 = gtFileName[:-4]+"_n1-dtree3.obj"
            denoizedFile1 = gtFileName[:-4]+"_n2-dtree3.obj"
            denoizedFile2 = gtFileName[:-4]+"_n3-dtree3.obj"

            # denoizedFile0 = gtFileName[:-4]+"_denoized_1.obj"
            # denoizedFile1 = gtFileName[:-4]+"_denoized_2.obj"
            # denoizedFile2 = gtFileName[:-4]+"_denoized_3.obj"

            heatFile0 = gtFileName[:-4]+"_dtree3_heatmap_1.obj"
            heatFile1 = gtFileName[:-4]+"_dtree3_heatmap_2.obj"
            heatFile2 = gtFileName[:-4]+"_dtree3_heatmap_3.obj"

            # heatFile0 = gtFileName[:-4]+"_heatmap_1.obj"
            # heatFile1 = gtFileName[:-4]+"_heatmap_2.obj"
            # heatFile2 = gtFileName[:-4]+"_heatmap_3.obj"

            if (os.path.isfile(RESULTS_PATH+heatFile0)) and (os.path.isfile(RESULTS_PATH+heatFile1)) and (os.path.isfile(RESULTS_PATH+heatFile2)):
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

            empiricMax = 20.0

            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]
            heatMapFilesList = [heatFile0,heatFile1,heatFile2]

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
                    # print("max color: "+str(np.amax(angColor)))
                    # print("min color: "+str(np.amin(angColor)))
                    # print("mean color: "+str(np.mean(angColor)))
                    angColor = 1 - angColor
                    angColor = np.maximum(angColor, np.zeros_like(angColor))
                    # print("max color: "+str(np.amax(angColor)))
                    # print("min color: "+str(np.amin(angColor)))
                    # print("mean color: "+str(np.mean(angColor)))
                    newV, newF = getHeatMapMesh(V0, faces_gt, angColor)

                    write_mesh(newV, newF, RESULTS_PATH+heatFile)
                    #write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile0)

                    # Fill arrays
                    nameArray.append(denoizedFile)
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

    # Color mesh by estimated normals
    elif running_mode == 9:
        
        # Take the opportunity to generate array of metrics on reconstructions
        nameArray = []      # String array, to now which row is what
        resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                            # [Max distance, Mean distance, Mean angle, std angle, face num]

        noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/noisy/"
        gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Kinect_v1/test/original/"

        # noisyFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/noisy/"
        # gtFolder = "/morpheo-nas/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/train/original/"

        # results file name
        csv_filename = RESULTS_PATH+"results.csv"
        
        empiricMax = 20.0

        # Get GT mesh
        for gtFileName in os.listdir(gtFolder):

            nameArray = []
            resultsArray = []
            if (not gtFileName.endswith(".obj")) or (gtFileName.startswith("Merlion")) or (gtFileName.startswith("armadillo")) or (gtFileName.startswith("agargoyle")) or \
            (gtFileName.startswith("dragon")):
                continue


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

            denoizedFile0 = gtFileName[:-4]+"_denoized_1.obj"
            denoizedFile1 = gtFileName[:-4]+"_denoized_2.obj"
            denoizedFile2 = gtFileName[:-4]+"_denoized_3.obj"

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
            # print("WARNING!!!!! Hardcoded a change in faces adjacency")
            # f_adj, edge_map, v_e_map = getFacesAdj2(faces_gt)
            

            faces_gt = np.array(faces_gt).astype(np.int32)
            faces = np.expand_dims(faces_gt,axis=0)
            #faces = np.array(faces).astype(np.int32)
            #f_adj = np.expand_dims(f_adj, axis=0)
            #edge_map = np.expand_dims(edge_map, axis=0)
            v_e_map = np.expand_dims(v_e_map, axis=0)

            

            noisyFilesList = [noisyFile0,noisyFile1,noisyFile2]
            noisyFilesWColorList = [noisyFileWColor0,noisyFileWColor1,noisyFileWColor2]
            denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]

            for fileNum in range(len(denoizedFilesList)):
                
                denoizedFile = denoizedFilesList[fileNum]
                noisyFile = noisyFilesList[fileNum]
                noisyFileWColor = noisyFilesWColorList[fileNum]
                
                if not os.path.isfile(RESULTS_PATH+denoizedFile):
                    
                    V0,_,_, _, _ = load_mesh(noisyFolder, noisyFile, 0, False)
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

                    print("running n"+str(fileNum+1)+"...")
                    upV0, upN0 = inferNet(V0, f_normals_pos0, fAdjs, edge_map0, v_e_map)
                    print("computing Hausdorff "+str(fileNum+1)+"...")
                    haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                    angDistVec = angularDiffVec(upN0, GTf_normals0)
                    angDist0, angStd0 = angularDiff(upN0, GTf_normals0)
                    print("max angle: "+str(np.amax(angDistVec)))

                    angColor = upN0[oldToNew]
                    
                    angColor = (angColor+1)/2

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=int, default=0)
    #parser.add_argument('--dataset_path')
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device', type=str, default='/cpu:0')
    parser.add_argument('--net_name', type=str, default='unnamed_net')
    parser.add_argument('--running_mode', type=int, default=0)
    

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
    #NUM_CLASSES = FLAGS.num_classes

    mainFunction()


