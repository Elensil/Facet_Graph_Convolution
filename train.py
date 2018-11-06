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

def inferNet(in_points, f_normals, f_adj, edge_map, v_e_map,images_lists,calibs_lists,num_fake_nodes,old_to_new_permutations,num_faces):

	with tf.Graph().as_default():
		random_seed = 0
		np.random.seed(random_seed)

		sess = tf.InteractiveSession()
		if(FLAGS.debug):	#launches debugger at every sess.run() call
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
		BATCH_SIZE=in_points.shape[0]
		#BATCH_SIZE=1
		NUM_POINTS=in_points.shape[1]
		NUM_FACES = f_normals.shape[1]
		MAX_EDGES = v_e_map.shape[2]
		NUM_EDGES = edge_map.shape[1]
                K_faces = f_adj[0].shape[2]
		NUM_IN_CHANNELS = f_normals.shape[2]


                NUM_CAMS = np.shape(images_lists)[1]
                IMG_WIDTH = np.shape(images_lists)[2]
                IMG_HEIGHT = np.shape(images_lists)[3]
                IMG_CHANNELS = np.shape(images_lists)[4]

		xp_ = tf.placeholder('float32', shape=(BATCH_SIZE, NUM_POINTS,3),name='xp_')

		fn_ = tf.placeholder('float32', shape=[BATCH_SIZE, NUM_FACES, NUM_IN_CHANNELS], name='fn_')
		#fadj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_FACES, K_faces], name='fadj')

		fadj0 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj0')
		fadj1 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj1')
		fadj2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None, K_faces], name='fadj2')

		e_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_EDGES,4], name='e_map_')
		ve_map_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_POINTS,MAX_EDGES], name='ve_map_')
		keep_prob = tf.placeholder(tf.float32)


                images_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_CAMS,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS])
                calibs_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_CAMS,3,4])

                input_images = images_lists
                input_calibs = calibs_lists
		
		batch = tf.Variable(0, trainable=False)
		fadjs = [fadj0,fadj1,fadj2]
		# --- Starting iterative process ---
		#rotTens = getRotationToAxis(fn_)
                #fn_normal_only = tf.slice(fn_,[0,0,0],[-1,-1,3])
		with tf.variable_scope("model"):
                        #n_conv = get_appearance_model_reg(fn_, fadj, ARCHITECTURE, keep_prob,images_,calibs_)
                        n_conv = get_model_reg_multi_scale_appearance(fn_, fadjs, ARCHITECTURE, keep_prob, images_, calibs_)
			#n_conv = get_model_reg(fn_, fadj, ARCHITECTURE, keep_prob)
			#n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
			#n_conv = get_model_reg_multi_scale(fn_normal_only, fadjs, ARCHITECTURE, keep_prob)


		n_conv = normalizeTensor(n_conv)

		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		# points shape should now be [NUM_POINTS, 3]
                predicted_normals = []
                for i in range(num_patches):
                    my_feed_dict = {fn_: f_normals[i], fadj0: f_adj[i][0], fadj1: f_adj[i][1], fadj2: f_adj[i][2],
                                    keep_prob:1.0, images_:input_images, calibs_:input_calibs}
                    outN = sess.run(n_conv,feed_dict=my_feed_dict)
                    predicted_normals.append(outN)
                    print("outN shape: ",outN.shape)
                    exit()

                #Now gather patches predictions and update vertices position

                new_normals = tf.placeholder('float32', shape=[BATCH_SIZE, None, 3], name='fn_')
                #refined_x = update_position(xp_,fadj, n_conv)
                refined_x = update_position2(xp_, new_normals, e_map_, ve_map_)
                points = tf.squeeze(refined_x)

                update_feed_dict = {xp_:in_points, new_normals: outN, e_map_: edge_map, ve_map_: v_e_map}
                outPoints = sess.run(points,feed_dict=update_feed_dict)
		sess.close()

        return outPoints, predicted_normals

def trainNet(f_normals_list, GTfn_list, f_adj_list, images_lists, calibs_lists, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list,valid_images_lists,valid_calibs_lists):
	
	random_seed = 0
	np.random.seed(random_seed)

	sess = tf.InteractiveSession()
	if(FLAGS.debug):	#launches debugger at every sess.run() call
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

        NUM_CAMS = np.shape(images_lists)[1]
        IMG_WIDTH = np.shape(images_lists)[2]
        IMG_HEIGHT = np.shape(images_lists)[3]
        IMG_CHANNELS = np.shape(images_lists)[4]

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

	#faces_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_FACES,3], name='faces_')
	keep_prob = tf.placeholder(tf.float32)

        images_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_CAMS,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS])
        calibs_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_CAMS,3,4])
	
	batch = tf.Variable(0, trainable=False)

	# --- Starting iterative process ---

	#rotTens = getRotationToAxis(fn_)

	fadjs = [fadj0,fadj1,fadj2]

	fn_normal_only = tf.slice(fn_,[0,0,0],[-1,-1,3])
	with tf.variable_scope("model"):

                #n_conv = get_model_reg(fn_, fadj, ARCHITECTURE, keep_prob)
                #n_conv = get_appearance_model_reg(fn_, fadj, ARCHITECTURE, keep_prob,images_,calibs_)
                n_conv = get_model_reg_multi_scale_appearance(fn_, fadjs, ARCHITECTURE, keep_prob, images_, calibs_)
		# n_conv = get_model_reg(fn_, fadj0, ARCHITECTURE, keep_prob)
		#n_conv = get_model_reg_multi_scale(fn_, fadjs, ARCHITECTURE, keep_prob)
		#n_conv = get_model_reg_multi_scale(fn_normal_only, fadjs, ARCHITECTURE, keep_prob)



	# n_conv = normalizeTensor(n_conv)
	# n_conv = tf.expand_dims(n_conv,axis=-1)
	# n_conv = tf.matmul(tf.transpose(rotTens,[0,1,3,2]),n_conv)
	# n_conv = tf.reshape(n_conv,[BATCH_SIZE,-1,3])
	# n_conv = tf.slice(fn_,[0,0,0],[-1,-1,3])+n_conv
	n_conv = normalizeTensor(n_conv)

	
	with tf.device(DEVICE):
                with tf.variable_scope("loss"):
                    #validLoss = faceNormalsLoss(n_conv, tfn_)
                    #validLoss = faceNormalsSampledLoss(n_conv, tfn_, sample_ind)
                    #customLoss = faceNormalsSampledLoss(n_conv, tfn_, sample_ind)
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
	# 				"assignment_3", "assignment_4", "assignment_5", "assignment_6", "assignment_7", "assignment_8"]:
	# 	print("opname = "+ opname)
	# 	varname = "model/"+opname
	# 	myvar = [var for var in tf.global_variables() if var.op.name==varname][0]
	# 	dictVar[varname] = myvar
	# 	listVar.append(myvar)

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
                saver.restore(sess, ckpt.model_checkpoint_path)
                #Extract from checkpoint filename
                globalStep = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
		#write_logs("Checkpoint restored\n")
                # saver = tf.train.import_meta_graph(RESULTS_PATH)+'/test_fc6-400.meta')
                # saver.restore(sess,tf.train.latest_checkpoint('./'))

	# Training

	train_loss=0
	train_samp=0

	with tf.device(DEVICE):
		lossArray = np.zeros([int(NUM_ITERATIONS/10),2])
		last_loss = 0
                first=True
		for iter in range(NUM_ITERATIONS):

			# Get random sample from training dictionary
			batch_num = random.randint(0,len(f_normals_list)-1)
			num_p = f_normals_list[batch_num].shape[1]
			random_ind = np.random.randint(num_p,size=10000)

                        input_images = [images_lists[batch_num]]
                        input_calibs = [calibs_lists[batch_num]]
                        #train_fd = {fn_: f_normals_list[batch_num], fadj: f_adj_list[batch_num], tfn_: GTfn_list[batch_num],
                        #                                sample_ind: random_ind, keep_prob:1, images_:input_images, calibs_:input_calibs}

                        train_fd = {fn_: f_normals_list[batch_num], fadj0: f_adj_list[batch_num][0],
                            fadj1: f_adj_list[batch_num][1],fadj2: f_adj_list[batch_num][2],
                            tfn_: GTfn_list[batch_num],sample_ind: random_ind, keep_prob:1,
                            images_:input_images, calibs_:input_calibs}


			#sess.run(customLoss,feed_dict=train_fd)
                        local_train_loss = sess.run(customLoss,feed_dict=train_fd)
                        #print("Local Training Loss: "+str(local_train_loss))
                        train_loss += local_train_loss
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
                                train_loss = 0
                                train_samp = 0

			# Compute validation loss
                        if (iter%20 == 0):
				valid_loss = 0
				valid_samp = len(valid_f_normals_list)
				valid_random_ind = np.random.randint(num_p,size=10000)
				for vbm in range(valid_samp):

                                    input_images = [valid_images_lists[vbm]]
                                    input_calibs = [valid_calibs_lists[vbm]]
                                    #valid_fd = {fn_: valid_f_normals_list[vbm], fadj: valid_f_adj_list[vbm], tfn_: valid_GTfn_list[vbm],
                                    #                sample_ind: valid_random_ind, keep_prob:1, images_:input_images, calibs_:input_calibs}
                                    valid_fd = {fn_: valid_f_normals_list[vbm], fadj0: valid_f_adj_list[vbm][0], fadj1: valid_f_adj_list[vbm][1],
                                                fadj2: valid_f_adj_list[vbm][2], tfn_: valid_GTfn_list[vbm],
                                                sample_ind: valid_random_ind, keep_prob:1, images_:input_images, calibs_:input_calibs}
                                    local_valid_loss = sess.run(customLoss,feed_dict=valid_fd)
                                    #print("Local Valid Loss: "+str(local_valid_loss))
                                    if local_valid_loss != local_valid_loss:
                                        print("Found NaN Loss!")
                                        print(valid_GTfn_list[vbm][0][0:50][:])
                                    valid_loss += local_valid_loss


                                valid_loss/=valid_samp
				print("Iteration %d, validation loss %g"%(iter, valid_loss))
				lossArray[int(iter/10),1]=valid_loss
				if iter>0:
                                    if first:
                                        lossArray[int(iter/10)-1,1] = valid_loss
                                        first=False
                                    else:
					lossArray[int(iter/10)-1,1] = (valid_loss+last_loss)/2
                                    last_loss=valid_loss

			sess.run(train_step,feed_dict=train_fd)
			# sess.run(train_step2,feed_dict=my_feed_dict)
			# sess.run(train_step3,feed_dict=my_feed_dict)

	#saver.save(sess, RESULTS_PATH+"archi13_rot_inv",global_step=4000)
	#saver.save(sess, RESULTS_PATH+"archi15_patches",global_step=100000)
	saver.save(sess, RESULTS_PATH+NET_NAME,global_step=globalStep)
	sess.close()

        csv_filename = RESULTS_PATH+'/'+NET_NAME+"_training.csv"
	f = open(csv_filename,'ab')
	np.savetxt(f,lossArray, delimiter=",")


def checkNan(my_feed_dict):
	sess = tf.get_default_session()
	nan0 = sess.run('nan0:0',feed_dict=my_feed_dict)
	# nan1 = sess.run('nan1:0',feed_dict=my_feed_dict)
	# nan2 = sess.run('nan2:0',feed_dict=my_feed_dict)
	# nan3 = sess.run('nan3:0',feed_dict=my_feed_dict)
	# nan4 = sess.run('nan4:0',feed_dict=my_feed_dict)
	# nan5 = sess.run('nan5:0',feed_dict=my_feed_dict)
	# nan6 = sess.run('nan6:0',feed_dict=my_feed_dict)

	print("nan0: " + str(np.any(nan0)))
	# print("nan1: " + str(np.any(nan0)))
	# print("nan2: " + str(np.any(nan0)))
	# print("nan3: " + str(np.any(nan0)))
	# print("nan4: " + str(np.any(nan0)))
	# print("nan5: " + str(np.any(nan0)))
	# print("nan6: " + str(np.any(nan0)))


def faceNormalsLoss(fn,gt_fn):
	#version 1
	n_dt = tensorDotProduct(fn,gt_fn)
	#loss = tf.acos(n_dt-1e-5)	# So that it stays differentiable close to 1
        close_to_one = 0.999999999
        loss = tf.acos(tf.maximum(tf.minimum(n_dt,close_to_one),-close_to_one))	# So that it stays differentiable close to 1 and -1
        print("GTFN shape: ",gt_fn.get_shape().as_list())
        gtfn_abs_sum = tf.reduce_sum(tf.abs(gt_fn),axis=2)
        print("Summed GTFN shape: ",gtfn_abs_sum.get_shape().as_list())
        fakenodes = tf.less_equal(gtfn_abs_sum,10e-4,name='FakeNodesMask')
        #fakenodes = tf.reduce_all(fakenodes,axis=-1)
        print("fakenodes shape: ",fakenodes.get_shape().as_list())
        print("Loss shape: ",loss.get_shape().as_list())
        #exit()

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

def faceNormalsSampledLoss(fn,gt_fn,indices):
	#version 1
	n_dt = tensorDotProduct(fn,gt_fn)
	# shape = [batch, N, 3] ?
	
	# loss = 1 - n_dt
	#loss = tf.acos(n_dt-1e-5) 	# So that it stays differentiable close to 1
	loss = tf.acos(0.9999*n_dt)	# So that it stays differentiable close to 1 and -1
	loss = 180*loss/math.pi
	print("loss shape: "+str(loss.shape))
	#indices = tf.reshape(indices,[1000])
	loss = tf.squeeze(loss)
	print("loss shape: "+str(loss.shape))
	print("indices shape: "+str(indices.shape))
	sampled_loss = tf.gather(loss, indices)
	sampled_loss = tf.reduce_mean(sampled_loss)
	return sampled_loss


def is_almost_equal(x,y,threshold):
	if np.sum((x-y)**2)<threshold**2:
		return True
	else:
		return False


def unique_columns2(data):
	dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
	dataf = np.asfortranarray(data).view(dt)
	u,uind = np.unique(dataf, return_inverse=True)
	u = u.view(data.dtype).reshape(-1,data.shape[0]).T
	return (u,uind)

def update_position(x, adj, n,iter_num=15):
	
	lmbd = 1/18
	_,_,K = adj.get_shape().as_list()
	#K = adj.shape[2]

	for it in range(iter_num):
		
		#Building neighbourhood
		neigh = get_slices(x,adj)
		#shape = (batch, points, neighbourhood (K), space_dims)
		# (one col per neighbour + center point (in 1st column))

		#computing vectors
		edges = tf.subtract(neigh[:,:,1:,:],tf.expand_dims(neigh[:,:,0,:],2))
		#shape = (batch, points, neighbourhood (K-1), space_dims)

		#WARNING!!! set empty neighbours to zero! (otherwise, vectors pointing to origin)
		non_zeros = tf.not_equal(adj[:,:,1:], 0)
		non_zeros = tf.tile(tf.expand_dims(non_zeros,axis=-1),[1,1,1,3])
		edges = tf.where(non_zeros, edges,tf.zeros_like(edges))

		#neighbourhood normals
		neigh_normals = get_slices(n, adj)
		#Compute edge normal as the average between the two vertices normals
		#edge_normals = tf.add(neigh_normals[:,:,1:,:],tf.expand_dims(neigh_normals[:,:,0,:],2))
		
		edge_normals = tf.tile(tf.expand_dims(neigh_normals[:,:,0,:],2),[1,1,K-1,1])

		edge_normals = normalizeTensor(edge_normals)

		# n . (xj-xi)
		dot_p = tensorDotProduct(edge_normals,edges)

		# Tile to multiply
		dot_p = tf.tile(tf.expand_dims(dot_p,axis=-1),[1,1,1,3])

		w_norm = tf.multiply(edge_normals,dot_p)

		x_update = lmbd * tf.reduce_sum(w_norm,axis=2)

		x = tf.add(x,x_update)
	return x

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

        v_edges=v_edges+1 	# start indexing from 1. Transform unused slots (-1) to 0
	

	# Offset both 'faces' columns: we switch from 0-indexing to 1-indexing
	e_offset = tf.constant([[[0,0,1,1]]],dtype = tf.int32)
	edge_map = edge_map + e_offset

	# Then, add zero-line (order is important, so that faces in first line stay 0), since we offset v_edges
	pad_line = tf.zeros([batch_size,1,4],dtype=tf.int32)
        edge_map = tf.concat([pad_line,edge_map], axis=1) 	# Add zero-line accordingly

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


		v_dp = tensorDotProduct(n_edges_vec,n_f_normals)		# since normals should be 0 for unused edges, resulting dot product should be zero. No need to deal with it explicitly
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



def normalTensor(x,adj,tensName=None):

	# Compute normals from 3D points and adjacency graph
	# average of adjacent faces' normals.
	# 2nd version: weighted average of adjacent faces' normals. - Weights given by face angle at vertex

	K = adj.shape[2]
	#x shape = (batch, points, space_dims)
	#adj shape = (batch, points, K)
	
	#Building neighbourhood
	neigh = get_slices(x,adj)
	#shape = (batch, points, neighbourhood (K), space_dims)
	# (one col per neighbour + center point (in 1st column))

	#computing edges vectors
	edges = tf.subtract(neigh[:,:,1:,:],tf.expand_dims(neigh[:,:,0,:],2))
	#shape = (batch, points, neighbourhood (K-1), space_dims)
	# (one col per neighbour)

	#WARNING!!! set empty neighbours to zero! (otherwise, vectors pointing to origin)
	non_zeros = tf.not_equal(adj[:,:,1:], 0)
	non_zeros = tf.tile(tf.expand_dims(non_zeros,axis=-1),[1,1,1,3])
	edges = tf.where(non_zeros, edges,tf.zeros_like(edges))
	
	#generate pairs of edges. The way vertices are organized, these pairs should correspond to faces
	edges1 = edges[:,:,:-1,:]
	edges2 = edges[:,:,1:,:]
	#shape = 2*(batch, points, (K-2), space_dim)

	#compute cross product (face normals)
	edges_cp = tf.cross(edges1,edges2)
	#shape = (batch, points, pairs of edges: (K-2)*(K-1)/2, space_dim)

	#normalize

	edges_cp = normalizeTensor(edges_cp)
	#edges_cp = tf.divide(edges_cp,tf.expand_dims(tf.norm(edges_cp,axis=3),axis=-1))

	#TODO: compute weights (angles)
	n_edges1 = normalizeTensor(edges1)
	n_edges2 = normalizeTensor(edges2)
	angle_cos = tensorDotProduct(n_edges1,n_edges2)

	gtone = tf.greater(angle_cos, tf.ones_like(angle_cos))

	angle = tf.where(gtone, tf.zeros_like(angle_cos),tf.acos(angle_cos))

	weighted_normals = tf.multiply(edges_cp,tf.tile(tf.expand_dims(angle,axis=-1),[1,1,1,3]))

	#averaging on all edges (summing and mormalizing)
	#normals = tf.reduce_sum(edges_cp,axis=2)
	normals = tf.reduce_sum(weighted_normals,axis=2)
	#shape = (batch, points, space_dim)

	#normalize
	normals = normalizeTensor(normals)
	#normals = tf.divide(normals,tf.expand_dims(tf.norm(normals,axis=2),axis=-1),name=tensName)
	normals = tf.identity(normals,name=tensName)
	return normals

def simpleNormalTensor(x,faces,tensName=None):
	''' Build mesh normals '''

	batch_size = x.shape[0]
	num_points = x.shape[1]
	num_faces = faces.shape[1]
	#Reshape x and faces
	faces_bis = tf.reshape(faces,[batch_size*num_faces,3])
	x_bis = tf.reshape(x, [batch_size*num_points,3])
	#gather
	#reshape result
	T = tf.gather(x_bis,faces_bis)
	print("T shape: "+str(T.shape))
	T = tf.reshape(T,[batch_size,num_faces,3,3])
	print("T shape: "+str(T.shape))
	# shape = (batch_size,face_num,3 (vertices),3 (coordinates))
	
	
	#T = verts[faces]
	E1 = tf.subtract(T[:,:,1,:],T[:,:,0,:])
	print("E1 shape: "+str(E1.shape))
	E2 = tf.subtract(T[:,:,2,:],T[:,:,0,:])
	print("E2 shape: "+str(E2.shape))
	N = tf.cross(E1, E2)
	#shape = (batch_size,face_num, 3 (space coordinates))
	print("N shape: "+str(N.shape))

	#print("Vertices shape: "+str(verts.shape))
	#print("Faces shape: "+str(faces.shape))
	#print("N shape: "+str(N.shape))
	#print "N: ", N[10]
	Nn=normalizeTensor(N)
	#print("Nn shape: "+str(Nn.shape))
	#normals = np.zeros(verts.shape, dtype=np.float32)
	
	#normals = tf.zeros_like(x)
	normals = tf.Variable(tf.zeros_like(x))

	Nn = tf.tile(Nn,[1,3,1])
	
	faces_t = tf.transpose(faces,[0,2,1])
	faces_col = tf.reshape(faces_t,[batch_size,num_faces*3])

	faces_col = tf.expand_dims(faces_col,axis=-1)
	#temp solution
	normals = tf.reshape(normals,[batch_size*num_points,3])

	print("ref shape: "+str(normals.shape))
	print("indices shape: "+str(faces_col.shape))
	print("updates shape: "+str(Nn.shape))
	#normals = tf.scatter_add(normals,faces_col,Nn)
	normals = tf.scatter_nd_add(normals,faces_col,Nn)

	normals = tf.reshape(normals,[batch_size,num_points,3])
	#for i in range(3):
	#	normals[faces[:,i]] += Nn

	#print("Normals shape: "+str(normals.shape))
	normals = normalizeTensor(normals)
	normals = tf.identity(normals,name=tensName)
	return normals


def normalizeTensor(x):
	with tf.variable_scope("normalization"):
                #norm = tf.norm(x,axis=-1)
                square = tf.square(x,name="square")
		square_sum = tf.reduce_sum(square,axis=-1,name="square_sum")
                norm = tf.sqrt(square_sum,name="sqrt")

                norm_non_zeros = tf.greater(norm,0.0)
                inv_norm = tf.where(norm_non_zeros,tf.reciprocal(norm,name="norm_division"),tf.zeros_like(norm,name="zeros"))
		newX = tf.multiply(x, tf.expand_dims(inv_norm,axis=-1),name="result")
	return newX


def mainFunction():

        pickleLoad = True
	pickleSave = True

	K_faces = 25

	training_meshes_num = [0]
	valid_meshes_num = [0]

        binDumpPath = "/morpheo-nas2/vincent/DeepMeshRefinement/BinaryDump/"


	###################################################################################################
	#	0 - Training on all meshes in a given folder
	#	1 - Run checkpoint on a given mesh as input
	#	2 - Run checkpoint on all meshes in a folder. Compute angular diff and Haus dist
	#	3 - Test mode
	###################################################################################################


        #Takes the path to noisy and GT meshes as input, and add data to the lists fed to tensorflow graph, with the right format
        def addMesh(inputFilePath,filename, gtFilePath, gtfilename, in_list, gt_list, adj_list, mesh_count_list,images_list,calibs_list):
                new_to_old_permutations_list = []
                num_faces = []
		# --- Load mesh ---
                V0,_,_, faces0, _ = load_mesh(inputFilePath, filename+'/Noisy/000.obj', 0, False)
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

                #Load Projection Matrices
                loaded_calibs = []
                for cam_calib in os.listdir(inputFilePath+'/'+filename+"/Calib/"):
                    cam_calib_file=inputFilePath+'/'+filename+"/Calib/"+cam_calib
                    if cam_calib_file.endswith(".txt"):
                        #Read txt file
                        loaded_calibs.append(read_calib_file(cam_calib_file))

                #print("Loaded Matrices shape:",np.shape(loaded_calibs))

                #Load Corresponding Images
                loaded_images = []
                for cam_image_folder in os.listdir(inputFilePath+'/'+filename+"/Images/"):
                    image_file_name = inputFilePath+'/'+filename+"/Images/"+cam_image_folder+"/0001.png"
                    loaded_images.append(load_image(image_file_name))
                #print("Loaded Images Shape: ",np.shape(loaded_images))
                if np.shape(loaded_calibs)[0] != np.shape(loaded_images)[0]:
                    print("Data Parsing Error: mismatch between camera numbers for images and calibs")
                    exit()

		# Get patches if mesh is too big
		facesNum = faces0.shape[0]
		if facesNum>maxSize:
			patchNum = int(facesNum/patchSize)+1
			for p in range(patchNum):
				faceSeed = np.random.randint(facesNum)
				testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

				#patchFNormals = f_normals0[fOldInd]
				patchFNormals = f_normals_pos[fOldInd]
				patchGTFNormals = GTf_normals0[fOldInd]

				# Convert to sparse matrix and coarsen graph
				coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
				adjs, newToOld = coarsen(coo_adj,4)

				# There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
				new_N = len(newToOld)
				old_N = patchFNormals.shape[0]

                                ##### Save number of triangles and patch new_to_old permutation #####
                                num_faces.append(old_N)
                                new_to_old_permutations_list.append(fOldInd)
                                #####################################################################

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

                                print("f_normals shape: ",np.shape(f_normals))
                                print("GTf_normals shape: ",np.shape(GTf_normals))
                                if np.sum(np.abs(GTf_normals)) > 0.0:
                                    in_list.append(f_normals)
                                    adj_list.append(fAdjs)
                                    gt_list.append(GTf_normals)
                                    images_list.append(loaded_images)
                                    calibs_list.append(loaded_calibs)

                                    print("Added patch: mesh " + filename + ", patch " + str(p) + " (" + str(mesh_count_list[0]) + ")")
                                    mesh_count_list[0]+=1
                                else:
                                    print("Found only fake nodes graph, discarding!")
                                    exit()

		else: 		#Small mesh case

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

                        ##### Save number of triangles and patch new_to_old permutation #####
                        num_faces.append(old_N) # Keep track of fake nodes
                        new_to_old_permutations_list.append([]) # Nothing to append here, faces are already correctly ordered
                        #####################################################################

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

			in_list.append(f_normals)
			adj_list.append(fAdjs)
			gt_list.append(GTf_normals)
                        images_list.append(loaded_images)
                        calibs_list.append(loaded_calibs)
		
                        print("Added mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

			mesh_count_list[0]+=1

            def addBigMesh(inputFilePath,filename, gtFilePath, gtfilename, in_list, gt_list, adj_list, mesh_count_list,images_list,calibs_list):
                    new_to_old_permutations_list = []
                    num_faces = []
                    # --- Load mesh ---
                    V0,_,_, faces0, _ = load_mesh(inputFilePath, filename+'/Noisy/000.obj', 0, False)
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

                    #Load Projection Matrices
                    loaded_calibs = []
                    for cam_calib in os.listdir(inputFilePath+'/'+filename+"/Calib/"):
                        cam_calib_file=inputFilePath+'/'+filename+"/Calib/"+cam_calib
                        if cam_calib_file.endswith(".txt"):
                            #Read txt file
                            loaded_calibs.append(read_calib_file(cam_calib_file))

                    #print("Loaded Matrices shape:",np.shape(loaded_calibs))

                    #Load Corresponding Images
                    loaded_images = []
                    for cam_image_folder in os.listdir(inputFilePath+'/'+filename+"/Images/"):
                        image_file_name = inputFilePath+'/'+filename+"/Images/"+cam_image_folder+"/0001.png"
                        loaded_images.append(load_image(image_file_name))
                    #print("Loaded Images Shape: ",np.shape(loaded_images))
                    if np.shape(loaded_calibs)[0] != np.shape(loaded_images)[0]:
                        print("Data Parsing Error: mismatch between camera numbers for images and calibs")
                        exit()

                    # Get patches if mesh is too big
                    facesNum = faces0.shape[0]
                    if facesNum>maxSize:
                            patchNum = int(facesNum/patchSize)+1
                            for p in range(patchNum):
                                    faceSeed = np.random.randint(facesNum)
                                    testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

                                    #patchFNormals = f_normals0[fOldInd]
                                    patchFNormals = f_normals_pos[fOldInd]
                                    patchGTFNormals = GTf_normals0[fOldInd]

                                    # Convert to sparse matrix and coarsen graph
                                    coo_adj = listToSparse(testPatchAdj, patchFNormals[:,3:])
                                    adjs, newToOld = coarsen(coo_adj,4)

                                    # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                                    new_N = len(newToOld)
                                    old_N = patchFNormals.shape[0]

                                    ##### Save number of triangles and patch new_to_old permutation #####
                                    num_faces.append(old_N)
                                    new_to_old_permutations_list.append(fOldInd)
                                    #####################################################################

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

                                    print("f_normals shape: ",np.shape(f_normals))
                                    print("GTf_normals shape: ",np.shape(GTf_normals))
                                    if np.sum(np.abs(GTf_normals)) > 0.0:
                                        in_list.append(f_normals)
                                        adj_list.append(fAdjs)
                                        gt_list.append(GTf_normals)
                                        images_list.append(loaded_images)
                                        calibs_list.append(loaded_calibs)

                                        print("Added patch: mesh " + filename + ", patch " + str(p) + " (" + str(mesh_count_list[0]) + ")")
                                        mesh_count_list[0]+=1
                                    else:
                                        print("Found only fake nodes graph, discarding!")
                                        exit()

                    else: 		#Small mesh case

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

                            ##### Save number of triangles and patch new_to_old permutation #####
                            num_faces.append(old_N) # Keep track of fake nodes
                            new_to_old_permutations_list.append([]) # Nothing to append here, faces are already correctly ordered
                            #####################################################################

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

                            in_list.append(f_normals)
                            adj_list.append(fAdjs)
                            gt_list.append(GTf_normals)
                            images_list.append(loaded_images)
                            calibs_list.append(loaded_calibs)

                            print("Added mesh " + filename + " (" + str(mesh_count_list[0]) + ")")

                            mesh_count_list[0]+=1
                    return num_faces, new_to_old_permutations_list
	if running_mode == 0:
                # Training patch size
                maxSize = 1100
                patchSize = 1000

		f_normals_list = []
		f_adj_list = []
		GTfn_list = []

		valid_f_normals_list = []
		valid_f_adj_list = []
		valid_GTfn_list = []
                images_lists = []
                calibs_lists = []

                valid_images_lists = []
                valid_calibs_lists = []



                inputFilePath = "/morpheo-nas2/vincent/DeepMeshRefinement/Data/train/images"
                #validFilePath = "/morpheo-nas2/vincent/DeepMeshRefinement/Data/test/images"
                validFilePath = "/morpheo-nas2/vincent/DeepMeshRefinement/Data/valid/images"

                training_meshes_num = [0]
                valid_meshes_num = [0]

		#print("training_meshes_num 0 " + str(training_meshes_num))
		if pickleLoad:
			# Training
			with open(binDumpPath+'f_normals_list', 'rb') as fp:
				f_normals_list = pickle.load(fp)
			with open(binDumpPath+'GTfn_list', 'rb') as fp:
				GTfn_list = pickle.load(fp)
			with open(binDumpPath+'f_adj_list', 'rb') as fp:
                                f_adj_list = pickle.load(fp)
                        with open(binDumpPath+'images_lists', 'rb') as fp:
                                images_lists = pickle.load(fp)
                        with open(binDumpPath+'calibs_lists', 'rb') as fp:
                                calibs_lists = pickle.load(fp)
			# Validation
			with open(binDumpPath+'valid_f_normals_list', 'rb') as fp:
				valid_f_normals_list = pickle.load(fp)
			with open(binDumpPath+'valid_GTfn_list', 'rb') as fp:
				valid_GTfn_list = pickle.load(fp)
			with open(binDumpPath+'valid_f_adj_list', 'rb') as fp:
				valid_f_adj_list = pickle.load(fp)
                        with open(binDumpPath+'valid_images_lists', 'rb') as fp:
                                valid_images_lists = pickle.load(fp)
                        with open(binDumpPath+'valid_calibs_lists', 'rb') as fp:
                                valid_calibs_lists = pickle.load(fp)

		else:
			# Training set
			for filename in os.listdir(inputFilePath):
                            print("Adding " + filename + " (" + str(training_meshes_num[0]) + ")")
                            gtfilename = filename+"/Ground_Truth/000.obj"
                            addMesh(inputFilePath, filename, inputFilePath, gtfilename, f_normals_list, GTfn_list, f_adj_list, training_meshes_num,images_lists,calibs_lists)

			# Validation set
			for filename in os.listdir(validFilePath):
                            gtfilename = filename+"/Ground_Truth/000.obj"
                            addMesh(validFilePath, filename, validFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_meshes_num,valid_images_lists,valid_calibs_lists)
					
			if pickleSave:
				# Training
				with open(binDumpPath+'f_normals_list', 'wb') as fp:
					pickle.dump(f_normals_list, fp)
				with open(binDumpPath+'GTfn_list', 'wb') as fp:
					pickle.dump(GTfn_list, fp)
				with open(binDumpPath+'f_adj_list', 'wb') as fp:
					pickle.dump(f_adj_list, fp)
                                with open(binDumpPath+'images_lists', 'wb') as fp:
                                        pickle.dump(images_lists, fp)
                                with open(binDumpPath+'calibs_lists', 'wb') as fp:
                                        pickle.dump(calibs_lists, fp)
				# Validation
				with open(binDumpPath+'valid_f_normals_list', 'wb') as fp:
					pickle.dump(valid_f_normals_list, fp)
				with open(binDumpPath+'valid_GTfn_list', 'wb') as fp:
					pickle.dump(valid_GTfn_list, fp)
				with open(binDumpPath+'valid_f_adj_list', 'wb') as fp:
					pickle.dump(valid_f_adj_list, fp)
                                with open(binDumpPath+'valid_images_lists', 'wb') as fp:
                                        pickle.dump(valid_images_lists, fp)
                                with open(binDumpPath+'valid_calibs_lists', 'wb') as fp:
                                        pickle.dump(valid_calibs_lists, fp)

                ## DEV: cut only 5 first samples of first entry #############################################################
                """
                num_samp = 1
                f_normals_list = f_normals_list[0:1]
                f_normals_list = [[f_normals_list[0][0][0:num_samp]]]
                #print(np.shape(f_normals_list))
                f_adj_list = f_adj_list[0:1]
                f_adj_list = [[f_adj_list[0][0][0:num_samp]]]
                #print(np.shape(f_adj_list))
                GTfn_list = GTfn_list[0:1]
                GTfn_list = [[GTfn_list[0][0][0:num_samp]]]
                #print(np.shape(GTfn_list))
                images_lists = images_lists[0:1]
                #print(np.shape(images_lists))
                calibs_lists = calibs_lists[0:1]
                #print(np.shape(calibs_lists))
                valid_f_normals_list = valid_f_normals_list[0:1]
                valid_f_normals_list = [[valid_f_normals_list[0][0][0:num_samp]]]
                #print(np.shape(valid_f_normals_list))
                valid_GTfn_list = valid_GTfn_list[0:1]
                valid_GTfn_list = [[valid_GTfn_list[0][0][0:num_samp]]]
                #print(np.shape(valid_GTfn_list))
                valid_f_adj_list = valid_f_adj_list[0:1]
                valid_f_adj_list = [[valid_f_adj_list[0][0][0:num_samp]]]
                #print(np.shape(valid_f_adj_list))
                valid_images_lists = valid_images_lists[0:1]
                #print(np.shape(valid_images_lists))
                valid_calibs_lists = valid_calibs_lists[0:1]
                #print(np.shape(valid_calibs_lists))

                f_normals_list = np.array(f_normals_list)
                GTfn_list = np.array(GTfn_list)
                f_adj_list = np.array(f_adj_list)
                images_lists = np.array(images_lists)
                calibs_lists = np.array(calibs_lists)
                valid_f_normals_list = np.array(valid_f_normals_list)
                valid_GTfn_list = np.array(valid_GTfn_list)
                valid_f_adj_list = np.array(valid_f_adj_list)
                valid_images_lists = np.array(valid_images_lists)
                valid_calibs_lists = np.array(valid_calibs_lists)
                """
                #############################################################

                trainNet(f_normals_list, GTfn_list, f_adj_list,images_lists,calibs_lists, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list,valid_images_lists,valid_calibs_lists)




                """
                elif running_mode == 1:
		# # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/MeshesDB/"
		# inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/BlenderDB/"
		# gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/BlenderDB/"
		# #inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/SmoothedDB/"

		inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/TrainingBase/Validation/"
		gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/TrainingBase/"

		inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/Generated/"
		gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/"

		inputFileName = "Chinese_dragon_65k_1_noisy.obj"
		gtFileName = "Chinese_dragon_65k_1_gt.obj"

		inputFileName = "bunny_iH_noisy_2.obj"
		gtFileName = "bunny_iH.obj"		

		V0,_,num_neighbours0, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)

		f_normals0 = computeFacesNormals(V0, faces0)
		_, edge_map0, v_e_map0 = getFacesAdj2(faces0)
		f_adj0 = getFacesLargeAdj(faces0,K_faces)

		# f_pos0 = getTrianglesBarycenter(V0, faces0)
		# f_pos0 = np.reshape(f_pos0,(-1,3))
		# f_normals0 = np.concatenate((f_normals0, f_pos0), axis=1)

		GT0,_,_,GTfaces0,_ = load_mesh(gtFilePath, gtFileName, 0, False)

		GTf_normals0 = computeFacesNormals(GT0, GTfaces0)


		print("Starting DL")
		V = np.expand_dims(V0, axis=0)
		GT = np.expand_dims(GT0, axis=0)
		#normals = np.expand_dims(normals0,axis=0)
		faces = np.expand_dims(faces0,axis=0)
		faces = np.array(faces).astype(np.int32)

		f_normals = np.expand_dims(f_normals0, axis=0)
		f_adj = np.expand_dims(f_adj0, axis=0)
		#GTf_normals = np.expand_dims(GTf_normals0, axis=0)
		edge_map = np.expand_dims(edge_map0, axis=0)
		v_e_map = np.expand_dims(v_e_map0, axis=0)

		upV, upN = run_thingy(V, faces, f_normals, f_adj, edge_map, v_e_map)

		haus_dist = oneSidedHausdorff(V0, GT0)
		print("noisy Haus_dist = " + str(haus_dist))

		haus_dist = oneSidedHausdorff(upV, GT0)
		print("denoized Haus_dist = " + str(haus_dist))

		angDist = angularDiff(f_normals0, GTf_normals0)
		print("noisy angular diff = "+str(angDist))

		angDist = angularDiff(upN, GTf_normals0)
		print("denoized angular diff = "+str(angDist))

		print("upV shape: " + str(upV.shape))
		print("faces shape: " + str(faces0.shape))

		write_mesh(upV, faces0, RESULTS_PATH+"testOut.obj")

		trainNet(f_normals_list, GTfn_list, f_adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list)
                """

	elif running_mode == 2:
                # Training patch size
                maxSize = 100100
                patchSize = 100000

                # Generate array of metrics on reconstructions
		nameArray = []		# String array, to now which row is what
		resultsArray = []	# results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
							# [Max distance, Mean distance, Mean angle, std angle, face num]
                #DataFolder = "/morpheo-nas2/vincent/DeepMeshRefinement/Data/test/images/"
                DataFolder = "/morpheo-nas2/vincent/DeepMeshRefinement/Data/dummy/images/"

                f_normals_list = []
                GTfn_list = []
                f_adj_list = []
                images_lists = []
                calibs_lists = []

                mesh_count = [0]

		# results file name
                csv_filename = RESULTS_PATH+"/results_eval.csv"
		
		# Get GT mesh
                for filename in os.listdir(DataFolder):
			nameArray = []
			resultsArray = []
                        gtFolder=DataFolder+filename+"/Ground_Truth/"
                        noisyFolder=DataFolder+filename+"/Noisy/"
                        gtFileName ="000.obj"
                        # Get smooth meshes
                        noisyFile0 = "000.obj"
                        denoizedFile0 = filename+"_denoized.obj"

                        # Load GT and Noisy Meshes
                        gtfilename = filename+"/Ground_Truth/000.obj"
                        faces_num, permutations = addBigMesh(DataFolder, filename, DataFolder, gtfilename, f_normals_list, GTfn_list, f_adj_list, mesh_count,images_lists,calibs_lists)

                        # Now recover vertices positions and create Edge maps
                        V0,_,_, faces_noisy, _ = load_mesh(noisyFolder, noisyFile0, 0, False)

                        _, edge_map_noisy , v_e_map_noisy  = getFacesAdj2(faces_noisy)
                        f_adj_noisy = getFacesLargeAdj(faces_noisy,K_faces)

                        edge_map_noisy = np.expand_dims(edge_map_noisy, axis=0)
                        v_e_map_noisy = np.expand_dims(v_e_map_noisy, axis=0)

                        print("running ...")
                        upV, upN0 = inferNet(V0, f_normals_list, f_adj_list, edge_map_noisy, v_e_map_noisy,images_lists,calibs_lists,faces_num, permutations,faces_noisy.shape[0])
                        write_mesh(upV, faces_noisy, RESULTS_PATH+denoizedFile0)
                        #print("computing Hausdorff 1...")
                        #haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
                        #angDist0, angStd0 = angularDiff(upN0, GTf_normals)
                        #write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile0)

                        # Fill arrays
                        #nameArray.append(denoizedFile0)
                        #resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum])

                        #outputFile = open(csv_filename,'a')
                        #nameArray = np.array(nameArray)
                        #resultsArray = np.array(resultsArray,dtype=np.float32)

                        #tempArray = resultsArray.flatten()
                        #resStr = ["%.7f" % number for number in tempArray]
                        #resStr = np.reshape(resStr,resultsArray.shape)

                        #nameArray = np.expand_dims(nameArray, axis=-1)

                        #finalArray = np.concatenate((nameArray,resStr),axis=1)
                        #for row in range(finalArray.shape[0]):
                        #        for col in range(finalArray.shape[1]):
                        #                outputFile.write(finalArray[row,col])
                        #                outputFile.write(' ')
                        #        outputFile.write('\n')

                        #outputFile.close()

        """
	elif running_mode == 3:
		inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/"
		
		
		#inputFileName = "bunny_iH_noisy_2.obj"
		inputFileName = "armadillo.obj"
		
		V0,_,_, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)

                    f_pos0 = getTrianglesBarycenter(V0, faces_gt)
                    f_pos0 = np.reshape(f_pos0,(-1,3))
                    f_normals0 = np.concatenate((f_normals0, f_pos0), axis=1)

                    V0 = np.expand_dims(V0, axis=0)
                    f_normals0 = np.expand_dims(f_normals0, axis=0)

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
                    GTf_normals0 = GTf_normals0[newToOld]train_samp
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



		f_normals0 = computeFacesNormals(V0, faces0)
		_, edge_map0, v_e_map0 = getFacesAdj2(faces0)
		f_adj0 = getFacesLargeAdj(faces0,K_faces)

				print("running n1...")
				upV0, upN0 = run_thingy(V0, faces, f_normals0, f_adj, edge_map, v_e_map)
				print("computing Hausdorff 1...")
				haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
				angDist0, angStd0 = angularDiff(upN0, GTf_normals)
				write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile0)

				print("running n1...")
				upV0, upN0 = inferNet(V0, faces0, f_normals_pos0, fAdjs, edge_map0, v_e_map)
				print("computing Hausdorff 1...")
				haus_dist0, avg_dist0 = oneSidedHausdorff(upV0, GT)
				angDist0, angStd0 = angularDiff(upN0, GTf_normals0)
				write_mesh(upV0, faces[0,:,:], RESULTS_PATH+denoizedFile0)


		faceSeed = np.random.randint(faces0.shape[0])

		for i in range(10):
			faceSeed = np.random.randint(faces0.shape[0])
			testPatchV, testPatchF, testPatchAdj = getMeshPatch(V0, faces0, f_adj0, 10000, faceSeed)

		# print("testPatchV = "+str(testPatchV))
		# print("testPatchF = "+str(testPatchF))
		# print("testPatchAdj = "+str(testPatchAdj))


                write_mesh(testPatchV, testPatchF, "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/testPatch"+str(i)+".obj")

                f_pos1 = getTrianglesBarycenter(V1, faces_gt)
                f_pos1 = np.reshape(f_pos1,(-1,3))
                f_normals_pos1 = np.concatenate((f_normals1, f_pos1), axis=1)

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(f_adj, f_pos1)
                adjs, newToOld = coarsen(coo_adj,4)
                oldToNew = np.array(inv_perm(newToOld))

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                old_N = facesNum
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                f_normals_pos1 = np.concatenate((f_normals_pos1,padding6),axis=0)
                GTf_normals1 = np.concatenate((GTf_normals, padding3),axis=0)
                faces1 = np.concatenate((faces_gt, padding3),axis=0)
                # Reorder nodes
                f_normals_pos1 = f_normals_pos1[newToOld]
                GTf_normals1 = GTf_normals1[newToOld]
                faces1 = faces1[newToOld]

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
                edge_map1 = np.concatenate((emap_v,emap_f),axis=-1)
                edge_map1 = np.expand_dims(edge_map1, axis=0)


                V1 = np.expand_dims(V1, axis=0)
                faces1 = np.expand_dims(faces1,axis=0)
                f_normals_pos1 = np.expand_dims(f_normals_pos1, axis=0)
                print("running n2...")
                upV1, upN1 = inferNet(V1, faces1, f_normals_pos1, fAdjs, edge_map1, v_e_map)
                print("computing Hausdorff 2...")
                haus_dist1, avg_dist1 = oneSidedHausdorff(upV1, GT)
                angDist1, angStd1 = angularDiff(upN1, GTf_normals1)
                write_mesh(upV1, faces[0,:,:], RESULTS_PATH+denoizedFile1)

        """
        """
        elif running_mode == 1:
                # # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/MeshesDB/"
                # inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/BlenderDB/"
                # gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/BlenderDB/"
                # #inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/TrainingBase/SmoothedDB/"

                inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/TrainingBase/Validation/"
                gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/TrainingBase/"

                inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/Generated/"
                gtFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/"


                inputFileName = "Chinese_dragon_65k_1_noisy.obj"
                gtFileName = "Chinese_dragon_65k_1_gt.obj"


                f_pos2 = getTrianglesBarycenter(V2, faces_gt)
                f_pos2 = np.reshape(f_pos2,(-1,3))
                f_normals_pos2 = np.concatenate((f_normals2, f_pos2), axis=1)

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparse(f_adj, f_pos2)
                adjs, newToOld = coarsen(coo_adj,4)
                oldToNew = np.array(inv_perm(newToOld))

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                old_N = facesNum
                padding6 =np.zeros((new_N-old_N,6))
                padding3 =np.zeros((new_N-old_N,3))
                f_normals_pos2 = np.concatenate((f_normals_pos2,padding6),axis=0)
                GTf_normals2 = np.concatenate((GTf_normals, padding3),axis=0)
                faces2 = np.concatenate((faces_gt, padding3),axis=0)
                # Reorder nodes
                f_normals_pos2 = f_normals_pos2[newToOld]
                GTf_normals2 = GTf_normals2[newToOld]
                faces2 = faces2[newToOld]

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
                edge_map2 = np.concatenate((emap_v,emap_f),axis=-1)
                edge_map2 = np.expand_dims(edge_map2, axis=0)


                V2 = np.expand_dims(V2, axis=0)
                faces2 = np.expand_dims(faces2,axis=0)
                f_normals_pos2 = np.expand_dims(f_normals_pos2, axis=0)
                print("running n3...")
                upV2, upN2 = inferNet(V2, faces2, f_normals_pos2, fAdjs, edge_map2, v_e_map)
                print("computing Hausdorff 3...")
                haus_dist2, avg_dist2 = oneSidedHausdorff(upV2, GT)
                angDist2, angStd2 = angularDiff(upN2, GTf_normals2)
                write_mesh(upV2, faces[0,:,:], RESULTS_PATH+denoizedFile2)

                V0,_,num_neighbours0, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)

                f_normals0 = computeFacesNormals(V0, faces0)
                _, edge_map0, v_e_map0 = getFacesAdj2(faces0)
                f_adj0 = getFacesLargeAdj(faces0,K_faces)



                outputFile = open(csv_filename,'a')
                nameArray = np.array(nameArray)
                resultsArray = np.array(resultsArray,dtype=np.float32)

                GT0,_,_,GTfaces0,_ = load_mesh(gtFilePath, gtFileName, 0, False)


                tempArray = resultsArray.flatten()
                resStr = ["%.7f" % number for number in tempArray]
                resStr = np.reshape(resStr,resultsArray.shape)



                GTf_normals0 = computeFacesNormals(GT0, GTfaces0)

                nameArray = np.expand_dims(nameArray, axis=-1)



                finalArray = np.concatenate((nameArray,resStr),axis=1)
                for row in range(finalArray.shape[0]):
                        for col in range(finalArray.shape[1]):
                                outputFile.write(finalArray[row,col])
                                outputFile.write(' ')
                        outputFile.write('\n')

                print("Starting DL")
                V = np.expand_dims(V0, axis=0)
                GT = np.expand_dims(GT0, axis=0)
                #normals = np.expand_dims(normals0,axis=0)
                faces = np.expand_dims(faces0,axis=0)
                faces = np.array(faces).astype(np.int32)

                outputFile.close()

                f_normals = np.expand_dims(f_normals0, axis=0)
                f_adj = np.expand_dims(f_adj0, axis=0)
                #GTf_normals = np.expand_dims(GTf_normals0, axis=0)
                edge_map = np.expand_dims(edge_map0, axis=0)
                v_e_map = np.expand_dims(v_e_map0, axis=0)

                upV, upN = run_thingy(V, faces, f_normals, f_adj, edge_map, v_e_map)


                haus_dist = oneSidedHausdorff(V0, GT0)
                print("noisy Haus_dist = " + str(haus_dist))

		inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/"
		
		
		#inputFileName = "bunny_iH_noisy_2.obj"
		inputFileName = "armadillo.obj"
		
		V0,_,_, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)

		inputFilePath = "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/Benchmark/"
		
		
		inputFileName = "bunny_iH.obj"
		#inputFileName = "armadillo.obj"
		
		V0,_,_, faces0, _ = load_mesh(inputFilePath, inputFileName, 0, False)


                haus_dist = oneSidedHausdorff(upV, GT0)
                print("denoized Haus_dist = " + str(haus_dist))


                angDist = angularDiff(f_normals0, GTf_normals0)
                print("noisy angular diff = "+str(angDist))

		faceSeed = np.random.randint(faces0.shape[0])

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



                angDist = angularDiff(upN, GTf_normals0)
                print("denoized angular diff = "+str(angDist))

		for i in range(10):
			faceSeed = np.random.randint(faces0.shape[0])
			testPatchV, testPatchF, testPatchAdj = getMeshPatch(V0, faces0, f_adj0, 10000, faceSeed)

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
		# 	faceSeed = np.random.randint(faces0.shape[0])
		# 	testPatchV, testPatchF, testPatchAdj = getMeshPatch(V0, faces0, f_adj0, 10000, faceSeed)


                print("upV shape: " + str(upV.shape))
                print("faces shape: " + str(faces0.shape))


                write_mesh(upV, faces0, RESULTS_PATH+"testPatch"+str(i)+".obj")
        """
        """
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
		# 	if (filename.endswith(".obj")):
		# 		gtfilename = filename[:-gtnameoffset]+".obj"
		# 		addMesh(validFilePath, filename, gtFilePath, gtfilename, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list, valid_meshes_num)



		# trainNet(f_normals_list, GTfn_list, f_adj_list, valid_f_normals_list, valid_GTfn_list, valid_f_adj_list)

			# write_mesh(testPatchV, testPatchF, "/morpheo-nas/marmando/DeepMeshRefinement/paper-dataset/testPatch"+str(i)+".obj")
        """


def refineMesh(x,displacement,normals,adj):		#params are tensors
	
	batch_size, num_points, in_channels = x.get_shape().as_list()
	K = adj.shape[2]
	#compute displacement vector for each point
	displacement_vector = tf.multiply(tf.tile(displacement,[1,1,3]),normals)

	#return displacement_vector+x
	# deformation model:
	# each point vi is displaced by (1/2)*di + (1/2) * sum(j in neighbourhood Ni) ((1/|Ni|) * dj)

	#First, compute neighbourhood size:

	# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
	adj_size = tf.count_nonzero(adj, 2)-1 		# -1 because we do not include point in its neighbourhood this time
	#deal with unconnected points: replace NaN with 0
	non_zeros = tf.not_equal(adj_size, 0)
	adj_size = tf.cast(adj_size, tf.float32)
	adj_size = tf.where(non_zeros,tf.reciprocal(adj_size,name="neighbourhood_size_inv")*0.5,tf.zeros_like(adj_size))	#1 / 2*|Ni|
	# [batch_size, input_size, 1, 1]
	#adj_size = tf.reshape(adj_size, [batch_size, input_size, 1])
	adj_size = tf.expand_dims(adj_size,axis=-1)

	# Then, get slices of displacement vectors and weight depending on neighbourhood
	# 

	#n_weight = get_slices(tf.expand_dims(adj_size,axis=-1),adj[:,:,1:])

	n_weight = tf.tile(adj_size,[1,1,K-1])
	#n_weight = get_slices(adj_size,adj[:,:,1:])

	n_weight = tf.concat([tf.constant(0.5,shape=[batch_size,num_points,1]),n_weight],axis=2)		# 1 for cloumn K=0, 1 for channels number (in_channels)
	n_weight = tf.tile(tf.expand_dims(n_weight,axis=-1),[1,1,1,3])

	n_displacement = get_slices(displacement_vector,adj)

	total_displacement = tf.multiply(n_weight,n_displacement)

	total_displacement = tf.reduce_sum(total_displacement, axis=2)

	return total_displacement+x


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--architecture', type=int, default=0)
	#parser.add_argument('--dataset_path')
	parser.add_argument('--results_path', type=str)
	parser.add_argument('--network_path', type=str)
	parser.add_argument('--num_iterations', type=int, default=100)
	parser.add_argument('--debug', type=bool, default=False)
        parser.add_argument('--device', type=str, default='/gpu:0')
	parser.add_argument('--net_name', type=str, default='unnamed_net')
	parser.add_argument('--pretrained', type=bool, default=False)
        parser.add_argument('--mode',type=int,default=0)

	FLAGS = parser.parse_args()

	ARCHITECTURE = FLAGS.architecture
	#DATASET_PATH = FLAGS.dataset_path
	RESULTS_PATH = FLAGS.results_path
	NETWORK_PATH = FLAGS.network_path
	NUM_ITERATIONS = FLAGS.num_iterations
	DEVICE = FLAGS.device

        running_mode = FLAGS.mode
	NET_NAME = FLAGS.net_name


	mainFunction()


