from __future__ import division
import numpy as np
import math
import tensorflow as tf
import os
try:
    import queue
except ImportError:
    import Queue as queue

import scipy.sparse
import itertools

#import h5py

def one_hot_encoding_batch_per_point(y, num_classes):
    y_one_hot = np.zeros((y.shape[0], y.shape[1], num_classes))
    for i in xrange(y.shape[0]):
        for j in xrange(y.shape[1]):
            y_one_hot[i, j, y[i][j]] = 1
    return y_one_hot

def one_hot_encoding_batch(y, num_classes):
    print("y shape: "+str(y.shape))
    print("y type: "+str(y.dtype))
    y = y.astype(int)
    print("y type: "+str(y.dtype))
    y_one_hot = np.zeros((y.shape[0], num_classes))
    for i in xrange(y.shape[0]):
        y_one_hot[i, y[i]] = 1
    return y_one_hot

def one_hot_encoding(y, num_classes):
    y_one_hot = np.zeros((num_classes))
    y_one_hot[y] = 1
    return y_one_hot


# -------------------------------------------------------------
# ---				GEOMETRY FUNCTIONS						---
# -------------------------------------------------------------

def normalize(a):
    norms = np.sqrt((a * a).sum(1))[:,np.newaxis]+0.00000001
    return a * (1 / norms)


def tensorDotProduct(x,y, name=None):
    if name==None:
        name="dot_product"
    with tf.variable_scope(name):
        return tf.reduce_sum(tf.multiply(x,y),axis=-1, name="result")


def computeNormals(verts,faces):    # Returns per vertex normal (right?)
    ''' Build mesh normals '''
    T = verts[faces]
    N = np.cross(T[::,1 ]-T[::,0], T[::,2]-T[::,0])

    # print("Vertices shape: "+str(verts.shape))
    # print("Faces shape: "+str(faces.shape))
    # print("N shape: "+str(N.shape))
    #print "N: ", N[10]
    Nn=normalize(N)
    #print("Nn shape: "+str(Nn.shape))
    normals = np.zeros(verts.shape, dtype=np.float32)
    for i in range(3): normals[faces[:,i]] += Nn
    #print("Normals shape: "+str(normals.shape))
    normals = normalize(normals)
    return normals

def computeFacesNormals(verts, faces):
    T = verts[faces]
    N = np.cross(T[::,1 ]-T[::,0], T[::,2]-T[::,0])
    Nn=normalize(N)

    return Nn


def getFacesAdj(faces):
    fnum = faces.shape[0]
    fadj = np.zeros([fnum,4], dtype=np.int32)     # triangular faces only
    find = np.ones([fnum], dtype=np.int8)

    for i in range(fnum):
        fadj[i,0] = i+1           # indexed from 1

        j = i+1                     # check next faces only
        v1 = faces[i,0]
        v2 = faces[i,1]
        v3 = faces[i,2]

        while(find[i]<4):
            if (v1 in faces[j,:]):
                if (v2 in faces[j,:]) or (v3 in faces[j,:]):
                    fadj[i,find[i]] = j+1
                    fadj[j,find[j]] = i+1
                    find[i]+=1
                    find[j]+=1
            if (v2 in faces[j,:]) and (v3 in faces[j,:]):
                fadj[i,find[i]] = j+1
                fadj[j,find[j]] = i+1
                find[i]+=1
                find[j]+=1
            j+=1

            if j==fnum:
                print("WARNING: inconsistent face data!!!")
                break

    return fadj

def getFacesAdj2(faces):    #trying other method using edges as itnermediate representation
    fnum = faces.shape[0]
    fadj = np.zeros([fnum,4], dtype=np.int32)     # triangular faces only
    find = np.ones([fnum], dtype=np.int8)

    # First, generate edges
    e_map = np.zeros([fnum*3,4],dtype=np.int32)    # for each edge, e_map(e) = [v1,v2,f1,f2]
    e_map -=1
    eind = 0
    vnum = np.amax(faces)+1
    v_e_map = np.zeros([vnum,50],dtype=np.int32)
    v_e_map-=1
    v_e_map_ind = np.zeros([vnum],dtype=np.int32)


    for f in range(fnum):
        v1 = faces[f,0]
        v2 = faces[f,1]
        v3 = faces[f,2]
        e12=False
        e13=False
        e23=False

        for ne in range(v_e_map_ind[v1]):                   # For each edge already linked to this vertex
            ce = v_e_map[v1,ne]                             # Get edge index
            if(e_map[ce,0]==v2)or(e_map[ce,1]==v2):
                e12=True                                    # edge 12 found
                if (e_map[ce,3]!=-1):
                    print("ERROR (12): face "+str(f)+", edge "+str(ce)+": ("+str(v1)+","+str(v2)+")")
                    print("Registered faces: "+str(e_map[ce,2])+", "+str(e_map[ce,3]))
                    print("v1 edges: ("+str(v_e_map[v1,:])+")")
                e_map[ce,3]=f                               # Add face to edge

            if(e_map[ce,0]==v3)or(e_map[ce,1]==v3):
                e13=True                                    # edge 13 found
                if (e_map[ce,3]!=-1):
                    print("ERROR (13): face "+str(f)+", edge "+str(ce)+": ("+str(v1)+","+str(v3)+")")
                    print("Registered faces: "+str(e_map[ce,2])+", "+str(e_map[ce,3]))
                    print("v1 edges: ("+str(v_e_map[v1,:])+")")
                e_map[ce,3]=f                               # Add face to edge

        for ne in range(v_e_map_ind[v2]):                   # Check v2 for edge 23
            ce = v_e_map[v2,ne]
            if(e_map[ce,0]==v3)or(e_map[ce,1]==v3):
                e23=True                                    # edge 23 found
                if (e_map[ce,3]!=-1):
                    print("ERROR (23): face "+str(f)+", edge "+str(ce)+": ("+str(v2)+","+str(v3)+")")
                    print("Registered faces: "+str(e_map[ce,2])+", "+str(e_map[ce,3]))
                    print("v2 edges: ("+str(v_e_map[v2,:])+")")
                e_map[ce,3]=f                               # Add face to edge

        # All potential edges have been checked
        # Add new ones if not found
        if not e12:
            # Add edge entry
            e_map[eind,0] = v1
            e_map[eind,1] = v2
            e_map[eind,2] = f
            # Add edge to vertex, increment vertex edges count
            v_e_map[v1, v_e_map_ind[v1]] = eind
            v_e_map_ind[v1]+=1
            v_e_map[v2, v_e_map_ind[v2]] = eind
            v_e_map_ind[v2]+=1
            # Increment total edge count
            eind+=1
        if not e13:
            # Add edge entry
            e_map[eind,0] = v1
            e_map[eind,1] = v3
            e_map[eind,2] = f
            # Add edge to vertex, increment vertex edges count
            v_e_map[v1, v_e_map_ind[v1]] = eind
            v_e_map_ind[v1]+=1
            v_e_map[v3, v_e_map_ind[v3]] = eind
            v_e_map_ind[v3]+=1
            # Increment total edge count
            eind+=1
        if not e23:
            # Add edge entry
            e_map[eind,0] = v2
            e_map[eind,1] = v3
            e_map[eind,2] = f
            # Add edge to vertex, increment vertex edges count
            v_e_map[v2, v_e_map_ind[v2]] = eind
            v_e_map_ind[v2]+=1
            v_e_map[v3, v_e_map_ind[v3]] = eind
            v_e_map_ind[v3]+=1
            # Increment total edge count
            eind+=1

    # Edge map done!
    # Now, go through it to generate faces adj graph

    for i in range(fnum):
        fadj[i,0] = i+1           # indexed from 1

    for e in range(eind):
        f1 = e_map[e,2]
        f2 = e_map[e,3]

        if (f1>0)and(f2>0):
            if find[f1]==4:
                print("ERROR: edge " + str(e) + " ("+str(e_map[e,:]) + ")")
                print("f1 adj: " + str(fadj[f1,:]))
            if find[f2]==4:
                print("ERROR: edge " + str(e) + " ("+str(e_map[e,:]) + ")")
                print("f2 adj: " + str(fadj[f2,:]))
            fadj[f1,find[f1]] = f2+1
            fadj[f2,find[f2]] = f1+1
            find[f1]+=1
            find[f2]+=1

    return fadj, e_map, v_e_map


def getFacesLargeAdj(faces, K):     # First try: don't filter duplicate for edge-connected faces (i.e. neighbours of type I will count twice)

    fnum = faces.shape[0]
    fadj = np.zeros([fnum,K], dtype=np.int32)     # triangular faces only
    find = np.ones([fnum], dtype=np.int8)

    vnum = int(fnum*0.55) + 50  # Arbitrary choice
    v_adj = np.zeros([vnum,3*K],dtype=np.int32)  # Will contain a list of adjacent faces for each vertex
    v_ind = np.zeros([vnum], dtype=np.int32)     # Keep track of index for each vertex

    unregistered_connections = 0

    # Go through all faces, fill v_adj
    for f in range(fnum):
        v1 = faces[f,0]
        v2 = faces[f,1]
        v3 = faces[f,2]

        v_adj[v1,v_ind[v1]]=f
        v_adj[v2,v_ind[v2]]=f
        v_adj[v3,v_ind[v3]]=f
        v_ind[v1]+=1
        v_ind[v2]+=1
        v_ind[v3]+=1


    for i in range(fnum):
        fadj[i,0] = i+1             # indexed from 1


    for v in range(vnum):           # Now, fill faces pairs
        if v_ind[v]==0:
            break                   # We've reach the last vertex
        for vf1 in range(v_ind[v]):
            for vf2 in range(vf1+1,v_ind[v]):
                f1 = v_adj[v,vf1]
                f2 = v_adj[v,vf2]
                if(find[f1]==K):
                    #print("Warning: face "+str(f1)+" already has "+str(K)+" neighbours!")
                    unregistered_connections+=1
                else:
                    fadj[f1,find[f1]] = f2+1
                    find[f1]+=1
                if(find[f2]==K):
                    #print("Warning: face "+str(f2)+" already has "+str(K)+" neighbours!")
                    unregistered_connections+=1
                else:
                    fadj[f2,find[f2]] = f1+1
                    find[f2]+=1
    if unregistered_connections>0:
        print("unregistered connections (faces): "+str(unregistered_connections/2))

    return fadj

def writeStringArray(myArr,strFileName, faces):

    outputFile = open(strFileName,"w")

    for row in range(myArr.shape[0]):
        for col in range(myArr.shape[1]):
            outputFile.write(myArr[row,col].decode('UTF_8'))
            outputFile.write(' ')
        outputFile.write('\n')

    #transform to one-indexed for obj format
    faces = faces+1
    
    faces = faces.astype(str)
    for row in range(faces.shape[0]):
        if((faces[row,0]=='1')and(faces[row,1]=='1')):
            break
        outputFile.write('f ')
        for col in range(faces.shape[1]):
            outputFile.write(faces[row,col])
            outputFile.write(' ')
        outputFile.write('\n')




def load_image(path,filename):
    f = os.path.join(path,filename)
    return skimage.data.imread(f)

def load_mesh(path,filename,K,bGetAdj):
    strFileName = os.path.join(path,filename)
    print(strFileName)
    vertices = []
    adj=[]
    # texcoords = []    not used
    normals = []
    faces = []
    free_ind=[]         #number of neighbours, per vertex
    
    #first, read file and build arrays of vertices and faces
    for line in open(strFileName, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'mtllib':
            continue
        if values[0] == 'v':
            vertices.append(list(map(float, values[1:4])))
            
        elif values[0] == 'vn':
            v = list(map(float, values[1:4]))
            normals.append(v)
        elif values[0] == 'vt':
            # texcoords.append(list(map(float, values[1:3])))
            continue
        elif values[0] in ('usemtl', 'usemat'):
            continue
        elif values[0] == 'f':
            for triNum in range(len(values)-3):  ## one line fan triangulation to triangulate polygons of size > 3
                v = values[1]
                w = v.split('/')
                faces.append(int(w[0])-1)           #vertex index (1st vertex)

                for v in values[triNum+2:triNum+4]:
                    w = v.split('/')
                    faces.append(int(w[0])-1)           #vertex index (additional vertices)


    
    vertices = np.array(vertices).astype(np.float32)

    nb_vert = vertices.shape[0]

    # If 16 bits are not enough to write vertex indices, use 32 bits 
    if nb_vert<65536:
        faces = np.array(faces).reshape(len(faces) // 3, 3).astype(np.uint16)
    else:
        faces = np.array(faces).reshape(len(faces) // 3, 3).astype(np.uint32)


    #print("Vertices and faces loaded")


            # #Then, remove duplicated vertices (keep the record of new indices), and generate adj array
            # vertT=np.transpose(vertices)

            # #print("vertices: "+str(vertices.shape))

            # new_vertices, new_vert_ind = unique_columns2(vertT)

            # print(len(new_vertices))
            # print(len(new_vertices[0]))
            # #print("new vertices: "+str(len(new_vert_ind)))

            # new_nb_vert = len(new_vertices[0])
            # new_vertices = np.array(new_vertices).reshape(3,new_nb_vert)
            # new_vertices = np.transpose(new_vertices)

            # #reindex faces:
            # for f in range(faces.shape[0]):
            #   for v in range(faces.shape[1]):     # = 3
            #       faces[f,v] = new_vert_ind[faces[f,v]]


    # This line replaces the above block that cleans duplicated vertices
    # Better not to use it when input data is already fine: risks of merging vertices (creating non-manifold points)
    # Besides, not desirable if there is a one-one relation between GT and noisy vertices
    new_vertices = vertices

    #WARNING: Faces are zero-indexed!!! (not like in obj files)

    #print("Duplicated vertices removed")

    #adjacency graph
    # Indices start at one and vertices must index themselves!

    unregistered_connections=0

    #Order matters!!
    # Edges must be in order (i.e. consecutive edges are adjacent and we turn in direct order (check input data)
    # At least, we assume faces in input mesh are oriented in direct order
    if(bGetAdj):        # boolean variable to avoid computing adjacency graph when not necessary
        #fill adj
        new_nb_vert = new_vertices.shape[0]             # get number of vertices
        adj = np.zeros((new_nb_vert,K))                 # create adj list
        temp_adj = np.zeros((new_nb_vert,K-1,2))        # create list of edges?
        free_ind = np.zeros(new_nb_vert,np.uint8)       # Number of registered neighbours, per vertex (so as not to exceed limit K)
        # print("adj shape: "+str(adj.shape))
        # print("free_ind shape: "+str(free_ind.shape))
        for i in range(new_nb_vert):
            adj[i,0] = i+1              # add vertex to its own neighbourhood   (indexed from 1)

        for f in range(faces.shape[0]):
            # v1 = new_vert_ind[faces[f,0]-1]
            # v2 = new_vert_ind[faces[f,1]-1]
            # v3 = new_vert_ind[faces[f,2]-1]
            v1 = faces[f,0]
            v2 = faces[f,1]
            v3 = faces[f,2]


            #print("v1 = "+str(v1)+", free_ind[v1] = "+str(free_ind[v1]))

            # For each of the 3 vertices, add opposite edge to temp_adj
            if(free_ind[v1]==(K-1)):
                unregistered_connections+=1
                #print("warning: vertex "+str(v1)+" already has "+str(K)+" neighbours")
            else:
                temp_adj[v1,free_ind[v1]]=np.array([v2+1,v3+1])
                free_ind[v1]+=1

            if(free_ind[v2]==(K-1)):
                unregistered_connections+=1
                #print("warning: vertex "+str(v2)+" already has "+str(K)+" neighbours")
            else:
                temp_adj[v2,free_ind[v2]] = np.array([v3+1,v1+1])
                free_ind[v2]+=1

            if(free_ind[v3]==(K-1)):
                unregistered_connections+=1
                #print("warning: vertex "+str(v3)+" already has "+str(K)+" neighbours")
            else:
                temp_adj[v3,free_ind[v3]] = np.array([v1+1,v2+1])
                free_ind[v3]+=1

        if unregistered_connections>0:
            print("unregistered connections (vertices): "+str(unregistered_connections/2))

        #For each vertex
        for v in range(temp_adj.shape[0]):
            # Add first edge
            first_ind = temp_adj[v,0,0]
            adj[v,1] = first_ind
            last_ind = temp_adj[v,0,1]
            adj[v,2] = last_ind
            free_ind=3


            while(free_ind<K):
                # Look for next edge, add 2nd vertex
                tuple_ind = np.where(temp_adj[v,:,0]==last_ind)
                res = tuple_ind[0]              #where returns a tuple (array,) in our case
                if (res.size==0):           # Not found?
                    break
                ind = res[0]
                last_ind = temp_adj[v,ind,1]
                if (last_ind==first_ind):       # Loop is complete
                    break
                adj[v,free_ind] = last_ind
                free_ind+=1

        #adj matrix should now be ordered correctly

    #Adj graph built!


    #Compute normals (per vertex)
    normals = computeNormals(new_vertices,faces)

    #free_ind returns connectivity for each vertex. Not necessary, for testing/information purposes only

    return new_vertices, adj, free_ind, faces, normals


def write_xyz(vec, strFileName):

    #outputFile = open(strFileName,"w")

    np.savetxt(strFileName, vec)

def write_mesh(vl,fl,strFileName):

    vnum = vl.shape[0]
    v_ch = vl.shape[1]
    vVec = np.full((vnum,1), 'v')

    vl = vl.flatten()

    vstr = ["%.6f" % number for number in vl]
    #print("vstr shape: "+str(vstr.shape))
    vstr = np.array(vstr)

    vstr = np.reshape(vstr, (vnum,v_ch))

    #vstr = vl.astype('|S8')
    vstr = np.concatenate((vVec,vstr),axis=1)

    #writeStringArray(vstr, strFileName, fl)


    outputFile = open(strFileName,"w")

    for row in range(vstr.shape[0]):
        for col in range(vstr.shape[1]):
            outputFile.write(vstr[row,col])
            outputFile.write(' ')
        outputFile.write('\n')

    # # write vertices
    # np.savetxt(outputFile,vstr, delimiter=",")

    #transform to one-indexed for obj format
    faces = fl
    faces = faces+1
    
    faces = faces.astype(str)
    for row in range(faces.shape[0]):
        if((faces[row,0]=='1')and(faces[row,1]=='1')):
            break
        outputFile.write('f ')
        for col in range(faces.shape[1]):
            outputFile.write(faces[row,col])
            outputFile.write(' ')
        outputFile.write('\n')


# Returns the one-sided Hausdorff distance from point set V0 to point set V1, normalized by diagonal length of the bounding box of U(V0,V1)
# V0: numpy array of shape (N0,3)
# V0: numpy array of shape (N1,3)
# Returns a pair of float: (max distance, mean distance)
def oneSidedHausdorff(V0,V1):
    
    #First, normalize
    xmin = min(np.amin(V0[:,0]),np.amin(V1[:,0]))
    ymin = min(np.amin(V0[:,1]),np.amin(V1[:,1]))
    zmin = min(np.amin(V0[:,2]),np.amin(V1[:,2]))
    xmax = max(np.amax(V0[:,0]),np.amax(V1[:,0]))
    ymax = max(np.amax(V0[:,1]),np.amax(V1[:,1]))
    zmax = max(np.amax(V0[:,2]),np.amax(V1[:,2]))

    diag = math.sqrt(math.pow(xmax-xmin,2)+math.pow(ymax-ymin,2)+math.pow(zmax-zmin,2))

    V0 = V0/diag
    V1 = V1/diag

    N0 = V0.shape[0]
    N1 = V1.shape[0]

    # Compute distance between every pair of points

    # Solution 1: most efficient, but too greedy in memory... :(
            # bV0 = np.reshape(V0, (N0,1,3))
            # bV1 = np.reshape(V1, (1,N1,3))
            # bV0 = np.tile(bV0, (1,N1,1))
            # bV1 = np.tile(bV1, (N0,1,1))
            # diff_mat = bV1-bV0
            # dist_mat = np.linalg.norm(diff_mat,axis=2)
            # # Take max of every min distance for each point of V0
            # print("dist_mat shape: "+str(dist_mat.shape))

            # dist_vec = np.amin(dist_mat,axis=1)
            # print("dist_vec shape: "+str(dist_vec.shape))

            # dist = np.amax(dist_vec)

    # Second solution: point by point
    distM = 0
    distAvg=0
    for v in range(N0):

        bV0 = np.reshape(V0[v,:], (1,3))
        bV0 = np.tile(bV0, (N1,1))

        diff_vec = V1-bV0
        dist_vec = np.linalg.norm(diff_vec,axis=1)
        vdist = np.amin(dist_vec)

        distAvg+=vdist
        # Update general dist if this point's dist is higher than the current max
        if vdist>distM:
            distM=vdist

    distAvg/=N0
    return distM, distAvg


# Takes two sets of face normals with one-one correspondance
# Returns a pair of floats: the average angular difference (in degrees) between pairs of normals, and the std

# Now, ignore cases when n1 is equal to zero (in our case, fake nodes, n1 is normally GT)
def angularDiff(n0,n1):

    faceNum = n0.shape[0]

    fakenodes = np.less_equal(np.absolute(n1),10e-4)
    fakenodes = np.all(fakenodes,axis=-1)
    
    n0 = normalize(n0)
    n1 = normalize(n1)

    # print("n0 example: "+str(n0[0,:]))
    # print("n1 example: "+str(n1[0,:]))

    dotP = np.sum(np.multiply(n0,n1),axis=1)
    #print("dotP example: "+str(dotP[0]))

    # print("min dotP = "+str(np.amin(dotP)))
    # print("max dotP = "+str(np.amax(dotP)))

    angDiff = np.arccos(0.999999*dotP)
    angDiff = angDiff*180/math.pi

    zeroVec = np.zeros_like(angDiff, dtype=np.int32)
    oneVec = np.ones_like(angDiff, dtype=np.int32)
    realnodes = np.where(fakenodes,zeroVec,oneVec)

    realIndices = np.where(fakenodes,zeroVec,np.arange(faceNum, dtype=np.int32))

    # print("angDiff shape: "+str(angDiff.shape))
    # print("sum realnodes: "+str(np.sum(realnodes)))
    # angDiffTest = np.where(fakenodes,zeroVec,angDiff)
    # angDiffTest = np.sum(angDiffTest)/np.sum(realnodes)

    angDiff = np.extract(fakenodes==False, angDiff)
    #angDiff = angDiff[realIndices]

    print("angDiff shape: "+str(angDiff.shape))
    
    # #Set loss to zero for fake nodes
    
    # print("angDiffTest = "+str(angDiffTest))
    # print("mean angDiff = "+str(np.mean(angDiff)))
    # print("angDiff example: "+str(angDiff[0]))
    return np.mean(angDiff), np.std(angDiff)

# Takes a mesh as input (vertices list vl, faces list fl), and returns a list of faces areas (faces are assumed to be triangular)
def getTrianglesArea(vl,fl):

    fnum = fl.shape[0]
    triArea = np.empty([fnum])

    for f in range(fnum):
        v0 = vl[fl[f,0],:]
        v1 = vl[fl[f,1],:]
        v2 = vl[fl[f,2],:]
        e1 = v1-v0
        e2 = v2-v0
        cp = np.cross(e1,e2)
        triArea[f] = 0.5 * np.linalg.norm(cp)

    return triArea

def getTrianglesBarycenter(vl,fl):

    fnum = fl.shape[0]
    triCenter = np.empty([fnum,3])

    # Normalize
    #First, normalize by diagonal of bounding box
    xmin = np.amin(vl[:,0])
    ymin = np.amin(vl[:,1])
    zmin = np.amin(vl[:,2])
    xmax = np.amax(vl[:,0])
    ymax = np.amax(vl[:,1])
    zmax = np.amax(vl[:,2])

    diag = math.sqrt(math.pow(xmax-xmin,2)+math.pow(ymax-ymin,2)+math.pow(zmax-zmin,2))

    vl = vl/diag

    for f in range(fnum):
        v0 = vl[fl[f,0],:]
        v1 = vl[fl[f,1],:]
        v2 = vl[fl[f,2],:]

        triCenter[f,:] = (v0+v1+v2)/3

        # if (f==0):
        #     print("v0 = ("+str(v0)+")")
        #     print("v1 = ("+str(v1)+")")
        #     print("v2 = ("+str(v2)+")")
        #     print("barycenter = ("+str(triCenter[f,:])+")")
    return triCenter


# This function takes a mesh as input, and returns a patch of given size, by growing around the input seed
def getMeshPatch(vIn,fIn,fAdjIn,faceNum,seed):

    K = fAdjIn.shape[1]
    fOut = np.empty((faceNum+K,3),dtype=int)
    fAdjOut = np.zeros((faceNum+K,K),dtype=int)
    vOut = np.empty((int(faceNum*0.6)+K,3),dtype=np.float32)  # Arbitrary vertex size

    vNewInd = np.zeros(vIn.shape[0],dtype=int)     # Array of correspondence between old vertex indices and new ones
    vNewInd -= 1
    vOldInd = np.zeros(vIn.shape[0],dtype=int)     # Array of correspondence between new vertex indices and old ones
    vOldInd -= 1

    fNewInd = np.zeros(fIn.shape[0],dtype=int)
    fNewInd -= 1
    fOldInd = np.zeros(fIn.shape[0],dtype=int)
    fOldInd -= 1


    vIt = [0]   # Using lists as a dirty trick for namespace reasons for inner functions
    fIt = [0]
    
    def addVertex(vind):
        if vNewInd[vind]==-1:
            vNewInd[vind]=vIt[0]
            vOldInd[vIt[0]] = vind
            vOut[vIt[0],:] = vIn[vind,:]
            vIt[0]+=1



    def addFace(find):
        vl0 = fIn[find,0]
        vl1 = fIn[find,1]
        vl2 = fIn[find,2]

        addVertex(vl0)
        addVertex(vl1)
        addVertex(vl2)

        fOut[fIt[0],0] = vNewInd[vl0]
        fOut[fIt[0],1] = vNewInd[vl1]
        fOut[fIt[0],2] = vNewInd[vl2]
        fNewInd[find] = fIt[0]
        fOldInd[fIt[0]] = find
        fIt[0] += 1

    fQueue = queue.Queue()         # Queue of faces to be added
    fQueue.put(seed) # Add seed to start the process
    #print("fQueue: "+str(fQueue.empty()))

    addFace(seed)

    while (fIt[0]<faceNum):    # Keep growing until we reach desired count
        #print("wuuuut")
        if fQueue.empty():
            break


        curF = fQueue.get()         # Get current face index
        #print("curF = "+str(curF))
        newFInd = fNewInd[curF]     # Get its new index
        # current face should already be added


        fAdjOut[newFInd,0] = newFInd+1  #Add itself first. Adj array is one-indexed!!


        #print("fAdj line: "+str(fAdjIn[curF,:]))
        # Process neighbours
        for neigh in range(1,K):    # Skip first entry, its the current face
            
            curN = fAdjIn[curF,neigh]-1

            if curN==-1:    # We've reached the last neighbour
                break

            if(fNewInd[curN]==-1):  # If face not processed, add it to faces list and add it to queue
                addFace(curN)
                fQueue.put(curN)

            fAdjOut[newFInd,neigh] = fNewInd[curN]+1    #fill new adj graph

    # We've reached the count.
    # Now, fill adjacency graph for remaining faces in the queue

    while not fQueue.empty():
        curF = fQueue.get()         # Get current face index
        
        newFInd = fNewInd[curF]     # Get its new index

        fAdjOut[newFInd,0] = newFInd+1  #Add itself first. Adj array is one-indexed!!

        neighCount = 1
        # Process neighbours
        for neigh in range(1,K):    # Skip first entry, its the current face
            
            curN = fAdjIn[curF,neigh]-1

            if curN==-1:    # We've reached the last neighbour
                break

            if(fNewInd[curN]==-1):  # If face not in the graph, skip it
                continue

            fAdjOut[newFInd,neighCount] = fNewInd[curN]+1    #fill new adj graph
            neighCount += 1


    vOut = vOut[:vIt[0],:]
    fOut = fOut[:fIt[0],:]
    fAdjOut = fAdjOut[:fIt[0],:]
    vOldInd = vOldInd[:vIt[0]]
    fOldInd = fOldInd[:fIt[0]]

    return vOut, fOut, fAdjOut, vOldInd , fOldInd    # return vOldInd (and fOldInd) as well, so we can run this once for GT/noisy mesh


# Convert an adjacency matrix from Nitika style to scipy sparse matrix
def listToSparse(Adj, nodes_pos):

    N = Adj.shape[0]
    K = Adj.shape[1]

    row_ind = np.zeros(N*K,dtype = np.int32)
    col_ind = np.zeros(N*K,dtype = np.int32)
    values = np.zeros(N*K,dtype = np.float32)
    cur_ind=0

    for n in range(N):  # For each node
        for neigh in range(1,K):    # Skip first neighbour, it encodes the current node
            nnode = Adj[n,neigh]-1  # It is one-indexed
            if nnode < 0:
                break

            row_ind[cur_ind] = n
            col_ind[cur_ind] = nnode
            n_pos = nodes_pos[n,:]
            nnode_pos = nodes_pos[nnode,:]
            values[cur_ind] = 1/(1000*np.linalg.norm(nnode_pos-n_pos))
            #values[cur_ind] = np.linalg.norm(nnode_pos-n_pos)
            cur_ind+=1

    row_ind = row_ind[:cur_ind]
    col_ind = col_ind[:cur_ind]
    values = values[:cur_ind]
    #values = np.ones(cur_ind,dtype = np.int8)

    coo = scipy.sparse.coo_matrix((values,(row_ind,col_ind)),shape=(N,N))

    return coo

# Convert an adjacency matrix from sparse matrix to Nitika list style
def sparseToList(Adj, K):

    N = Adj.shape[0]

    #listAdj = np.zeros((N,K),dtype = np.int32)

    # Initialize Adj mat with 1st row set to 1
    initAdj0 = np.zeros((N,K-1),dtype = np.int32)
    initAdj1 = np.arange(N)+1
    initAdj1 = np.expand_dims(initAdj1, axis=-1)
    #print("0 shape: "+str(initAdj0.shape))
    #print("1 shape: "+str(initAdj1.shape))
    listAdj = np.concatenate((initAdj1,initAdj0),axis = 1)


    curNeigh = np.ones(N, dtype=np.int8)   # For each node, keep count of neighbours

    cx = Adj.tocoo()    
    for i,j,_ in zip(cx.row, cx.col, cx.data):
        if(i!=j):
            if(curNeigh[i]==K):
                print("Warning: saturated node! ("+str(i)+","+str(j)+")")
            else:
                listAdj[i,curNeigh[i]] = j+1
                curNeigh[i]+=1

    return listAdj

def inv_perm(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


# return max curvature, min curvature, and average. Used to generate clusters for classification
def computeCurvature(fpos, fn, adj):
    
    #keep neighbours only
    adj_n = adj[:,1:]
    adj_n = adj_n-1

    neighbours_normals = fn[adj_n]

    neighbours_pos = fpos[adj_n]

    # print(" fn 0: "+str(fn[0,:]))

    # print(" adj 0: "+str(adj[0,:]))

    # print(" adj_n 0: "+str(adj_n[0,:]))

    K = adj_n.shape[1]

    fn_n = np.tile(np.expand_dims(fn,axis=1), [1,K,1])
    # [N, K, 3]
    fpos_n = np.tile(np.expand_dims(fpos,axis=1), [1,K,1])


    fvec = np.subtract(neighbours_pos, fpos_n)


    # print("fn_n 0: "+str(fn_n[0,:,:]))
    # print("n_n 0: "+str(neighbours_normals[0,:,:]))

    #dotP = np.sum(np.multiply(fn_n,neighbours_normals),axis=2)

    dotP = np.sum(np.multiply(fn_n,fvec),axis=2)

    # print("dp 0: "+str(dotP[0,:]))
    non_zeros = np.not_equal(adj_n, np.zeros_like(adj_n)-1)

    dotP = np.where(non_zeros,dotP,np.zeros_like(dotP))

    dotPWeight = np.where(non_zeros,np.ones_like(dotP),np.zeros_like(dotP))



    # print("dp 0: "+str(dotP[0,:]))
    # [N, K]

    curv_min = np.amin(dotP,axis=1,keepdims=True)
    curv_max = np.amax(dotP,axis=1,keepdims=True)
    curv_mean = np.sum(dotP,axis=1,keepdims=True)/np.sum(dotPWeight,axis=1,keepdims=True)


    curv_stat = np.concatenate((curv_min,curv_max,curv_mean),axis=1)

    # print("curv_stat 0: "+str(curv_stat[0,:,]))
    return curv_stat


def customKMeans(points, k, iternum=500):

    # Initialize random centroids
    centroids = points.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]

    for i in range(iternum):
        closest = closest_centroid(points, centroids)
        move_centroids(points, closest, centroids)

    return centroids, closest


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    centroid_ind = np.argmin(distances, axis=0)
    return centroid_ind.astype(int)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

