import random
import math
import sys
import numpy as np
import os
from utils import *
import argparse


def getAverageEdgeLength(V,faces):

    edges_num=0
    edges_length=0

    fnum = faces.shape[0]

    for f in range(fnum):

        v0 = V[faces[f,0],:]
        v1 = V[faces[f,1],:]
        v2 = V[faces[f,2],:]

        e0 = v1-v0
        e1 = v2-v0
        e2 = v2-v1

        el = np.linalg.norm(e0)+np.linalg.norm(e1)+np.linalg.norm(e2)
        edges_length+= el
        edges_num+=3

    # edges will be counted twice, but it won't change the average value
    
    return edges_length/edges_num




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infolder', type=str, default='')
    parser.add_argument('--outfolder', type=str, default='')
    
    FLAGS = parser.parse_args()

    inputfolder = FLAGS.infolder
    destfolder = FLAGS.outfolder


    # get list of all files in directory
    file_list = sorted(os.listdir(inputfolder))

    # get a list of files ending in 'obj'
    obj_list = [item for item in file_list if item.endswith('.obj')]

    K = 0  #Kind of useless here ?

    for item in obj_list:


        # Import obj from input dir
        path_to_file = os.path.join(inputfolder, item)

        gt_name = item[:-4]+'_gt'
        noisy_name = item[:-4]+'_noisy'
        
        # Load mesh
        V,_,_, faces, normals = load_mesh(inputfolder, item, K, False)

        el = getAverageEdgeLength(V,faces)


        # --- Displace vertices along normal ---
        # Compute random value per point
        d1 = 0.1 * el * np.random.normal(0,1,(len(normals),1))
        d2 = 0.2 * el * np.random.normal(0,1,(len(normals),1))
        d3 = 0.3 * el * np.random.normal(0,1,(len(normals),1))

        # Compute corresponding displacement along normal
        d1 = np.tile(d1, (1,3))
        d2 = np.tile(d2, (1,3))
        d3 = np.tile(d3, (1,3))
        d1 = np.multiply(d1, normals)
        d2 = np.multiply(d2, normals)
        d3 = np.multiply(d3, normals)

        # Add it to position
        V1 = np.add(V, d1)
        V2 = np.add(V, d2)
        V3 = np.add(V, d3)

        write_mesh(V1,faces,destfolder + noisy_name + "_1.obj")
        write_mesh(V2,faces,destfolder + noisy_name + "_2.obj")
        write_mesh(V3,faces,destfolder + noisy_name + "_3.obj")

    