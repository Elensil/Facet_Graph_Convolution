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
from train import *

TF_VERSION = int(tf.__version__[0])
if TF_VERSION==2:
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf



def infer(withVerts=False):
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    maxSize = MAX_PATCH_SIZE
    patchSize = MAX_PATCH_SIZE

    noisyFolder = VALID_DATA_PATH

    # Get GT mesh
    for noisyFile in os.listdir(noisyFolder):

        if (not noisyFile.endswith(".obj")):
            continue
        print("processing noisy file: "+noisyFile)

        denoizedFile = noisyFile[:-4]+"_denoised_gray.obj"

        noisyFilesList = [noisyFile]
        denoizedFilesList = [denoizedFile]

        for fileNum in range(len(denoizedFilesList)):
            denoizedFile = denoizedFilesList[fileNum]
            noisyFile = noisyFilesList[fileNum]

            noisyFileWColor = noisyFile[:-4]+"_original_normals.obj"
            denoizedFileWColor = noisyFile[:-4]+"_denoised_color.obj"

            if withVerts:
                noisyFileWInferredColor0 = noisyFile[:-4]+"_fine_normals_s.obj"
                noisyFileWInferredColor1 = noisyFile[:-4]+"_mid_normals_s.obj"
                noisyFileWInferredColor2 = noisyFile[:-4]+"_coarse_normals_s.obj"
                noisyFileWInferredColor3 = noisyFile[:-4]+"_coarse_normals2_s.obj"
                noisyFileWInferredColor4 = noisyFile[:-4]+"_coarse_normals3_s.obj"
                faceMeshFile = noisyFile[:-4]+"_face_mesh.obj"
                faceMeshFile1 = noisyFile[:-4]+"_face_mesh1.obj"
                faceMeshFile2 = noisyFile[:-4]+"_face_mesh2.obj"
            else:
                noisyFileWInferredColor0 = noisyFile[:-4]+"_inferred_normals.obj"

            if os.path.isfile(RESULTS_PATH+denoizedFile):
                    if B_OVERWRITE_RESULT:
                        print("Warning: %s will be overwritten. (To deactivate overwriting, change parameter in settings.py)"%denoizedFile)
                    else:
                        print("Skipping %s. File already exists. (For automatic overwriting, change parameter in settings.py)"%denoizedFile)
                        continue

            print("Adding mesh "+noisyFile+"...")
            t0 = time.time()
            inputMesh = InferenceMesh(maxSize, coarseningStepNum, coarseningLvlNum)
            if withVerts:
                inputMesh.addMeshWithVertices(noisyFolder, noisyFile)
            else:
                inputMesh.addMesh(noisyFolder, noisyFile)

            print("mesh added ("+str(1000*(time.time()-t0))+"ms)")

            faces = inputMesh.faces

            print("Inference ...")
            t0 = time.time()
            if withVerts:
                upV0, upV0mid, upV0coarse, upN0, upN1, upN2, upP0, upP1, upP2 = inferNet(inputMesh)
            else:
                upV0, upN0 = inferNetOld(myTS)

            print("Inference complete ("+str(1000*(time.time()-t0))+"ms)")

            write_mesh(upV0, faces, RESULTS_PATH+denoizedFile)
            if withVerts:
                write_mesh(upV0mid, faces, RESULTS_PATH+noisyFile[:-4]+"_d_mid.obj")
                write_mesh(upV0coarse, faces, RESULTS_PATH+noisyFile[:-4]+"_d_coarse.obj")

            V0 = inputMesh.vertices
            faces_noisy = inputMesh.faces
            f_normals0 = inputMesh.normals
            angColor0 = (upN0+1)/2
            angColorNoisy = (f_normals0+1)/2

            newVn0, newFn0 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor0)
            write_mesh(newVn0, newFn0, RESULTS_PATH+noisyFileWInferredColor0)
            
            newVnoisy, newFnoisy = getColoredMesh(np.squeeze(V0), faces_noisy, angColorNoisy)
            write_mesh(newVnoisy, newFnoisy, RESULTS_PATH+noisyFileWColor)

            if withVerts:
                angColor1 = (upN1+1)/2
                angColor2 = (upN2+1)/2
                newVn1, newFn1 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor1)
                newVn2, newFn2 = getColoredMesh(np.squeeze(V0), faces_noisy, angColor2)
                write_mesh(newVn1, newFn1, RESULTS_PATH+noisyFileWInferredColor1)
                write_mesh(newVn2, newFn2, RESULTS_PATH+noisyFileWInferredColor2)
    

if __name__ == "__main__":

	infer(withVerts=INCLUDE_VERTICES)
	print("Inference complete. Results saved to "+ RESULTS_PATH)