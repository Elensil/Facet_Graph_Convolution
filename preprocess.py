from settings import *
from trainingSet import *
import pickle
import os


def pickleData(withVerts=False):

    coarseningStepNum = COARSENING_STEPS
    coarseningLvlNum = COARSENING_LVLS
    maxSize = MAX_PATCH_SIZE
    binDumpPath = BINARY_DUMP_PATH

    myTS = TrainingSet(maxSize, coarseningStepNum, coarseningLvlNum)
    myValidSet = TrainingSet(maxSize,coarseningStepNum, coarseningLvlNum)
    if withVerts:
        tsPickleName = 'trainingSetWithVertices.pkl'
        vsPickelName = 'validSetWithVertices.pkl'
    else:
        tsPickleName = 'trainingSet.pkl'
        vsPickleName = 'validSet.pkl'
    # Training set
    for filename in os.listdir(TRAINING_DATA_PATH):
        if (filename.endswith(".obj")):
            print("Adding %s (%i)"%(filename, myTS.mesh_count))
            gtfilename = getGTFilename(filename)
            for it in range(TRAINING_DATA_REDUNDANCY):
                if withVerts:
                    myTS.addMeshWithVerticesAndGT(TRAINING_DATA_PATH, filename, GT_DATA_PATH, gtfilename)
                else:
                    myTS.addMeshWithGT(TRAINING_DATA_PATH,filename,GT_DATA_PATH,gtfilename)
        
    with open(binDumpPath+tsPickleName,'wb') as fp:
        pickle.dump(myTS,fp)
    
    # Validation set
    for filename in os.listdir(VALID_DATA_PATH):
        if (filename.endswith(".obj")):
            gtfilename = getGTFilename(filename)
            if withVerts:
                myValidSet.addMeshWithVerticesAndGT(VALID_DATA_PATH, filename, GT_DATA_PATH, gtfilename)
            else:
                myValidSet.addMeshWithGT(VALID_DATA_PATH,filename,GT_DATA_PATH,gtfilename)

    with open(binDumpPath+vsPickleName,'wb') as fp:
        pickle.dump(myValidSet,fp)

if __name__ == "__main__":

    if not os.path.exists(BINARY_DUMP_PATH):
        os.makedirs(BINARY_DUMP_PATH)
        
    pickleData(INCLUDE_VERTICES)

    print("Preprocessing complete. Dump files saved to "+ BINARY_DUMP_PATH)