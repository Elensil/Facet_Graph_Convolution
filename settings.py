TRAINING_DATA_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train2/"
VALID_DATA_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
BINARY_DUMP_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/CycleConvExp/BinDump/Synthetic/OrderedOneLevel/"

COARSENING_STEPS = 2
K_faces = 23







def getGTFilename(filename):
    # For DTU
    fileNumStr = filename[4:7]
    gtfilename = 'stl'+fileNumStr+'_total.obj'
    return gtfilename