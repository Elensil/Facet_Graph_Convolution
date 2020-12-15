

# --- Paths Parameters ---

# TRAINING_DATA_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/train2/"
# VALID_DATA_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/DTU/Data/noisy/furu/valid/"
# BINARY_DUMP_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/CycleConvExp/BinDump/Synthetic/OrderedOneLevel/"
# BINARY_DUMP_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/CycleConvExp/BinDump/Synthetic/"

BASE_PATH = "/morpheo-nas2/marmando/DeepMeshRefinement/"

# -- Data --
DATA_PATH = BASE_PATH + "real_paper_dataset/"
TRAINING_DATA_PATH = DATA_PATH + "Synthetic/train/noisy/"          # Folder where noisy (training) meshes are stored. Used when preprocessing and pickling data
VALID_DATA_PATH = DATA_PATH + "Synthetic/train/valid/"             # Folder where noisy (valid) meshes are stored. Used when preprocessing and pickling data
GT_DATA_PATH = DATA_PATH + "Synthetic/train/original/"             # Folder where ground truth meshes are stored. Used when preprocessing and pickling data


BINARY_DUMP_PATH = BASE_PATH + "TEMP_Cleaning/bindump/"
BINARY_DUMP_PATH = BASE_PATH + "TEMP_Cleaning/binDumpVertices/"
NETWORK_PATH = BASE_PATH + "TEMP_Cleaning/netVerts/"
RESULTS_PATH = BASE_PATH + "TEMP_Cleaning/ResultsVerts/"

# --- Data Parameters ---

MAX_PATCH_SIZE = 20000              # When loading a mesh (for training or inference), il is split up into smaller patches if the number of faces is greater than MAX_PATCH_SIZE. Lower this in case of memory issues.
MIN_PATCH_SIZE = 2000               # Patches are grown to be at least this size, in order to guarantee a minimum receptive field for inference.
K_faces = 23                        # Maximum number of neighbours that can be taken into account in face graphs. A higher value will used up more memory. With a smaller K, some edges might be ignored.
TRAINING_DATA_REDUNDANCY = 1        # When preprocessing training data, add each mesh TRAINING_DATA_REDUNDANCY times. Increasing this number enables some form of data augmentation (there is randomness in the patch cut and the coarsening).


# --- Network Parameters ---

SAVEITER = 500                     # During training, a checkpoint is saved every SAVEITER iterations.

COARSENING_STEPS = 2                # Number of coarsening iterations performed in each pooling layer. Default value (used in the paper) is 2
COARSENING_LVLS = 3                 # Number of "resolution" levels of graph used in the network. Default is 3. WARNING: This will only automate the data preprocessing (graph coarsening is precomputed). The network architecture is hard-coded!


# BOOLEAN PARAMETERS
B_OVERWRITE_RESULT = True

# --- Visualization ---
HEATMAP_MAX_ANGLE = 30.0            # Sets the scale for heatmaps of angular error. This value is set to red (highest error). Values above are truncated.



# This function returns the filename of the ground truth mesh, given the name of the corresponding noisy mesh.
# Adapt to your data. 
def getGTFilename(filename):
    # For DTU
    fileNumStr = filename[4:7]
    gtfilename = 'stl'+fileNumStr+'_total.obj'
    
    gtnameoffset = 7
    gtfilename = filename[:-gtnameoffset]+".obj"


    return gtfilename
