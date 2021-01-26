# --- Paths Parameters ---
BASE_PATH = "CHANGE THIS PATH/"

# -- Data --
DATA_PATH = BASE_PATH + "Data/"
TRAINING_DATA_PATH = DATA_PATH + "Synthetic/train/noisy/"           # Folder where noisy (training) meshes are stored. Used when preprocessing and pickling data
VALID_DATA_PATH = DATA_PATH + "Synthetic/train/valid/"              # Folder where noisy (validation) meshes are stored. Used when preprocessing and pickling data
TEST_DATA_PATH = DATA_PATH + "DemoData/"                            # Folder where noisy (test) meshes are stored. Used by default for inference
GT_DATA_PATH = DATA_PATH + "Synthetic/train/original/"              # Folder where ground truth meshes are stored. Used when preprocessing and pickling data
TEST_GT_DATA_PATH = DATA_PATH + "Synthetic/test/original/"          # Folder where test ground truth meshes are stored.

BINARY_DUMP_PATH = BASE_PATH + "Preprocessed_Data/"

NETWORK_PATH = BASE_PATH + "Networks/Default/"                      # Model parameters are saved to/restored from this folder. Can be overriden with --network_path
RESULTS_PATH = BASE_PATH + "Results/Default/"                       # Folder where results are saved. Can be overriden with --results_path


# --- Data Parameters ---

MAX_PATCH_SIZE = 20000              # When loading a mesh (for training or inference), it is split up into smaller patches if the number of faces is greater than MAX_PATCH_SIZE. Lower this in case of memory issues.
                                    # Increase it for more efficient training and faster preprocessing
MIN_PATCH_SIZE = 2000               # Patches are grown to be at least this size, in order to guarantee a minimum receptive field for inference.
K_faces = 23                        # Maximum number of neighbours that can be taken into account in face graphs. A higher value will used up more memory. With a smaller K, some edges might be ignored.
TRAINING_DATA_REDUNDANCY = 1        # When preprocessing training data, add each mesh TRAINING_DATA_REDUNDANCY times. Increasing this number enables some form of data augmentation (there is randomness in the patch cut and the coarsening).


# --- Network Parameters ---

INCLUDE_VERTICES = False            # False by default. Set to True to use both extensions presented in the paper. See main README.md for more details
SAVEITER = 5000                     # During training, a checkpoint is saved every SAVEITER iterations.
COARSENING_STEPS = 2                # Number of coarsening iterations performed in each pooling layer. Default value (used in the paper) is 2
COARSENING_LVLS = 3                 # Number of "resolution" levels of graph used in the network. Default is 3. WARNING: This will only automate the data preprocessing (graph coarsening is precomputed). The network architecture is hard-coded!
NUM_ITERATIONS = 300000             # Number of training iterations run by the network. can be overriden by command line argument --num_iterations

# BOOLEAN PARAMETERS
B_OVERWRITE_RESULT = False           # For inference: If False, meshes present in the output directory will not be processed again

# --- Visualization ---
HEATMAP_MAX_ANGLE = 30.0            # Sets the scale for heatmaps of angular error. This value is set to red (highest error). Values above are truncated.
                                    # (Unused for the moment)

# This function returns the filename of the ground truth mesh, given the name of the corresponding noisy mesh.
# Adapt to your data. 
def getGTFilename(filename):
    gtnameoffset = 7
    gtfilename = filename[:-gtnameoffset]+".obj"
    return gtfilename

def getGTFilenameFromDenoised(filename):
    gtnameoffset = 21
    gtfilename = filename[:-gtnameoffset]+".obj"
    return gtfilename
