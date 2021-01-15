from model import *
from utils import *
from settings import *
from dataClasses import *
import os


# Compute metrics and heatmaps on denoised meshes + GT
def computeMetrics():

    # Take the opportunity to generate array of metrics on reconstructions
    nameArray = []      # String array, to now which row is what
    resultsArray = []   # results array, following the pattern in the xlsx file given by author of Cascaded Normal Regression.
                        # [Max distance, Mean distance, Mean angle, std angle, face num]

    gtFolder = "/morpheo-nas2/marmando/DeepMeshRefinement/real_paper_dataset/Synthetic/test/original/"

    # results file name
    csv_filename = RESULTS_PATH+"results_heat.csv"
    
    angDict={}

    # Get GT mesh
    for gtFileName in os.listdir(gtFolder):

        nameArray = []
        resultsArray = []

        denoizedFile0 = gtFileName[:-4]+"_n1_denoised_gray.obj"
        denoizedFile1 = gtFileName[:-4]+"_n2_denoised_gray.obj"
        denoizedFile2 = gtFileName[:-4]+"_n3_denoised_gray.obj"

        heatFile0 = gtFileName[:-4]+"_heatmap_1.obj"
        heatFile1 = gtFileName[:-4]+"_heatmap_2.obj"
        heatFile2 = gtFileName[:-4]+"_heatmap_3.obj"

        # Load GT mesh
        GT,_,_,faces_gt,_ = load_mesh(gtFolder, gtFileName, 0, False)
        GTf_normals = computeFacesNormals(GT, faces_gt)
        denseGT = getDensePC(GT, faces_gt, res=1)
        facesNum = faces_gt.shape[0]
        # We only need to load faces once. Connectivity doesn't change for noisy meshes
        # Same for adjacency matrix

        f_adj = getFacesLargeAdj(faces_gt,K_faces)

        faces_gt = np.array(faces_gt).astype(np.int32)
        faces = np.expand_dims(faces_gt,axis=0)

        denoizedFilesList = [denoizedFile0]
        heatMapFilesList = [heatFile0]

        denoizedFilesList = [denoizedFile0,denoizedFile1,denoizedFile2]
        heatMapFilesList = [heatFile0,heatFile1,heatFile2]

        for fileNum in range(len(denoizedFilesList)):
            
            denoizedFile = denoizedFilesList[fileNum]
            heatFile = heatMapFilesList[fileNum]
            
            if not os.path.isfile(RESULTS_PATH+heatFile):
                
                V0,_,_, _, _ = load_mesh(RESULTS_PATH, denoizedFile, 0, False)
                f_normals0 = computeFacesNormals(V0, faces_gt)

                print("computing Hausdorff "+ denoizedFile + " " + str(fileNum+1)+"...")
                # haus_dist0, avg_dist0 = oneSidedHausdorff(V0, GT)

                haus_dist0, _, avg_dist0, _ = hausdorffOverSampled(V0, GT, V0, denseGT, accuracyOnly=True)

                angDistVec = angularDiffVec(f_normals0, GTf_normals)

                borderF = getBorderFaces(faces_gt)

                angDistIn = angDistVec[borderF==0]
                angDistOut = angDistVec[borderF==1]

                angDistIn0 = np.mean(angDistIn)
                angStdIn0 = np.std(angDistIn)
                angDistOut0 = np.mean(angDistOut)
                angStdOut0 = np.std(angDistOut)
                angDist0 = np.mean(angDistVec)
                angStd0 = np.std(angDistVec)

                angDistSquare = np.mean(np.square(angDistVec))
                angDistVecRad = angDistVec*math.pi/180
                angDistSquareRad = np.mean(np.square(angDistVecRad))
                print("angDistSquare " + denoizedFile + " = %f"%angDistSquare)
                print("angDistSquareRad " + denoizedFile + " = %f"%angDistSquareRad)
                print("sqrt angDistSquare " + denoizedFile + " = %f"%np.sqrt(angDistSquare))

                #print("ang dist, std = (%f, %f)"%(angDist0, angStd0))

                angDist0, angStd0 = angularDiff(f_normals0, GTf_normals)
                print("max angle: "+str(np.amax(angDistVec)))
                dictLabel = denoizedFile[:-4]
                dictLabel = dictLabel.replace('-','_')
                angDict[dictLabel] = angDistVec
                # --- Test heatmap ---
                angColor = angDistVec / HEATMAP_MAX_ANGLE
                angColor = 1 - angColor
                angColor = np.maximum(angColor, np.zeros_like(angColor))

                colormap = getHeatMapColor(1-angColor)
                newV, newF = getColoredMesh(V0, faces_gt, colormap)

                # newV, newF = getHeatMapMesh(V0, faces_gt, angColor)

                write_mesh(newV, newF, RESULTS_PATH+heatFile)
                
                # Fill arrays
                nameArray.append(denoizedFile)
                resultsArray.append([haus_dist0, avg_dist0, angDist0, angStd0, facesNum, angDistIn0, angStdIn0, angDistOut0, angStdOut0])

        if not nameArray:
            continue
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

        scipy.io.savemat(RESULTS_PATH+"angDiffFinal.mat",mdict=angDict)


if __name__ == "__main__":
	computeMetrics()
