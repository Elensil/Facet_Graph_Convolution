import time
from utils import *
from settings import *
from lib.coarsening import *

class MeshPair():

    K_faces = 23

    def __init__(self, inputFilePath, filename, gtFilePath, gtFileName):

        self.V,_,_, self.F, _ = load_mesh(inputFilePath, filename, 0, False)
        self.N = computeFacesNormals(self.V, self.F)

        self.gtV,_,_,_,_ = load_mesh(gtFilePath, gtFileName, 0, False)
        self.gtN = computeFacesNormals(self.gtV, self.F)

        self.P = getTrianglesBarycenter(self.V, self.F)

        self.adj = getFacesLargeAdj(self.F, K_faces)

        
# class Patch():

#     def __init__(self, mesh, patchSize, faceSeed, mask):

#         self.adj, self.faceInd, slef.nextSeed = getGraphPatch_wMask(mesh.adj, patchSize, faceSeed, mask)   

#         self.N = mesh.N[self.faceInd]
#         self.P = mesh.P[self.faceInd]
#         self.gtN = mesh.gtN[self.faceInd]
#         self.size = self.faceInd.shape[0]
#         self.K = mesh.adj.shape[1]

#     def coarsen(self, coarseningLvlNum, coarseningStepNum):
#         # Convert to sparse matrix and coarsen graph
#         coo_adj = listToSparseWNormals(self.adj, self.P, self.N)
#         has_sat = True

#         while has_sat:
#             print("Coarsening...")
#             tc0 = time.clock()
#             adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)
#             tc1 = time.clock()
#             print("Coarsening complete ("+str(1000*(tc1-tc0))+"ms)")
#             has_sat = False
#             # Change adj format
#             fAdjs = []
#             for lvl in range(coarseningLvlNum):
#                 tsl0 = time.clock()
#                 fadj, has_sat_temp = sparseToList(adjs[coarseningStepNum*lvl],self.K)
#                 print("sparse to list conversion ("+str(1000*(time.clock()-tsl0))+"ms)")
#                 fadj = np.expand_dims(fadj, axis=0)
#                 fAdjs.append(fadj)
#                 has_sat = has_sat or has_sat_temp
#         self.adjPyramid = fAdjs[]
#         self.perm = newToOld

#         self.numFakeNodes = len(newToOld) - self.size
                    
#                     padding6 =np.zeros((self.numFakeNodes,patchFNormals.shape[1]))
#         padding3 =np.zeros((self.numFakeNodes,3))
#         self.N = np.concatenate((self.N, padding3),axis=0)
#         self.P = np.concatenate((self.P, padding3),axis=0)
#         self.gtN = np.concatenate((self.gtN, padding3),axis=0)
#         # Reorder nodes
#         self.N = self.N[newToOld]
#         self.P = self.P[newToOld]
#         self.gtN = self.gtN[newToOld]

                    

# class InferenceMesh(PreprocessedData):


class myGenData():
    def __init__(self, maxSize, coarseningStepNum, coarseningLvlNum):
        self.in_list = []
        self.gt_list = []
        self.adj_list = []
        self.mesh_count = 0
        self.num_faces = []
        self.patch_indices = []
        self.permutations = []
        self.maxSize = maxSize
        self.patchSize = maxSize
        self.coarseningStepNum = coarseningStepNum
        self.coarseningLvlNum = coarseningLvlNum

class PreprocessedData():
    def __init__(self, maxSize, coarseningStepNum, coarseningLvlNum):
        self.in_list = []
        self.gt_list = []
        self.adj_list = []
        self.mesh_count = 0
        self.num_faces = []
        self.patch_indices = []
        self.permutations = []
        self.maxSize = maxSize
        self.patchSize = maxSize
        self.coarseningStepNum = coarseningStepNum
        self.coarseningLvlNum = coarseningLvlNum


    def addMeshNew(inputFilePath, filename, gtFilePath, gtFileName):

        myMesh = MeshPair(inputFilePath, filename, gtFilePath, gtFileName)

        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>self.maxSize:
            print("Dividing mesh into patches: %i faces (%i max allowed)"%(facesNum,maxSize))
            patchNum = 0
            nextSeed = -1
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                if nextSeed==-1:
                    faceSeed = np.random.randint(toBeProcessed.shape[0])
                    faceSeed = toBeProcessed[faceSeed]
                else:
                    faceSeed = nextSeed
                    if faceCheck[faceSeed]==1:
                        print("ERROR: Bad seed returned!!")
                        return

                curPatch = Patch(myMesh, patchSize, faceSeed, faceCheck)

                faceCheck[curPatch.faceInd] = 1

                if curPatch.size<100:
                    continue

                if coarseningLvlNum>1:
                    curPatch.coarsen(self.coarseningLvlNum, self.coarseningStepNum)



    def addMesh_TimeEfficient(self, inputFilePath,filename):

        # --- Load mesh ---
        t0 = time.clock()
        V0,_,_, faces0, _ = load_mesh(inputFilePath, filename, 0, False)
        self.V0 = V0[np.newaxis,:,:]

        self.edge_map, self.v_e_map = getEdgeMap(faces0, maxEdges = 20)
        self.edge_map = np.expand_dims(self.edge_map, axis=0)
        self.v_e_map = np.expand_dims(self.v_e_map, axis=0)

        t1 = time.clock()
        print("mesh loaded ("+str(1000*(t1-t0))+"ms)")

        # print("faces0 shape: "+str(faces0.shape))
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        t2 = time.clock()
        print("normals computed ("+str(1000*(t2-t1))+"ms)")
        
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        t3 = time.clock()
        print("Adj computed ("+str(1000*(t3-t2))+"ms)")
        # Get faces position
        f_pos0 = getTrianglesBarycenter(V0, faces0)
        t4 = time.clock()
        print("faces barycenters computed ("+str(1000*(t4-t3))+"ms)")


        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)

        t7 = time.clock()


        # Get patches if mesh is too big
        facesNum = faces0.shape[0]

        # Load GT
        # GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)
        # GTf_normals0 = computeFacesNormals(GT0, faces0)

        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>self.maxSize:
            print("Dividing mesh into patches: %i faces (%i max allowed)"%(facesNum,self.maxSize))
            patchNum = 0
            nextSeed = -1
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                if nextSeed==-1:
                    faceSeed = np.random.randint(toBeProcessed.shape[0])
                    faceSeed = toBeProcessed[faceSeed]
                else:
                    faceSeed = nextSeed
                    if faceCheck[faceSeed]==1:
                        print("ERROR: Bad seed returned!!")
                        return
                tp0 = time.clock()

                # testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)

                testPatchAdj, fOldInd, nextSeed = getGraphPatch_wMask(f_adj0, self.patchSize, faceSeed, faceCheck)
                # coo_adj, fOldInd, nextSeed = getSparseGraphPatch_wMask(f_adj0,f_pos0, f_normals0, patchSize, faceSeed, faceCheck)
                tp1 = time.clock()
                print("\nMesh patch extracted ("+str(1000*(tp1-tp0))+"ms)")
                # print("Patch size = %i"%testPatchAdj.shape[0])
                faceCheck[fOldInd]=1
                print("Total added faces = %i"%np.sum(faceCheck==1))

                patchFNormals = f_normals_pos[fOldInd]
                # patchGTFNormals = GTf_normals0[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue

                if self.coarseningLvlNum>1:
                    # Convert to sparse matrix and coarsen graph
                    # tls0 = time.clock()
                    coo_adj = listToSparseWNormals(testPatchAdj, patchFNormals[:,-3:], patchFNormals[:,:3])
                    # print("list to sparse conversion ("+str(1000*(time.clock()-tls0))+"ms)")
                    has_sat = True

                    while has_sat:
                        print("Coarsening...")
                        tc0 = time.clock()
                        adjs, newToOld = coarsen(coo_adj,(self.coarseningLvlNum-1)*self.coarseningStepNum)
                        tc1 = time.clock()
                        print("Coarsening complete ("+str(1000*(tc1-tc0))+"ms)")
                        has_sat = False
                        # Change adj format
                        fAdjs = []
                        for lvl in range(self.coarseningLvlNum):
                            tsl0 = time.clock()
                            fadj, has_sat_temp = sparseToList(adjs[self.coarseningStepNum*lvl],K_faces)
                            print("sparse to list conversion ("+str(1000*(time.clock()-tsl0))+"ms)")
                            fadj = np.expand_dims(fadj, axis=0)
                            fAdjs.append(fadj)
                            has_sat = has_sat or has_sat_temp



                    # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                    new_N = len(newToOld)
                    
                    padding6 =np.zeros((new_N-old_N,patchFNormals.shape[1]))
                    # padding6 =np.zeros((new_N-old_N,33))
                    padding3 =np.zeros((new_N-old_N,3))
                    patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                    # patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                    # Reorder nodes
                    patchFNormals = patchFNormals[newToOld]
                    # patchGTFNormals = patchGTFNormals[newToOld]

                else:   # One level only: no coarsening
                    fAdjs = []
                    fAdjs.append(testPatchAdj[np.newaxis,:,:])


                ##### Save number of triangles and patch new_to_old permutation #####
                self.num_faces.append(old_N)
                self.patch_indices.append(fOldInd)
                if self.coarseningLvlNum>1:
                    self.permutations.append(inv_perm(newToOld))
                #####################################################################

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                #f_adj = np.expand_dims(testPatchAdj, axis=0)
                # GTf_normals = np.expand_dims(patchGTFNormals, axis=0)

                self.in_list.append(f_normals)
                self.adj_list.append(fAdjs)
                # self.gt_list.append(GTf_normals)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(self.mesh_count) + ")")
                self.mesh_count+=1
                patchNum+=1
        else:       #Small mesh case
            old_N = facesNum

            if self.coarseningLvlNum>1:
                # Convert to sparse matrix and coarsen graph
                # print("f_adj0 shape: "+str(f_adj0.shape))
                # print("f_pos0 shape: "+str(f_pos0.shape))
                coo_adj = listToSparseWNormals(f_adj0, f_pos0, f_normals0)
                
                has_sat = True

                while has_sat:
                    print("Coarsening...")
                    adjs, newToOld = coarsen(coo_adj,(self.coarseningLvlNum-1)*self.coarseningStepNum)
                    has_sat = False

                    # Change adj format
                    fAdjs = []
                    for lvl in range(self.coarseningLvlNum):
                        fadj, has_sat_temp = sparseToList(adjs[self.coarseningStepNum*lvl],K_faces)
                        fadj = np.expand_dims(fadj, axis=0)
                        fAdjs.append(fadj)
                        has_sat = has_sat or has_sat_temp

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                padding6 =np.zeros((new_N-old_N,f_normals_pos.shape[1]))
                padding3 =np.zeros((new_N-old_N,3))
                f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
                # GTf_normals0 = np.concatenate((GTf_normals0, padding3),axis=0)

                # Reorder nodes
                f_normals_pos = f_normals_pos[newToOld]
                # GTf_normals0 = GTf_normals0[newToOld]

            else:
                fAdjs = []
                fAdjs.append(f_adj0[np.newaxis,:,:])

            ##### Save number of triangles and patch new_to_old permutation #####
            self.num_faces.append(old_N) # Keep track of fake nodes
            self.patch_indices.append([])
            if self.coarseningLvlNum>1:
                self.permutations.append(inv_perm(newToOld)) # Nothing to append here, faces are already correctly ordered
            #####################################################################

            

            

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            #f_adj = np.expand_dims(f_adj0, axis=0)
            # GTf_normals = np.expand_dims(GTf_normals0, axis=0)

            self.in_list.append(f_normals)
            self.adj_list.append(fAdjs)
            # self.gt_list.append(GTf_normals)
        
            
            self.mesh_count+=1


class TrainingSet(PreprocessedData):
    pass