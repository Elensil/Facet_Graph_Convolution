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

class PreprocessedData(object):
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
        self.minPatchSize = MIN_PATCH_SIZE

        # Additional lists for vertices stuff
        self.vOldInd_list = []
        self.fOldInd_list = []
        self.v_list = []
        self.gtv_list = []
        self.faces_list = []
        self.v_faces_list = []


    # def addMeshNew(inputFilePath, filename, gtFilePath, gtFileName):

    #     myMesh = MeshPair(inputFilePath, filename, gtFilePath, gtFileName)

    #     faceCheck = np.zeros(facesNum)
    #     faceRange = np.arange(facesNum)
    #     if facesNum>self.maxSize:
    #         print("Dividing mesh into patches: %i faces (%i max allowed)"%(facesNum,maxSize))
    #         patchNum = 0
    #         nextSeed = -1
    #         while(np.any(faceCheck==0)):
    #             toBeProcessed = faceRange[faceCheck==0]
    #             if nextSeed==-1:
    #                 faceSeed = np.random.randint(toBeProcessed.shape[0])
    #                 faceSeed = toBeProcessed[faceSeed]
    #             else:
    #                 faceSeed = nextSeed
    #                 if faceCheck[faceSeed]==1:
    #                     print("ERROR: Bad seed returned!!")
    #                     return

    #             curPatch = Patch(myMesh, patchSize, faceSeed, faceCheck)

    #             faceCheck[curPatch.faceInd] = 1

    #             if curPatch.size<100:
    #                 continue

    #             if coarseningLvlNum>1:
    #                 curPatch.coarsen(self.coarseningLvlNum, self.coarseningStepNum)


    def addMesh(self, inputFilePath, filename):
        V,_,_, faces, _ = load_mesh(inputFilePath, filename, 0, False)
        self.addMesh_TimeEfficient(V,faces)


    def addMesh_TimeEfficient(self, V0, faces0, GTV=None):
        addGT = False
        if GTV is not None:
            addGT = True
        # --- Load mesh ---
        t0 = time.clock()
        self.edge_map, self.v_e_map = getEdgeMap(faces0, maxEdges = 20)
        self.edge_map = np.expand_dims(self.edge_map, axis=0)
        self.v_e_map = np.expand_dims(self.v_e_map, axis=0)

        t1 = time.clock()
        print("mesh loaded ("+str(1000*(t1-t0))+"ms)")

        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)

        # self.normals = f_normals0
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

        if addGT:
            GTf_normals0 = computeFacesNormals(GTV, faces0)

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

                testPatchAdj, fOldInd, nextSeed = getGraphPatch_wMask(f_adj0, self.patchSize, faceSeed, faceCheck, self.minPatchSize)
                tp1 = time.clock()
                print("\nMesh patch extracted ("+str(1000*(tp1-tp0))+"ms)")

                faceCheck[fOldInd]=1
                print("Total added faces = %i"%np.sum(faceCheck==1))

                patchFNormals = f_normals_pos[fOldInd]
                if addGT:
                    patchGTFNormals = GTf_normals0[fOldInd]

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
                    padding3 =np.zeros((new_N-old_N,3))
                    patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                    # Reorder nodes
                    patchFNormals = patchFNormals[newToOld]
                    
                    if addGT:
                        patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                        patchGTFNormals = patchGTFNormals[newToOld]

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

                self.in_list.append(f_normals)
                self.adj_list.append(fAdjs)
                if addGT:
                    GTf_normals = np.expand_dims(patchGTFNormals, axis=0)
                    self.gt_list.append(GTf_normals)

                # print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(self.mesh_count) + ")")
                self.mesh_count+=1
                patchNum+=1
        else:       #Small mesh case
            old_N = facesNum

            if self.coarseningLvlNum>1:
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
                # Reorder nodes
                f_normals_pos = f_normals_pos[newToOld]

                if addGT:
                    GTf_normals0 = np.concatenate((GTf_normals0, padding3),axis=0)
                    GTf_normals0 = GTf_normals0[newToOld]

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
            

            self.in_list.append(f_normals)
            self.adj_list.append(fAdjs)

            if addGT:
                GTf_normals = np.expand_dims(GTf_normals0, axis=0)
                self.gt_list.append(GTf_normals)
        
            
            self.mesh_count+=1


    def addMeshWithVertices(self, V0, faces0, GTV=None):

        vNum = V0.shape[0]
        # Compute normals
        f_normals0 = computeFacesNormals(V0, faces0)
        # Get adjacency
        f_adj0 = getFacesLargeAdj(faces0,K_faces)
        # Get faces position
        # print("WARNING: temp change to face position normalization!! TO BE REMOVED!!!")
        f_pos0 = getTrianglesBarycenter(V0, faces0, normalize=True)
        # f_pos0 = np.reshape(f_pos0,(-1,3))

        f_area0 = np.expand_dims(getTrianglesArea(V0,faces0, normalize=True), axis=1)

        f_borderCh0 = np.expand_dims(getBorderFaces(faces0),axis=1)

        # print("WARNING!!! Added area channel and binary channel for border faces")
        # f_normals_pos = np.concatenate((f_normals0, f_area0, f_borderCh0, f_pos0), axis=1)
        # print("WARNING!!! Added binary channel for border faces")
        # f_normals_pos = np.concatenate((f_normals0, f_borderCh0, f_pos0), axis=1)
        f_normals_pos = np.concatenate((f_normals0, f_pos0), axis=1)

        # Load GT
        GT0,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)

        gtf_normals0 = computeFacesNormals(GT0, faces0)

        # Normalize vertices
        V0, GT0 = normalizePointSets(V0,GT0)


        # Get patches if mesh is too big
        facesNum = faces0.shape[0]
        faceCheck = np.zeros(facesNum)
        faceRange = np.arange(facesNum)
        if facesNum>maxSize:
            patchNum = 0
            # while((np.any(faceCheck==0))and(patchNum<3)):
            while(np.any(faceCheck==0)):
                toBeProcessed = faceRange[faceCheck==0]
                faceSeed = np.random.randint(toBeProcessed.shape[0])
                faceSeed = toBeProcessed[faceSeed]

                testPatchV, testPatchF, testPatchAdj, vOldInd, fOldInd = getMeshPatch(V0, faces0, f_adj0, patchSize, faceSeed)
                faceCheck[fOldInd]+=1

                patchFNormals = f_normals_pos[fOldInd]
                patchGTFNormals = gtf_normals0[fOldInd]

                old_N = patchFNormals.shape[0]

                # Don't add small disjoint components
                if old_N<100:
                    continue
                
                # For CNR dataset: one-one correspondence between vertices
                # patchGTV = GT0[vOldInd]

                # For DTU: take slice of GT points
                patchBB = getBoundingBox(testPatchV)
                patchGTV = takePointSetSlice(GT0,patchBB)
                
                # If no GT in the window, skip this patch (fake surface)
                if patchGTV.shape[0]<testPatchV.shape[0]:
                    continue


                self.vOldInd_list.append(vOldInd)
                self.fOldInd_list.append(fOldInd)

                # Convert to sparse matrix and coarsen graph
                coo_adj = listToSparseWNormals(testPatchAdj, patchFNormals[:,-3:], patchFNormals[:,:3])
                adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)

                # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
                new_N = len(newToOld)
                
                # padding6 =np.zeros((new_N-old_N,6))
                padding6 =np.zeros((new_N-old_N,patchFNormals.shape[1]))
                padding3 =np.zeros((new_N-old_N,3))
                minusPadding3 = padding3-1
                patchFNormals = np.concatenate((patchFNormals,padding6),axis=0)
                testPatchF = np.concatenate((testPatchF,minusPadding3),axis=0)
                patchGTFNormals = np.concatenate((patchGTFNormals, padding3),axis=0)
                # Reorder nodes
                patchFNormals = patchFNormals[newToOld]
                testPatchF = testPatchF[newToOld]
                patchGTFNormals = patchGTFNormals[newToOld]

                oldToNew = inv_perm(newToOld)


                ##### Save number of triangles and patch new_to_old permutation #####
                self.num_faces.append(old_N)
                self.patch_indices.append(fOldInd)

                self.permutations.append(oldToNew)
                #####################################################################

                # Change adj format
                fAdjs = []
                for lvl in range(coarseningLvlNum):
                    fadj, _ = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                    fadj = np.expand_dims(fadj, axis=0)
                    fAdjs.append(fadj)
                        # fAdjs = []
                        # f_adj = np.expand_dims(testPatchAdj, axis=0)
                        # fAdjs.append(f_adj)

                v_faces = getVerticesFaces(testPatchF,25,testPatchV.shape[0])

                # Expand dimensions
                f_normals = np.expand_dims(patchFNormals, axis=0)
                v_pos = np.expand_dims(testPatchV,axis=0)
                faces = np.expand_dims(testPatchF, axis=0)
                gtv_pos = np.expand_dims(patchGTV,axis=0)
                v_faces = np.expand_dims(v_faces,axis=0)
                gtf_normals = np.expand_dims(patchGTFNormals, axis=0)

                self.v_list.append(v_pos)
                self.gtv_list.append(gtv_pos)
                n_list.append(f_normals)
                adj_list.append(fAdjs)
                self.faces_list.append(faces)
                self.v_faces_list.append(v_faces)
                gtn_list.append(gtf_normals)

                print("Added training patch: mesh " + filename + ", patch " + str(patchNum) + " (" + str(self.mesh_count) + ")")
                self.mesh_count+=1
                patchNum+=1
        else:       #Small mesh case

            # Convert to sparse matrix and coarsen graph
            coo_adj = listToSparseWNormals(f_adj0, f_pos0, f_normals0)
            adjs, newToOld = coarsen(coo_adj,(coarseningLvlNum-1)*coarseningStepNum)
            # There will be fake nodes in the new graph: set all signals (normals, position) to 0 on these nodes
            new_N = len(newToOld)
            old_N = facesNum
            # padding6 =np.zeros((new_N-old_N,6))
            padding6 =np.zeros((new_N-old_N,f_normals_pos.shape[1]))
            padding3 =np.zeros((new_N-old_N,3))
            minusPadding3 = padding3-1
            minusPadding3 = minusPadding3.astype(int)

            faces0 = np.concatenate((faces0,minusPadding3),axis=0)

            f_normals_pos = np.concatenate((f_normals_pos,padding6),axis=0)
            gtf_normals = np.concatenate((gtf_normals0, padding3),axis=0)

            oldToNew = inv_perm(newToOld)

            ##### Save number of triangles and patch new_to_old permutation #####
            self.num_faces.append(old_N) # Keep track of fake nodes
            self.patch_indices.append([])
            self.permutations.append(oldToNew)
            self.fOldInd_list.append([])
            self.vOldInd_list.append([])
            #####################################################################

            # Reorder nodes
            f_normals_pos = f_normals_pos[newToOld]
            faces0 = faces0[newToOld]
            gtf_normals = gtf_normals[newToOld]
            

            # Change adj format
            fAdjs = []
            for lvl in range(coarseningLvlNum):
                fadj, _ = sparseToList(adjs[coarseningStepNum*lvl],K_faces)
                fadj = np.expand_dims(fadj, axis=0)
                fAdjs.append(fadj)


            # fadj = sparseToList(adjs[4],K_faces)
            # fadj = np.expand_dims(fadj, axis=0)
            # fAdjs.append(fadj)
            # fadj = sparseToList(adjs[5],K_faces)
            # fadj = np.expand_dims(fadj, axis=0)
            # fAdjs.append(fadj)

            v_faces = getVerticesFaces(faces0,25,V0.shape[0])

            # Expand dimensions
            f_normals = np.expand_dims(f_normals_pos, axis=0)
            v_pos = np.expand_dims(V0,axis=0)
            gtv_pos = np.expand_dims(GT0,axis=0)
            faces = np.expand_dims(faces0, axis=0)
            v_faces = np.expand_dims(v_faces,axis=0)
            gtf_normals = np.expand_dims(gtf_normals,axis=0)

            self.v_list.append(v_pos)
            self.gtv_list.append(gtv_pos)
            self.in_list.append(f_normals)
            self.adj_list.append(fAdjs)
            self.faces_list.append(faces)
            self.v_faces_list.append(v_faces)
            self.gt_list.append(gtf_normals)
        
            print("Added training mesh " + filename + " (" + str(self.mesh_count) + ")")

            self.mesh_count+=1

        return vNum, facesNum




'''
Changes from PreprocessedData:
    - minimum patch size is set equal to max patch size instead of global parameter MIN_PATCH_SIZE
    - new method addMeshWithGT : just like addMesh, but with GT loading (no way!)
'''

class TrainingSet(PreprocessedData):
    
    # Override constructor
    def __init__(self, maxSize, coarseningStepNum, coarseningLvlNum):
        # Call parent constructor
        super(TrainingSet,self).__init__(maxSize, coarseningStepNum, coarseningLvlNum)
        # But edit min patch size
        self.minPatchSize = self.patchSize


    def addMeshWithGT(self, inputFilePath, filename, gtFilePath, gtfilename):
        V,_,_, faces, _ = load_mesh(inputFilePath, filename, 0, False)
        GTV,_,_,_,_ = load_mesh(gtFilePath, gtfilename, 0, False)

        self.addMesh_TimeEfficient(V,faces, GTV=GTV)
    

# This class is meant to load one mesh only (though it can separate it in different patches)
class InferenceMesh(PreprocessedData):

    # Override parent method in order to set whole mesh data (vertices, faces, normals)
    def addMesh(self, inputFilePath, filename):
        V,_,_, faces, _ = load_mesh(inputFilePath, filename, 0, False)
        self.addMesh_TimeEfficient(V,faces)

        self.vertices = V[np.newaxis,:,:]
        self.faces = faces
        self.normals = computeFacesNormals(V, faces)

    # Override parent method in order to set whole mesh data (vertices, faces, normals)
    def addMeshWithVertices(self, inputFilePath, filename):
        V,_,_, faces, _ = load_mesh(inputFilePath, filename, 0, False)
        self.fNum = faces.shape[0]
        self.vNum = V.shape[0]
        super(InferenceMesh,self).addMeshWithVertices(V, faces)
        self.vertices = V[np.newaxis,:,:]
        self.faces = faces
        self.normals = computeFacesNormals(V, faces)
