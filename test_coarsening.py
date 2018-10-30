from lib.coarsening import *
import scipy.sparse
from utils import *

if __name__ == '__main__':
	testAdj = np.array([[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,1],[0,0,1,1,0],[0,0,1,0,1]])

	testPos = np.array([[1,1,1],[1,0,0],[0,0,0],[-100,-2,0],[-1,0,0]])

	A = scipy.sparse.coo_matrix(testAdj)
	
	print(str(A.toarray()))

	lA = sparseToList(A,5)

	print("list A: \n"+str(lA))

	spA = listToSparse(lA,testPos)

	print("sparse A: \n"+str(spA.toarray()))

	graphs,perms = coarsen(spA,3)

	print("graphs: 0 - \n"+str(graphs[0].toarray()))
	print("graphs: 1 - \n"+str(graphs[1].toarray()))
	print("graphs: 2 - \n"+str(graphs[2].toarray()))
	print("graphs: 3 - \n"+str(graphs[3].toarray()))

	print("perm: "+str(perms))