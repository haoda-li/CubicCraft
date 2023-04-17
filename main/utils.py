import numpy as np
import open3d as o3d
import scipy.sparse as sp
from igl import *

def test():
    print("hello world")


def normalize_unitbox(V=None):
    '''
    :param V:a matrix of vertex positions
    :return: V: a matrix of vertex positions (in a unit box)
    '''
    # subtract the minimum value in each column from every element in that column
    V = V - np.min(V, axis=0)
    # divide by the maximum value in the matrix
    V = V / np.max(V)
    return V

def cube_style_precomputation(V, F, data):
    '''

    :param V: a matrix of vertex positions
    :param F: a matrix of faces information
    :param data: cubic style data
    :return: the V, F and data after precomputation
    '''
    data.reset()
    rows = V.shape[0]
    cols = V.shape[1]

    # extract the per-vertex normals as a NumPy array
    data.N = per_vertex_normals(V, F)
    # compute the cotangent Laplacian matrix of a triangular mesh
    data.L = cotmatrix(V, F)
    # compute the barycentric mass matrix
    M = massmatrix(V, F, type=igl.MASSMATRIX_TYPE_BARYCENTRIC)
    data.VA = M.diagonal()

    # computer vertex triangle adjaceny
    adjFList, VI = vertex_triangle_adjacency(F, rows)
    """
    example output of adjaceny list
    adjFList: [0 1 2 0 2 0 1 1 2]
    VI: [0 3 5 7 9]
    """

    # right-hand side constructor of global poisson solve for various ARAP energies
    data.K = arap_rhs(V, F, cols, igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)

    # for i in range(rows):
    #     data.hEList.append([])
    #     data.WVecList.append([])
    #     data.dVList.append([])
    """
    above code should equal to 
    data.hEList.resize(V.rows());
    data.WVecList.resize(V.rows());
    data.dVList.resize(V.rows());
    """

    for i in range(rows):
        adjF = adjFList[VI[i] : VI[i + 1]]
        # for _ in range(len(adjF) * 3):
        #     data.hEList[i].append([0, 0])
        #     data.WVecList[i].append([0])

        data.hEList.append(np.zeros((len(adjF) * 3, 2)))
        data.WVecList.append(np.zeros((len(adjF) * 3)))
        """
        above code should be equal to
        data.hEList[ii].resize(adjF.size()*3, 2);
        data.WVecList[ii].resize(adjF.size()*3);
        """
        for j in range(len(adjF)):
            v0 = F[adjF[j]][0]
            v1 = F[adjF[j]][1]
            v2 = F[adjF[j]][2]
            # compute adjacent half-edge indices of a vertex
            data.hEList[i][3 * j][0] = v0
            data.hEList[i][3 * j][1] = v1
            data.hEList[i][3 * j + 1][0] = v1
            data.hEList[i][3 * j + 1][1] = v2
            data.hEList[i][3 * j + 2][0] = v2
            data.hEList[i][3 * j + 2][1] = v0
            # compute WVec = vec(W)
            data.WVecList[i][3 * j] = data.L[v0, v1]
            data.WVecList[i][3 * j + 1] = data.L[v1, v2]
            data.WVecList[i][3 * j + 2] = data.L[v2, v0]

        # compute [dV] matrix for each vertex
        data.dVList.append(np.zeros((3, len(adjF) * 3)))

        V_hE0 = V[data.hEList[i][:, 0], :]
        V_hE1 = V[data.hEList[i][:, 1], :]
        """
        this should have the same result as following codes:
        igl::slice(V,data.hEList[ii].col(0),1,V_hE0);
        igl::slice(V,data.hEList[ii].col(1),1,V_hE1);
        """
        data.dVList[i] = (V_hE1 - V_hE0).T

    # TODO: igl::min_quad_with_fixed_precompute(data.L,data.b,SparseMatrix<double>(),false,data.solver_data);

    
    data.zAll = np.random.rand(3, rows)
    data.uAll = np.random.rand(3, rows)
    data.rhoAll = np.ones(rows) * data.rhoInit

    return V, F, data


def cube_style_single_iteration(V, U, data):
    """

    :param V:
    :param U:
    :param data:
    :return:
    """
    pass