import numpy as np
import open3d as o3d
import scipy.sparse as sp
from igl import *
from multiprocessing import Pool
from math import sqrt

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

        data.hEList.append(np.zeros((len(adjF) * 3, 2)).astype(np.int32))
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

    :param V: a matrix of vertex positions
    :param U: a matrix of vertex positions
    :param data: cubic style data
    :return: V, U and data after precomputation
    """

    # local step
    RAll = np.zeros((3, 3 * len(V)))
    V, U, RAll, data = fit_rotations_l1(V, U, RAll, data)

    # global step
    Upre = U

    # TODO: igl::columnize(RAll, V.rows(), 2, Rcol);
    Rcol = np.zeros((9 * len(V), 1))

    Bcol = data.K * Rcol
    for dim in range(V.shape[1]):
        Bc = Bcol[dim * V.shape[0]:(dim + 1) * V.shape[0], 0:1]
        # have the same effect as Bc = Bcol.block(dim*V.rows(),0,V.rows(),1);
        bcc = data.bc[:, dim:dim+1]
        # bcc = data.bc.col(dim);

        #TODO: min_quad_with_fixed_solve(data.solver_data,Bc,bcc,VectorXd(),Uc);
        #TODO: U.col(dim) = Uc;


    #print optimization date
    data.reldV = np.abs(U - Upre).max() + np.abs(U - V).max()
    print(f"reldV:  {data.reldV}\n")


    return V, U, data


def process_row_data(ii,V, U, RAll, data):
    z = data.zAll[:, ii: ii + 1]
    u = data.uAll[:, ii: ii + 1]
    n = data.N[ii].T
    rho = data.rhoAll[ii]

    # get energy parameters
    # Note: dVn = [dV n], dUn = [dU z-u]
    hE = data.hEList[ii]

    U_hE0 = U[hE[:, 0], :]
    U_hE1 = U[hE[:, 1], :]
    dU = (U_hE1 - U_hE0).T

    dV = data.dVList[ii]
    WVec = data.WVecList[ii]
    Spre = dV * np.diag(WVec) * dU.T

    R = np.zeros((3, 3))

    # start to ADMM
    for k in range(data.maxIter_ADMM):
        # R step
        S = Spre + (rho * n * (z - u).T)
        R = orthogonal_procrustes(S)

        # z step
        zOld = z
        z = shrinkage(R * n + u, data.Lambda * data.VA(ii) / rho, z)

        # u step
        u = u + R * n - z
        # TODO: whether use out to substitue .noalias()

        # compute residual
        r_norm = np.linalg.norm((z - R * n))
        s_norm = np.linalg.norm((-rho * (z - zOld)))

        # rho step
        if r_norm > data.mu * s_norm:
            rho = data.tao * rho
            u = u / data.tao
        elif s_norm > data.mu * r_norm:
            rho = rho / data.tao
            u = u * data.tao

        # stopping criteria
        nz = float(len(z))
        eps_pri = sqrt(2.0 * nz) * data.ABSTOL + data.RELTOL * max(np.linalg.norm(R * n), np.linalg.norm(z))
        eps_dual = sqrt(1.0 * nz) * data.ABSTOL + data.RELTOL * (np.linalg.norm(rho * u))

        if r_norm < eps_pri and s_norm < eps_dual:
            # save parameters
            data.zAll[:, ii: ii + 1] = z
            data.uAll[:, ii: ii + 1] = u
            data.rhoAll[ii] = rho
            # Set the block of RAll to R
            RAll[:, 3 * ii:3 * ii + 3] = R

            # save objective
            objVal = np.trace(0.5 * ((R * dV - dU) * np.diag(WVec) * (R * dV - dU).T)) + data.Lambda * data.VA[
                ii] * np.sum(np.abs(R * n))
            data.objValVec[ii] = objVal
            break

    # ADMM end
    return V, U, RAll, data


def fit_rotations_l1(V, U, RAll, data):
    """
    :param V: a matrix of vertex positions
    :param U: a matrix of vertex positions
    :param RAll:
    :param data: data: cubic style data
    :return: result after fit rotation
    """
    data.objValVec = np.zeros(len(V))

    # TODO: DO not know how to substitue for igl::parallel_for, use pool instead
    with Pool(processes=1000) as pool:
        def process(ii):
            return process_row_data(ii, V, U, RAll, data)

        V, U, RAll, data = pool.map(process, range(V.shape[0]))

    data.objVal = np.sum(data.objValVec)
    return V, U, RAll, data

def orthogonal_procrustes(S):
    """
    :param S: the original S
    :return: using SVD to calculate the rotation matrix R
    """
    # Compute the SVD of the matrix
    SU, C, SV = np.linalg.svd(S)
    R = SV * SU.T
    if np.linalg.det(R) < 0:
        SU[:, 2] = -SU[:, 2]
        R = SV * SU.T

    assert (R.determinant() > 0), "the determinant of rotation matrix is smaller than 0"
    return R

def shrinkage(x, k, z):
    """
    :param x: a vector
    :param k: a constant
    :param z:
    :return:
    """
    tmp1 = x - k
    posMax = np.maximum(tmp1, 0.0).max()

    tmp2 = -1 * x - k
    negMax = np.maximum(tmp2, 0.0).max()

    z = posMax - negMax
    return z