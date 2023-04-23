import igl
import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_matrix

class CubeStylier:
    
    def __init__(self, mesh_file=None, V=None, F=None) -> None:
        
        if mesh_file is not None:
            V, F = igl.read_triangle_mesh(mesh_file)
        self.V, self.F = V, F
        NV = V.shape[0]
            
        # coefs
        self.cubeness = 4e-1
        self.rho = 1e-4
        self.ABSTOL = 1e-5
        self.RELTOL = 1e-3
        self.mu = 5
        self.tao = 2 
        self.maxIter_ADMM = 20
        
        # arap constraints
        self.handles = np.array([0])
        self.handles_pos = V[self.handles]
        
        # pre-computed mesh properties
        self.L = igl.cotmatrix(V, F)  # (NV, NV) sparse
        self.N = igl.per_vertex_normals(V, F)[:, :, None]  # (NV, 3, 1)
        self.VA = igl.massmatrix(V, F).diagonal() # (NV, )
        
        self.arap_rhs = igl.arap_rhs(V, F, 3, igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)
        
        # vertex_face_adj[vf_indices[i]:vf_indices[i+1]] is the face indices adjacent to V[i]
        VF, NI = igl.vertex_triangle_adjacency(F, NV)
        self.NI = NI * 3
        self.hElist = np.empty((VF.shape[0] * 3, 2), dtype=np.int32)
        self.hElist[::3, 0] = F[VF, 0]
        self.hElist[1::3, 0] = F[VF, 1]
        self.hElist[2::3, 0] = F[VF, 2]
        self.hElist[::3, 1] = F[VF, 1]
        self.hElist[1::3, 1] = F[VF, 2]
        self.hElist[2::3, 1] = F[VF, 0]
        self.dVlist = V[self.hElist[:, 1]] - V[self.hElist[:, 0]]
        self.Wlist = self.L[self.hElist[:, 1], self.hElist[:, 0]]
        # params to be updated
        self.reset()
        
    def reset(self):
        NV = self.V.shape[0]
        self.U = self.V.copy()
        self.zAll = np.zeros((NV, 3, 1))
        self.uAll = np.zeros((NV, 3, 1))
        self.rhoAll = np.full((NV, ), self.rho)
        self.RAll = np.zeros((NV, 3, 3))
        
    @staticmethod
    def fit_R(S):
        U, X, V_t = np.linalg.svd(S)
        R = (U @ V_t).T
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = (U @ V_t).T
        return R
    
    @staticmethod
    def shrinkage(x, k):
        return np.maximum(0, x - k) - np.maximum(0, -x - k)
    
    def step(self):
        Aeq = csc_matrix((0, 0))
        Beq = np.array([])
        
        NV = self.V.shape[0]
        dUlist = self.U[self.hElist[:, 1]] - self.U[self.hElist[:, 0]]
        for ii in tqdm(range(NV)):
            z = self.zAll[ii]
            u = self.uAll[ii]
            rho = self.rhoAll[ii]
            n = self.N[ii]
            
            W = self.Wlist[0, self.NI[ii]:self.NI[ii+1]]
            W.resize(W.shape[1])
            W = np.diag(W)
            dV = self.dVlist[self.NI[ii]:self.NI[ii+1]]
            dU = dUlist[self.NI[ii]:self.NI[ii+1]]
            Spre = dV.T @ W @ dU
            
            for _ in range(self.maxIter_ADMM):
                S = Spre + rho * (n @ (z-u).T)
                R = self.fit_R(S)
                z_old = z
                z = self.shrinkage(R @ n + u, self.cubeness * self.VA[ii] / rho)
                u += R @ n - z
                
                r_norm = np.linalg.norm(z - R @ n)
                s_norm = np.linalg.norm(-rho * (z - z_old))
                
                if r_norm > self.mu * s_norm:
                    rho *= self.tao
                    u /= self.tao
                elif s_norm > self.mu * r_norm:
                    rho /= self.tao
                    u *= self.tao
                
                eps_pri = np.sqrt(6) * self.ABSTOL + self.RELTOL * np.maximum(np.linalg.norm(R @ n), np.linalg.norm(z))
                eps_dual = np.sqrt(3) * self.ABSTOL + self.RELTOL * np.linalg.norm(rho * u)
                
                if r_norm < eps_pri and s_norm < eps_dual:
                    self.zAll[ii] = z
                    self.uAll[ii] = u
                    self.rhoAll[ii] = rho
                    self.RAll[ii] = R
                    break
                
        Rcol = self.RAll.reshape(NV * 3 * 3, 1, order='F')
        Bcol = self.arap_rhs @ Rcol
        B = Bcol.reshape(int(Bcol.shape[0] / 3), 3, order='F')
        _, self.U = igl.min_quad_with_fixed(self.L, B, self.handles, self.handles_pos, Aeq, Beq, False)
        
        
cube = CubeStylier("../meshes/bunny.obj")
for _ in range(10):
    cube.step()
    
igl.write_triangle_mesh("result.obj", cube.U, cube.F)