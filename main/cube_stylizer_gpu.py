import taichi as ti
from scipy.sparse import csc_matrix
import igl
import numpy as np
ti.init(arch=ti.gpu)


@ti.data_oriented
class CubeStylizer:
    def __init__(self, mesh_file=None, V=None, F=None) -> None:
        if mesh_file is not None:
            V, F = igl.read_triangle_mesh(mesh_file)
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
        
        # compute necessary geometry properties
        L = igl.cotmatrix(V, F)  # NV * NV sparse
        self.L = L
        N = igl.per_vertex_normals(V, F)  # NV * 3
        VA = igl.massmatrix(V, F).diagonal() # NV
        # vertex_face_adj[vf_indices[i]:vf_indices[i+1]] is the face indices adjacent to V[i]
        VF, NI = igl.vertex_triangle_adjacency(F, NV)
        self.max_degree = np.max(NI[1:] - NI[:-1])
        NI *= 3
        self.arap_rhs = igl.arap_rhs(V, F, 3, igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)
        
        # set_up taichi field for parallelized local step
        self.V = ti.Vector.field(n=3, shape=(NV,), dtype=ti.f32)
        self.V.from_numpy(V)
        
        self.U = ti.Vector.field(n=3, shape=(NV,), dtype=ti.f32)
        self.U.from_numpy(V)
        
        self.F = ti.Vector.field(n=3, shape=(F.shape[0], ), dtype=ti.i32)
        self.F.from_numpy(F)
        
        self.vertex_face_adj = ti.field(ti.i32, shape=VF.shape)
        self.vertex_face_adj.from_numpy(VF)
        
        self.vf_indices = ti.field(ti.i32, shape=NI.shape)
        self.vf_indices.from_numpy(NI)
        
        self.normals = ti.Vector.field(n=3, shape=(NV,), dtype=ti.f32)
        self.normals.from_numpy(N)
        
        self.laplacian = ti.field(ti.f32, shape=(NV, NV))
        self.laplacian.from_numpy(L.todense())
        
        self.vertex_area = ti.field(ti.f32, shape=(NV, ))
        self.vertex_area.from_numpy(VA)
        
        self.zAll = ti.Vector.field(n=3, dtype=ti.f32, shape=(NV,))
        self.uAll = ti.Vector.field(n=3, dtype=ti.f32, shape=(NV,))
        self.rhoAll = ti.field(ti.f32, shape=(NV, ))
        self.RAll = ti.Matrix.field(n=3, m=3, shape=(NV, ), dtype=ti.f32)
        
        
        
        self.half_edge_list = ti.Vector.field(n=2, shape=(VF.shape[0] * 3,), dtype=ti.i32)
        self.W_list = ti.field(ti.f32, shape=(VF.shape[0] * 3,))
        self.dV_list = ti.Vector.field(n=3, shape=(VF.shape[0] * 3,), dtype=ti.f32)
        
        self.reset()
        self._precompute()
        
    def reset(self):
        self.zAll.fill(0.)
        self.uAll.fill(0.)
        self.rhoAll.fill(self.rho)
        self.RAll.fill(0.)
        self.U.copy_from(self.V)
        
    @ti.kernel
    def _precompute(self):
        for i in range(self.vertex_face_adj.shape[0]):
            
            adj = self.vertex_face_adj[i]
            for j in range(3):
                vi0, vi1 = self.F[adj][j], self.F[adj][(j + 1) % 3]
                self.half_edge_list[3 * i + j][0] = vi0
                self.half_edge_list[3 * i + j][1] = vi1
                self.dV_list[3 * i + j] = self.V[vi1] - self.V[vi0]
                self.W_list[3 * i + j] = self.laplacian[vi1, vi0]
    
    @staticmethod    
    @ti.func
    def fit_R(S):
        U, X, V = ti.svd(S)
        R = V @ U.transpose()
        if R.determinant() < 0:
            U[0, 2] *= -1
            U[1, 2] *= -1
            U[2, 2] *= -1
            R = V @ U.transpose()
        return R
    
    @staticmethod
    @ti.func
    def shrinkage(x, k):
        return ti.max(x - k, 0) - ti.max(-x-k, 0)
    
    @ti.kernel
    def fit_rotation_l1(self):
        for vi in range(self.V.shape[0]):
            z = self.zAll[vi]
            u = self.uAll[vi]
            n = self.normals[vi]
            rho = self.rhoAll[vi]
            
            size = self.vf_indices[vi+1] - self.vf_indices[vi]
            W = ti.Matrix.zero(dt=ti.f32,n = self.max_degree * 3, m = self.max_degree * 3)
            dU = ti.Matrix.zero(dt=ti.f32, n=self.max_degree * 3, m=3)
            dV = ti.Matrix.zero(dt=ti.f32, n=3, m=self.max_degree * 3)
            
            for s in range(size):
                vf_idx = self.vf_indices[vi] + s
                W[s, s] = self.W_list[vf_idx]
                
                u0 = self.U[self.half_edge_list[vf_idx][0]]
                u1 = self.U[self.half_edge_list[vf_idx][1]]
                dU[s, :] = u1 - u0
                dV[:, s] = self.dV_list[vf_idx]
            Spre = dV @ W @ dU
            
            for _ in range(self.maxIter_ADMM):
                S = Spre + rho * (n.outer_product(z - u))
                R = self.fit_R(S)
                z_old = z
                z = self.shrinkage(R @ n + u, self.cubeness * self.vertex_area[vi] / rho)
                u += R @ n - z
                
                
                r_norm = (z - R @ n).norm()
                s_norm = (-rho * (z - z_old)).norm()
                if r_norm > self.mu * s_norm:
                    rho *= self.tao
                    u /= self.tao
                elif s_norm > self.mu * r_norm:
                    rho /= self.tao
                    u *= self.tao
                    
                self.zAll[vi] = z
                self.uAll[vi] = u
                self.rhoAll[vi] = rho
                self.RAll[vi] = R
    def step(self):
        Aeq = csc_matrix((0, 0))
        Beq = np.array([])
        self.fit_rotation_l1()
        Rcol = self.RAll.to_numpy().reshape(self.V.shape[0] * 3 * 3, 1, order='F')
        Bcol = self.arap_rhs @ Rcol
        B = Bcol.reshape(int(Bcol.shape[0] / 3), 3, order='F')
        _, U = igl.min_quad_with_fixed(self.L, B, self.handles, self.handles_pos, Aeq, Beq, False)
        self.U.from_numpy(U)
        
    def iterate(self, step_num=10):
        for i in range(step_num):
            print(f"\033[34m[INFO] Interation step: {i}\033[0m")
            self.step()
            

        
def main():
    cube = CubeStylizer("../meshes/bunny.obj")
    cube.iterate(10)
    igl.write_triangle_mesh("result.obj", cube.U.to_numpy(), cube.F.to_numpy())


if __name__ == '__main__':
    main()
