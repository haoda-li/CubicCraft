import taichi as ti
import taichi.math as tim
from scipy.sparse import csc_matrix
import igl
import numpy as np

@ti.data_oriented
class CubeStylizer:
    def __init__(self, mesh_file=None, V=None, F=None) -> None:
        if mesh_file is not None:
            V, F = igl.read_triangle_mesh(mesh_file)
        NV = V.shape[0]
        
        # coefs
        self.cubeness = ti.field(dtype=ti.f32, shape=())
        self.cubeness[None] = 4e-1
        self.euler_angles = ti.field(dtype=ti.f32, shape=(3, ))
        self.euler_angles.fill(0.)
        self.rho = 1e-4
        self.ABSTOL = 1e-5
        self.RELTOL = 1e-3
        self.mu = 5
        self.tao = 2 
        self.maxIter_ADMM = 10
        
        # arap constraints
        self.handles = np.array([0])
        self.handles_pos = V[self.handles]
        
        # compute necessary geometry properties
        L = igl.cotmatrix(V, F)  # NV * NV sparse
        self.L = L
        N = igl.per_vertex_normals(V, F)  # NV * 3
        VA = igl.massmatrix(V, F).diagonal() # NV
        self.arap_rhs = igl.arap_rhs(V, F, 3, igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)
        
        # set_up taichi field for parallelized local step
        self.V = ti.Vector.field(n=3, shape=(NV,), dtype=ti.f32)
        self.V.from_numpy(V)
        
        self.U = ti.Vector.field(n=3, shape=(NV,), dtype=ti.f32)
        self.U.from_numpy(V)
        
        self.F = ti.field(shape=(F.shape[0] * 3, ), dtype=ti.i32)
        self.F.from_numpy(F.flatten())
        
        self.normals = ti.Vector.field(n=3, shape=(NV,), dtype=ti.f32)
        self.normals.from_numpy(N)
        
        self.vertex_area = ti.field(ti.f32, shape=(NV, ))
        self.vertex_area.from_numpy(VA)
        
        self.zAll = ti.Vector.field(n=3, dtype=ti.f32, shape=(NV,))
        self.uAll = ti.Vector.field(n=3, dtype=ti.f32, shape=(NV,))
        self.rhoAll = ti.field(ti.f32, shape=(NV, ))
        self.RAll = ti.Matrix.field(n=3, m=3, shape=(NV, ), dtype=ti.f32)
        
        
        VF, NI = igl.vertex_triangle_adjacency(F, NV)
        self.NI = ti.field(ti.i32, shape=NI.shape)
        self.NI.from_numpy(NI * 3)
        
        hElist = np.empty((VF.shape[0] * 3, 2), dtype=np.int32)
        hElist[::3, 0] = F[VF, 0]
        hElist[1::3, 0] = F[VF, 1]
        hElist[2::3, 0] = F[VF, 2]
        hElist[::3, 1] = F[VF, 1]
        hElist[1::3, 1] = F[VF, 2]
        hElist[2::3, 1] = F[VF, 0]
        dVlist = V[hElist[:, 1]] - V[hElist[:, 0]]
        Wlist = self.L[hElist[:, 1], hElist[:, 0]]
        Wlist.resize(Wlist.shape[1])
        
        self.hElist = ti.Vector.field(n=2, shape=(VF.shape[0] * 3,), dtype=ti.i32)
        self.hElist.from_numpy(hElist)
        self.Wlist = ti.field(ti.f32, shape=(VF.shape[0] * 3,))
        self.Wlist.from_numpy(Wlist)
        self.dVlist = ti.Vector.field(n=3, shape=(VF.shape[0] * 3,), dtype=ti.f32)
        self.dVlist.from_numpy(dVlist)
        
        self.reset()
        
    def reset(self):
        self.zAll.fill(0.)
        self.uAll.fill(0.)
        self.rhoAll.fill(self.rho)
        self.RAll.fill(0.)
        self.U.copy_from(self.V)
    
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
        return ti.max(x - k, 0) - ti.max(- x - k, 0)
    
    @ti.kernel
    def fit_rotation_l1(self):
        for vi in range(self.V.shape[0]):
            z = self.zAll[vi]
            u = self.uAll[vi]
            Rot_local = tim.rot_yaw_pitch_roll(
                tim.radians(self.euler_angles[2]), 
                tim.radians(self.euler_angles[0]), 
                tim.radians(self.euler_angles[1])
            )[:3, :3]
            n = Rot_local @ self.normals[vi]
            rho = self.rhoAll[vi]
            
            size = self.NI[vi+1] - self.NI[vi]
            
            Spre = ti.Matrix.zero(dt=ti.f32, n=3, m=3)
            for s in range(size):
                vf_idx = self.NI[vi] + s
                u0 = self.U[self.hElist[vf_idx][0]]
                u1 = self.U[self.hElist[vf_idx][1]]
                wdu = self.Wlist[vf_idx] *(u1 - u0)
                dv = self.dVlist[vf_idx]
                Spre += dv.outer_product(wdu)
            
            for _ in range(self.maxIter_ADMM):
                S = Spre + rho * (n.outer_product(z - u))
                R = self.fit_R(S)
                z_old = z
                z = self.shrinkage(R @ n + u, self.cubeness[None] * self.vertex_area[vi] / rho)
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
            
    def save_mesh(self, path):
        V = self.U.to_numpy()
        F = self.F.to_numpy()
        F = F.reshape(F.shape[0] // 3, 3)
        igl.write_triangle_mesh(path, V, F)


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    cube = CubeStylizer("../meshes/bunny.obj")
    cube.iterate(10)
    # cube.save_mesh("result.obj")