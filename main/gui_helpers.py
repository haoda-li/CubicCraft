import taichi as ti
import taichi.math as tim
import numpy as np
import igl

from collections import OrderedDict


@ti.data_oriented
class CoordinateHelper:
    def __init__(self, size=1.2) -> None:
        self.V = ti.Vector.field(n=3, shape=(6,), dtype=ti.f32)
        self.V.fill(0.)
        self.size = size
        self.colors = ti.Vector.field(n=3, shape=(6,), dtype=ti.f32)
        self.colors.from_numpy(np.array([
            [0.0, 0.5, 0.5], [1.0, 0.5, 0.5], 
            [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], 
            [0.5, 0.5, 0.0], [0.5, 0.5, 1.0], 
        ]))
        
 
    @ti.kernel
    def rotate(self, angle_x: float, angle_y: float, angle_z: float):
        rot_matrix = tim.rotation3d(
                tim.radians(angle_x), 
                tim.radians(angle_y), 
                tim.radians(angle_z)
            )
        self.V[0] = rot_matrix[:3, 0] * self.size
        self.V[1] = -rot_matrix[:3, 0] * self.size
        self.V[2] = rot_matrix[:3, 1] * self.size
        self.V[3] = -rot_matrix[:3, 1] * self.size
        self.V[4] = rot_matrix[:3, 2] * self.size
        self.V[5] = -rot_matrix[:3, 2] * self.size
        
        
        
        
@ti.data_oriented
class HandleHelper:
    def __init__(self, V, F) -> None: 
        self.handles = np.array([0])
        self.handle_pos_np = V.to_numpy()[[0]]
        self.handle_pos = ti.Vector.field(n=3, shape=(1, ), dtype=ti.f64)
        self.handle_pos[0] = V[0]
        self.active_handle = 0
        self.colors = ti.Vector.field(n=3, shape=(1, ), dtype=ti.f32)
        self.colors.fill(0.5)
        self.colors[self.active_handle] = (1, 0, 0)
        
        self.V = V
        self.F = F
        self.ray_o = ti.Vector.field(n=3, shape=(), dtype=ti.f64)
        self.ray_d = ti.Vector.field(n=3, shape=(), dtype=ti.f64)
        self.z_far = 1000.
        self.z_near = 0.1
        
    def _cast_ray(self, x, y, camera, window):
        width, height = window.get_window_shape()
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(width/height)
        points_proj = np.array([
            ((x - 0.5) * 2, (y - 0.5) * 2, 1, 1),
            ((x - 0.5) * 2, (y - 0.5) * 2, 0, 1)
        ]).T
        
        points = np.linalg.inv(proj.T @ view.T) @ points_proj
        points /= points[3, :]
        ray_d = points[:3, 1] - points[:3, 0]
        ray_d /= np.linalg.norm(ray_d)
        self.ray_o.from_numpy(points[:3, 0])
        self.ray_d.from_numpy(ray_d)
    
    @ti.kernel
    def _ray_mesh_intersect(self) -> ti.i32:
        ray_o = self.ray_o[None]
        ray_d = self.ray_d[None]
        NF = self.F.shape[0] // 3
        
        min_t = self.z_far
        min_vid = -1
        for i in range(NF):
            p1 = self.V[self.F[i * 3 + 0]]
            p2 = self.V[self.F[i * 3 + 1]]
            p3 = self.V[self.F[i * 3 + 2]]
            E1, E2, S = p2 - p1, p3 - p1, ray_o - p1
            S1, S2 = ray_d.cross(E2), S.cross(E1)
            M = 1. / S1.dot(E1)
            t = S2.dot(E2) * M
            b1 = S1.dot(S) * M
            b2 = S2.dot(ray_d) * M
            if t >= self.z_near and \
                b1 >= 0 and b1 <= 1 and \
                b2 >= 0 and b2 <= 2 and \
                b1 + b2 <= 1:
                
                old_min_t = ti.atomic_min(min_t, t)
                if (t < old_min_t):
                    b3 = 1 - b1 - b2
                    if b1 < b2 and b1 < b3:
                        min_vid = self.F[i * 3 + 0]
                    elif b2 < b1 and b2 < b3:
                        min_vid = self.F[i * 3 + 1]
                    else:
                        min_vid = self.F[i * 3 + 2]
        return min_vid
    
    def add_handle(self, x, y, camera, window):
        self._cast_ray(x, y, camera, window)
        vid = self._ray_mesh_intersect()
        if vid != -1 and vid not in self.handles:
            self.handles = np.concatenate((self.handles, [vid]))
            new_shape = len(self.handles)
            self.handle_pos = ti.Vector.field(n=3, shape=(new_shape, ), dtype=ti.f64)
            self.handle_pos_np = np.empty((new_shape, 3), order="F")
            for i, handle in enumerate(self.handles):
                self.handle_pos_np[i] = self.V[handle].to_numpy()
            self.handle_pos.from_numpy(self.handle_pos_np)
            self.active_handle = new_shape - 1
            self.colors = ti.Vector.field(n=3, shape=(new_shape, ), dtype=ti.f32)
            self.colors.fill(0.5)
            self.colors[self.active_handle] = (1, 0, 0)
            
    def delete_handle(self):
        if len(self.handles) == 1:
            return
        
        self.handles = np.concatenate((
            self.handles[:self.active_handle], self.handles[self.active_handle + 1:]
            ))
        new_shape = self.handle_pos.shape[0] - 1
        new_shape = len(self.handles)
        self.handle_pos = ti.Vector.field(n=3, shape=(new_shape, ), dtype=ti.f64)
        self.handle_pos_np = np.empty((new_shape, 3), order="F")
        for i, handle in enumerate(self.handles):
            self.handle_pos_np[i] = self.V[handle].to_numpy()
        self.handle_pos.from_numpy(self.handle_pos_np)
        self.active_handle = new_shape - 1
        self.colors = ti.Vector.field(n=3, shape=(new_shape, ), dtype=ti.f32)
        self.colors.fill(0.5)
        self.colors[self.active_handle] = (1, 0, 0)
        
    def next_handle(self):
        self.colors[self.active_handle] = (0.5, 0.5, 0.5)
        self.active_handle += 1
        if self.active_handle >= len(self.handles):
            self.active_handle = 0
        self.colors[self.active_handle] = (1, 0, 0)
        
    def move_handle(self, dx, dy, dz):
        self.handle_pos[self.active_handle][0] += dx
        self.handle_pos_np[self.active_handle][0] += dx
        self.handle_pos[self.active_handle][1] += dy
        self.handle_pos_np[self.active_handle][1] += dy
        self.handle_pos[self.active_handle][2] += dz
        self.handle_pos_np[self.active_handle][2] += dz
        
    def reset_positions(self):
        for i, handle in enumerate(self.handles):
            self.handle_pos_np[i] = self.V[handle].to_numpy()
        self.handle_pos.from_numpy(self.handle_pos_np)
    
    def get_handles(self):
        return self.handles, self.handle_pos_np