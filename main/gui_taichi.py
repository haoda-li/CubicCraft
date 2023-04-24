import taichi as ti
import igl
import sys, os
from cube_stylizer_gpu import CubeStylizer
ti.init(arch=ti.gpu)

input_mesh_path = sys.argv[1]
input_path_pre, input_ext = os.path.splitext(input_mesh_path)

save_img_idx = 0
save_mesh_idx = 0

paused = True
show_wireframe = False
obj_color = (255/255, 139/255, 91/255)


# load mesh and run one step for warm start
V, F = igl.read_triangle_mesh(input_mesh_path)
V /= max(V.max(axis=0) - V.min(axis=0))
V -= 0.5 * (V.max(axis=0) + V.min(axis=0))
cube = CubeStylizer(V=V, F=F)
cube.step()


# create window and load context
window = ti.ui.Window("Cube Craft", (1920, 1080), vsync=True, fps_limit=20)
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((1,1,1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, -2, 0)
camera.lookat(0, 0, 0)
    
def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.point_light(pos=(0.5, -1.5, 1.5), color=(1, 1, 1))
    scene.mesh(cube.U, cube.F, cube.normals, color=obj_color, show_wireframe=show_wireframe)
    canvas.scene(scene)
    
def get_options():
    global paused
    global obj_color
    global show_wireframe
    global save_result_name
    global save_img_idx
    global save_mesh_idx
    
    with gui.sub_window("Controls", 0.8, 0.00, 0.2, 0.5) as w:
        if w.button("Reset"):
            cube.reset()
        paused = w.checkbox("Paused", paused)
        show_wireframe = w.checkbox("show wireframe", show_wireframe)
        cube.cubeness[None] = w.slider_float("cubeness", cube.cubeness[None], 0., 2.)
        obj_color = w.color_edit_3("mesh color", obj_color)
        # if w.button("Save image"):
        #     output_path = f"{input_path_pre}_{save_img_idx}.jpg"
        #     window.save_image(output_path)
        #     save_img_idx += 1
        if w.button("Save mesh"):
            output_path = f"{input_path_pre}_output_{save_mesh_idx}.{input_ext}"
            cube.save_mesh(output_path)
            save_mesh_idx += 1
        

while window.running:
    if not paused:
        cube.step()
    render()
    get_options()
    
    window.show()