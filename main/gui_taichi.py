import taichi as ti
import taichi.math as tim
import igl
import sys, os
from cube_stylizer_gpu import CubeStylizer, CoordinateHelper
ti.init(arch=ti.gpu)

input_mesh_path = sys.argv[1] if len(sys.argv) > 1 else "../meshes/bunny.obj"
input_path_pre, input_ext = os.path.splitext(input_mesh_path)

save_img_idx = 0
save_mesh_idx = 0

paused = True
show_wireframe = False
obj_color = (255/255, 139/255, 91/255)


# load mesh and run one step for warm start

cube = CubeStylizer(input_mesh_path)
coord_helper = CoordinateHelper()
cube.step()


# create window and load context
window = ti.ui.Window("Cube Craft", (1920, 1080), vsync=True, fps_limit=20)
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((1,1,1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, 2, 0)
camera.lookat(0, 0, 0)

  
def render():
    
    scene.set_camera(camera)
    scene.ambient_light((0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 3, 3), color=(1, 1, 1))
    scene.point_light(pos=(0.5, -1.5, 1.5), color=(1, 1, 1))
    scene.mesh(cube.U, cube.F, color=obj_color, show_wireframe=show_wireframe)
    coord_helper.rotate(cube.coordinate_angles[0], cube.coordinate_angles[1])
    scene.lines(coord_helper.V, 2, per_vertex_color=coord_helper.colors)
    canvas.scene(scene)
    
def controls():
    global paused
    global obj_color
    global show_wireframe
    global save_result_name
    global save_img_idx
    global save_mesh_idx
    
    camera.track_user_inputs(window, movement_speed=0.05, yaw_speed=4, pitch_speed=4, hold_key=ti.ui.RMB)
    
    with gui.sub_window("Controls", 0.8, 0.00, 0.2, 0.5) as w:
        if w.button("Reset"):
            cube.reset()
        paused = w.checkbox("Paused", paused)
        show_wireframe = w.checkbox("show wireframe", show_wireframe)
        w.text("Coefficients")
        cube.cubeness[None] = w.slider_float("cubeness", cube.cubeness[None], 0., 10.)
        w.text("Cube direction")
        cube.coordinate_angles[0] = w.slider_float("rho", cube.coordinate_angles[0], 0., 45.)
        cube.coordinate_angles[1] = w.slider_float("theta", cube.coordinate_angles[1], 0., 45.)
        
        obj_color = w.color_edit_3("mesh color", obj_color)
        if w.button("Save mesh"):
            output_path = f"{input_path_pre}_output_{save_mesh_idx}.{input_ext}"
            cube.save_mesh(output_path)
            save_mesh_idx += 1
            
            
        

while window.running:
    if not paused:
        cube.step()
    controls()
    render()
    
    window.show()