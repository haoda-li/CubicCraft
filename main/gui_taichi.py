import taichi as ti
import taichi.math as tim
import igl
import sys, os
from cube_stylizer_gpu import CubeStylizer 
from gui_helpers import CoordinateHelper, HandleHelper
ti.init(arch=ti.gpu)

input_mesh_path = sys.argv[1] if len(sys.argv) > 1 else "../meshes/bunny.obj"
input_path_pre, input_ext = os.path.splitext(input_mesh_path)

save_img_idx = 0
save_mesh_idx = 0

paused = True
editing_handles = True
show_wireframe = False
show_coord_helper = True
show_handle_helper = True
obj_color = (255/255, 139/255, 91/255)
obj_color_paused = (107/255, 110/255, 95/255)


# load mesh and run one step for warm start

cube = CubeStylizer(input_mesh_path)
coord_helper = CoordinateHelper()
handle_helper = HandleHelper(cube.V, cube.F)

# create window and load context
window = ti.ui.Window("Cube Craft", (1920, 1080), vsync=True, fps_limit=20)
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((1,1,1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.z_near(0.1)
camera.z_far(1000.)

preset_positions = [
    (0, 0, 2),
    (0, 2, 0),
    (0, -2, 0),
    (2, 0, 0),
    (-2, 0, 0),
]
preset_up = [
    (0, 1, 0), 
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 0)
]
preset_idx = 0
camera.position(0, 0, 2)
camera.up(0, 1, 0)
camera.lookat(0, 0, 0)


def render():
    scene.ambient_light((0.3, 0.3, 0.3))
    scene.point_light(pos=(2, 2, 0), color=(1, 1, 1))
    scene.point_light(pos=(0, 0, 2), color=(1, 1, 1))
    scene.point_light(pos=(-2, 0, -2), color=(1, 1, 1))
    scene.point_light(pos=(-1, -2, -1), color=(1, 1, 1))
    if editing_handles:
        scene.mesh(cube.V, cube.F, color=obj_color_paused, show_wireframe=show_wireframe)
    else:
        scene.mesh(cube.U, cube.F, color=obj_color, show_wireframe=show_wireframe)
    if show_handle_helper:
        scene.particles(handle_helper.handle_pos, 0.01, per_vertex_color=handle_helper.colors)
    coord_helper.rotate(cube.coordinate_angles[0], cube.coordinate_angles[1], cube.coordinate_angles[2])
    if show_coord_helper:
        scene.lines(coord_helper.V, 2, per_vertex_color=coord_helper.colors)
    canvas.scene(scene)
    
def controls():
    global paused
    global obj_color
    global obj_color_paused
    global show_wireframe
    global save_result_name
    global save_img_idx
    global save_mesh_idx
    global editing_handles
    global show_coord_helper
    global show_handle_helper
    global preset_idx
    
    camera.track_user_inputs(window, movement_speed=0.05, yaw_speed=4, pitch_speed=4, hold_key=ti.ui.RMB)
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.BACKSPACE and editing_handles: 
            handle_helper.delete_handle()
        if window.event.key == ti.ui.LMB and editing_handles: 
            x, y = window.get_cursor_pos()
            handle_helper.add_handle(x, y, camera, window)
        if window.event.key == ti.ui.TAB: 
            handle_helper.next_handle()
            
    if not editing_handles and show_handle_helper:
        if window.is_pressed(ti.ui.SHIFT):
            handle_helper.move_handle(0, 0, -0.01)
        if window.is_pressed(ti.ui.SPACE):
            handle_helper.move_handle(0, 0, 0.01)
        if window.is_pressed(ti.ui.DOWN):
            handle_helper.move_handle(0, -0.01, 0)
        if window.is_pressed(ti.ui.UP):
            handle_helper.move_handle(0, 0.01, 0)
        if window.is_pressed(ti.ui.LEFT):
            handle_helper.move_handle(-0.01, 0, 0)
        if window.is_pressed(ti.ui.RIGHT):
            handle_helper.move_handle(0.01, 0, 0)
        
    with gui.sub_window("Deformation", 0.8, 0.0, 0.2, 0.2) as w_deform:
        if editing_handles:
            if w_deform.button("Enter deformation"):
                editing_handles = False
            w_deform.text("LEFT CLICK mesh\n  add new constraint handle\nBACKSPACE\n  Delete current handle (Red)\nTAB\n  Switch to next handle\n\n", color=(.5, .5, .5))
        else:
            if w_deform.button("Edit handles"):
                handle_helper.reset_positions()
                editing_handles = True
            if paused:
                if w_deform.button("Step"):
                    cube.step(*handle_helper.get_handles())
            else:
                cube.step(*handle_helper.get_handles())
                
            if w_deform.button("Restart Optimization"):
                cube.reset()
            paused = w_deform.checkbox("Paused", paused)
            show_handle_helper = w_deform.checkbox("show handle points", show_handle_helper)
            w_deform.text("UP/DOWN/LEFT/RIGHT/SHIFT/SPACE\n  move current handle along x/y/z in world space\nTAB\n  Switch current handle\n\n", color=(.5, .5, .5))
            
            
    with gui.sub_window("Cubic Stylization", 0.8, 0.2, 0.2, 0.3) as w_cube:
        cube.cubeness[None] = w_cube.slider_float("cubeness", cube.cubeness[None], 0., 10.)
        w_cube.text("\n\nCube direction")
        cube.coordinate_angles[0] = w_cube.slider_float("rotate x", cube.coordinate_angles[0], -45., 45.)
        cube.coordinate_angles[1] = w_cube.slider_float("rotate y", cube.coordinate_angles[1], -45., 45.)
        cube.coordinate_angles[2] = w_cube.slider_float("rotate z", cube.coordinate_angles[2], -45., 45.)
        show_coord_helper = w_cube.checkbox("show cube directions", show_coord_helper)
        
    with gui.sub_window("Misc", 0.8, 0.5, 0.2, 0.2) as w_display:
        show_wireframe = w_display.checkbox("show wireframe", show_wireframe)
        if editing_handles:
            obj_color_paused = w_display.color_edit_3("mesh color", obj_color_paused)
        else:
            obj_color = w_display.color_edit_3("mesh color", obj_color)
        if w_display.button("Save mesh"):
            output_path = f"{input_path_pre}_output_{save_mesh_idx}.{input_ext}"
            cube.save_mesh(output_path)
            save_mesh_idx += 1
        if w_display.button("Preset Camera View"):
            preset_idx = preset_idx + 1
            if preset_idx == len(preset_positions):
                preset_idx = 0
            camera.position(*preset_positions[preset_idx])
            camera.up(*preset_up[preset_idx])
            camera.lookat(0, 0, 0)
        w_display.text("Mesh will be saved to the input_mesh's path", color=(.5, .5, .5))
        w_display.text("Use\n  Mouse right drag\n  Q W E\n  A S D\nTo move camera", color=(.5, .5, .5))
            
            
        

while window.running:
    scene.set_camera(camera)
    controls()
    render()
    
    window.show()