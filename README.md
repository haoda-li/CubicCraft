# CS284A-cubic-craft

This project is a python implementation of <a href="https://www.dgp.toronto.edu/projects/cubic-stylization/">Cubic Stylization [Liu, Jacobson, 2019]</a>.

## requirements

```
pip install -r requirements.txt
```

## Execution

Inside the `main` dir, run

```
python gui_taichi.py [path to mesh.obj]
```

If no mesh is provided, the code will use default mesh _bunny.obj_.

## Experiments with GUI

- Use `Preset Camera View` to change to a specifc camera view that you like. You can also use keyboard **Q, W, E, A, S, D** to control the camera.
You can also show the wireframe of your mesh or change the default mesh color.

- Press `Enter deformation` button and left click loaded mesh to add new handle points. You can use **backspace** on your keyboard to delete handle points and
use **TAB** to switch to another handle point. One default handle point is already added and at least one handle point is needed
for cubic stylization.

- Use `Step` button to cubic stylize your mesh just one iteration or 
unclick the `paused` button to finish cubic stylization. You can change the `cubeness` parameter to generate
new mesh with different cubic-stylized extent.

- Rotate three axises to cubic stylize your mesh in different orientation.

- If you have at least two handle points, you can do experiments with as-rigid-as-possible deformation. Use keyboard
**UP/DOWN/LEFT/RIGHT/SHIFT/SPACE** to move your selected handle point and use **TAB** to switch to another handle point.

- Use `Save mesh` button to save your favourite cubic-stylized mesh!

![](docs/assets/GUI.png)
