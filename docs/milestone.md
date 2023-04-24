# Cubic Craft

[Haoda Li](https://github.com/haoda-li), 
[Puyuan Yi](https://github.com/JamesYi2953), 
[Victor Li](https://github.com/weiji-li), 
[Zhen Jiang](https://github.com/Jz1116), 


In this project, we present a stylization tool to automatically manipulate 3D objects into a cubic style. Our tool uses a cubic stylization algorithm [@cubic_style] to cubify the object while preserving the geometric details. With our tool, 3D artists can create Minecraft-styled objects with ease. 


<figure markdown>
  ![](assets/cubic.jpg)
  <figcaption>Demonstration figure from Cubic Stylization</figcaption>
</figure>

## External Links

- [Our slides](https://docs.google.com/presentation/d/12iifKoNhjGInhJqSMDu6pAhNFBX3i4AgdjXz3iN-nas/edit?usp=share_link)
- [Our video](https://drive.google.com/file/d/1zCyl1HJOp3MiYYKhZTIQK5oc2nY-kC44/view?usp=share_link)

## Current Progress

- We have successfully finished our base-line algorithm of CPU-based cubic stylization. Given a mesh, our cubic craft algorithm
 stylize the object into a cubic shape. Therefore, the object have a cubic look. We did experiments based
 on several traditional meshes, such as bunny.obj and armadillo.obj. Here are the sample pictures for reference:

 <figure markdown>
  ![](assets/cubic.jpg)
  <figcaption>Demonstration figure from Cubic Stylization</figcaption>
</figure>

- We have created a GUI for users to directly interact with our implemented cubic craft algorithm. This GUI provides sliders 
for tuning the parameters, including cube orientation and "cubeness". In addition, this GUI has lots of basic
graphic functions such as changing the environment light and changing the mesh's material. Here are the sample pictures
for your reference:

<figure markdown>
  ![](assets/cubic.jpg)
  <figcaption>Demonstration figure from Cubic Stylization</figcaption>
</figure>

## Future works

- Our current cubic stylization is CPU-based, which takes a long time to run the algorithm. It will
take nearly 30 seconds to run 10 iterations on bunny mesh and take several minutes to run 10 iterations on
armadillo. To accelerate our algorithm, we will utilize Taichi to implement a GPU-based cubic stylization algorithm.

- Building handle functions. Based on this functions, users will also be able to set up handle points and deform the object by dragging the points.


## References
\bibliography