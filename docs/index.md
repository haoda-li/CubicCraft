# Cubic Craft

[Haoda Li](https://github.com/haoda-li), 
[Puyuan Yi](#), 
[XXXX](#), 
[XXXX](#), 


In this project, we present a stylization tool to automatically manipulate 3D objects into a cubic style. Our tool includes two parts: a cubic stylization algorithm [@cubic_style] to cubify the object while preserving the geometric details, and a voxelization algorithm to create a voxel representation. With our tool, 3D artists can create Minecraft-styled objects with ease. 


## Problem Description
Non-realistic modeling can provide a unique art style for animations and video games. One of the most popular area for non-realistic modeling is voxel art, which is the 3D version of pixel art. Although there are existing software to convert 3D objects into voxels, the voxels will preserve the object's geometric shape and is less interesting. In our project, we provide an additional cubic stylization algorithm to stylize the object into a cubic shape. Therefore, the object have a cubic look at both macro and micro level. 

## Goals and Deliverables

## Schedule

## Resources
- [Cubic Stylization](https://arxiv.org/pdf/1910.02926.pdf) [@cubic_style]
- [As-rigid-as-possible surface modeling](https://igl.ethz.ch/projects/ARAP/arap_web.pdf) [@arap]
- [The Computational Geometry Algorithms Library (CGAL)](https://www.cgal.org/)[@cgal] as the major software platform for CPU implementation. 
- [Taichi language](https://docs.taichi-lang.org/) [@hu2019taichi] and [Taichi voxel challenge](https://github.com/taichi-dev/voxel-challenge) for rendering the final results. If GPU optimization is possible, we can also implement cubic stylization as a Taichi kernel. 

\bibliography