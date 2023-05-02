# CubicCraft: A Mesh Stylization Tool

[Haoda Li](https://github.com/haoda-li), 
[Puyuan Yi](https://github.com/JamesYi2953), 
[Victor Li](https://github.com/weiji-li), 
[Zhen Jiang](https://github.com/Jz1116), 

[Download SIGGRAPH Styled Report :fontawesome-solid-file-pdf:](./assets/cubic_craft.pdf){ .md-button .md-button--primary } 

## Abstract
We present a stylization tool to automatically manipulate triangle meshes into a cubic style. Our tool uses a cubic stylization algorithm[@cubic_style] to cubify the user's provided meshes. The algorithm extends the as-rigid-as-possible energy [@arap] with an additional L1 regularization, hence can work seamlessly with ADMM optimization. Cubic stylization works only on the vertex positions, hence preserving the geometrical details and topology. In addition, we implemented the algorithm with GPU acceleration and achieves real-time interactive editing. We also created a user-friendly interaction surface to let users easily change the algorithm's hyperparameter and cubify their own mesh. With our tool, 3D artists can create Minecraft-styled objects with ease.


<figure markdown>
  ![](assets/teaser.jpg){ width="1080" }
  <figcaption>Cubic Craft turns triangle meshes (grey) into cubic-styled meshes (green)</figcaption>
</figure>

## Introduction
With the increasing availability of image stylization filters and non-photorealistic rendering techniques, creating artistic images has become much more accessible to non-professional users. However, the direct stylization of 3D shapes and non-realistic modeling has not yet been given as much attention. Despite the advancements in technology, professional industries like visual effects and video games still rely on trained modelers to meticulously create non-realistic geometric assets. This is because exploring geometric styles presents a greater challenge, as it involves dealing with arbitrary topologies, curved metrics, and non-uniform discretization. While image stylization tools have made it easier to generate artistic imagery, there is still a lack of effective tools for generating artistic geometry, which remains a major obstacle to the development of geometric stylization.

The focus of this paper is on a specific style of sculpture, namely the cubic style. This style has been prevalent throughout art history (ancient sculptures) and modern game history (Minecraft). In this work, we have developed a stylization tool _cubic stylization_ based on GPU that takes a 3D shape as input and outputs a deformed shape that has the same style as cubic sculptures. This tool is aimed at helping artists and designers achieve the cubic style more easily and efficiently, while also providing a new way to explore and experiment with this timeless artistic tradition.

Our implemented method _cubic stylization_ formulates the task as an energy optimization problem, which preserves the geometric details of a shape while transforming it into a cubic form. Specifically, this energy function combines an as-rigid-as-possible (ARAP) energy with a special L1 regularization. This energy can be minimized efficiently using the local-global approach with the Alternating Direction Method of Multipliers (ADMM). This method has strong flexibility that allowing artists and designers to achieve a wide range of stylistic variations within the cubic style, providing them with greater creative freedom and expressive potential.

Our main contributions are summarized as follows:

- We implemented GPU-accelerated _cubic stylization_ algorithm, which allows real-time cubic style object generation and rendering.
- We created a user-friendly GUI to interact with our algorithm. Users can easily change the hyperparameter of the algorithm and observe different results.


## Related Works
In this section, our primary focus is on exploring methods for processing geometry. Specifically, we will be discussing various techniques for studying geometric styles and deformation methods that share common technical similarities. Our aim is to provide a comprehensive overview of these methods and their applications and to highlight their significance in the field of geometry processing.

### Shape Deformation
The subfield of shape deformation has been the subject of considerable research in computer graphics and related fields. Several notable works have contributed to this area, including the as-rigid-as-possible (ARAP) energy model [@arap], which has become a popular method for shape deformation due to its ability to preserve the rigid structure of the object being deformed. Another important contribution is the Laplacian-based deformation technique [@sorkine2004lscm], which uses the Laplacian operator to deform a shape while preserving its surface details. In addition, several works have explored the use of physically-based deformation models, such as the finite-element method and the mass-spring system [@nealen2006physically].

More recent work in shape deformation has focused on extending these techniques to handle more complex shapes and deformation scenarios. For example, the Cage-based deformation method [@le2015cage] uses a cage mesh to define the deformation space, allowing for more intuitive and flexible deformation. Other works have explored the use of machine learning techniques, such as neural networks, for shape deformation [@li2017deepdeform].

The subfield of shape deformation has made significant contributions to the field of computer graphics and related disciplines and continues to be an active area of research with numerous exciting directions for future exploration. Our cubic stylization is also a specific kind of shape deformation that combines ARAP energy term and a L1 regularization term.

<figure markdown>
  ![](assets/uv_texture.jpg){ width="1080" }
  <figcaption>Shape deformation preserves vertex attributes</figcaption>
</figure>

### Different Geometric Style
Research on different geometric styles has been a topic of interest in computer graphics and related fields. Two main approaches have been taken to explore this area: discriminative geometric styles and generative geometric styles.

Discriminative geometric styles focus on identifying and analyzing different styles in existing geometry. For example, the work by Kim et al. [@kim2013discriminative] proposes a method for identifying and characterizing different geometric styles in furniture design, while the work by Zhou et al. [@zhou2016discriminative] focuses on identifying different styles in fashion design.

On the other hand, generative geometric styles aim to create new geometry in a particular style. One notable example is the work by Huang et al.  [@huang2018generative], which proposes a method for generating 3D models in a particular style using a generative adversarial network (GAN). Another example is the work by Kalogerakis et al. [@kalogerakis2012probabilistic], which uses a probabilistic model to generate 3D shapes with a particular style.

Overall, research on different geometric styles has led to a deeper understanding of the principles and characteristics of different styles in various domains, as well as the development of methods for creating new geometry in a particular style. These techniques have potential applications in a variety of fields, such as architecture, product design, and entertainment.


## Method

Our method is based on cubic stylization [@cubic_style], a method to deform the input mesh into a cubic stylized mesh. Generally, the method adds a new L1 regularization on the deformation with As-rigid-as-possible (ARAP) [@arap] energy optimization. By regularizing each vertex normal to align with the axis, the mesh can have a cubic style, while maintaining the local geometric details.

The problem description is illustrated as follows: given a triangle mesh $S = (V, F)$ and a set of constraints on vertex positions. We want to output a deformed shape $\tilde V$. The output shape will have each sub-component in the style of axis-aligned cubes and will retain the geometric details of the original mesh. 

We will describe our method and implementation in the following sections: In [As-rigid-as-possible Deformation section](#as-rigid-as-possible-deformation), we will talk about the As-rigid-as-possible (ARAP) Deformation [@arap], and elaborate on its energy functions. Then we will talk about cubic stylization [@cubic_style] in [Cubic Stylization section](#cubic-stylization). The implementation part is in [Implementation section](#implementation).


### As-rigid-as-possible Deformation


<figure markdown>
  ![](assets/deformation.jpg){ width="720" }
  <figcaption>Cubified Mesh with as-rigid-as-possible deformation</figcaption>
</figure>

The thought of ARAP energy is very intuitive: given the cell $C_i$ corresponding to vertex $i$, and its deformed version $\tilde C_i$, ARAP defines the approximate rigid transformation between the two cells by observing the edges emanating from the vertex $i$ in $S$ and $\tilde S$, where $S$ and $\tilde S$ denote the original triangle mesh and the deformed triangle mesh. Note that $\tilde S$ should have the same connectivity as $S$. If the deformation $C_i \rightarrow \tilde C_i$ is rigid, there must exists a rotation matrix $R_i$ such that:

$$\tilde V_i - \tilde V_j = R_i(V_i - V_j),  \forall j \in \mathcal N(i)$$ 

$ \mathcal N(i)$ denotes the set of vertices connected to vertex $i$, also called the one-ring neighbors. 

When the deformation is not rigid, we can still find the best
approximating rotation matrix $R_i$ that fits the above equations in a weighted least squares sense,  i.e., minimizes

$$E(C_i, \tilde C_i) = \sum_{i\in V}\sum_{j\in \mathcal N(i)} w_{ij} \|R_i d_{ij} -  \tilde{d_{ij}}\|_F^2$$

where $w_{ij}$ is the cotangent weight [@pinkall1993computing] between vertex $i$ and vertex $j$, and $d_{ij} = V_i - V_j$ and $\tilde d_{ij} = \tilde V_i - \tilde V_j$. What we need is to solve for vertex position $\tilde V_i$ and per-vertex rotations $R_i$ that minimizes the energy function above. 

For deformation, we are given user-defined constraints on some vertex positions and we need to update all other vertices to minimize the energy. [@arap] uses alternating minimization strategy. For each iteration, we first fix vertex positions to find the optimal rotations for each vertex, and then fix the rotations to update vertex positions. The rotation updates only depends on the one-ring neighbor for each vertex, hence we call it a local step. We will talk more details about local steps in the next section. The vertices update, or the global step, can be directly derived by setting the partial derivative w.r.t. each vertex position to $0$. Eventually, we need to solve a system of $3N$ equations of $3N$ unknowns, where each vertex corresponds to the equation 

$$\sum_{j\in \mathcal N(i)} w_{ij} \tilde{d_{ij}} = \sum_{j\in \mathcal N(i)} \frac{w_{ij}}{2} (R_i+R_j)d_{ij}$$


### Cubic Stylization

In this section, we will illustrate the cubic stylization algorithm. Intuitively, an object is cubic style if its normals are aligned with the three dominant directions. Therefore, [@cubic_style] proposed an additional L1 regularization term on the rotated normal. Combining with the ARAP energy, the full energy term is listed as follows:

$$E(C_i, C_i^{\prime}) = \sum_{i\in V}\sum_{j\in \mathcal N(i)} \frac{w_{ij}}{2}\|R_id_{ij} - \tilde d_{ij}\|_F^2 + \lambda a_i \|R_i\tilde n_i\|_1$$

In the L1 regularization term, $\hat n_i$ denotes the area-weighted unit normal vector of $v_i$ and $a_i$ is the barycentric area of $v_i$. and $\lambda$ is the "cubeness" parameter.

The local step involves finding the rotation matrix $R_1,\cdots, R_n$, for each vertex $i$, we are to optimize:

$$R_i^* = \arg\min_{R_i\in SO(3)}\sum_{j\in \mathcal N(i)} \frac{w_{ij}}{2}\|R_id_{ij} - \tilde d_{ij}\|_F^2 + \lambda a_i \|R_i\tilde n_i\|_1$$

note that the ARAP energy can be expressed in matrix formations 

$$\frac12 (R_iD_i-\tilde D_i)^T W_i (R_iD_i-\tilde D_i) = \frac12\|R_iD_i-\tilde D_i\|_{W_i}^2$$

where $D_i,\tilde D_i \in \mathbb R^{3\times |\mathcal N(i)|}$ are stacked rim/spoke edge vectors and $W_i$ is the diagonal matrix of $w_1,...,w_n$. Then, write $z = R_i\hat n_i$, we can turn the formation into

\begin{align*}
\text{minimize}_{z_, R_i} \quad &\frac12 \|R_iD_i-\tilde D_i\|_{W_i}^2+\lambda a_i\|z\|_1\\
\text{subject to} \quad &z-R_i\hat n_i = 0
\end{align*}

Now We can solve the local step using the alternating direction method of multipliers (ADMM) updates [@admm}. Applying ADMM, the update steps are

\begin{align*}
R_i^{k+1} &= \arg\min \frac12\|R_iD_i-\tilde D_i\|_{W_i}^2 + \frac{\rho^k}2\|R_i\hat n_i - z^k + u^k\|_2^2\\
z^{k+1} &= \arg\min \lambda a_i \|z\|_1 + \frac{\rho^k}2\|R_i^{k+1} \hat n_i - z + u^k\|_2^2\\
\tilde u^{k+1} &= u^k + R_i^{k+1} \hat n_i - z^{k+1}\\
\rho^{k+1}, u^{k+1} &= \text{update}(\rho^k)   
\end{align*}

Then, consider each update, The rotation update can be viewed as 


$$R_i^{k+1} = \arg\max tr(R_i M_i)$$

$$M_i = \begin{bmatrix}[D_i]&[\hat n_i]\end{bmatrix}
\begin{bmatrix}[W_i]&0\\0&\rho^k\end{bmatrix}
\begin{bmatrix}[\tilde D_i]\\ [(z^k-u^k)^T] \end{bmatrix}$$

This becomes an Orthogonal Procrustes problem, and the solution is given through single value decomposition

$$M = U\Sigma V^T, R = UV^T$$

up to $\det(R) > 0$ by alternating the sign of $U$'s column. The $z$ update is an instance of lasso problem, which can be solved with a shrinkage step

$$z^{k+1} = S_{\lambda a_i/\rho^k}(R_i^{k+1}\hat n_i + u^k)$$

where the shrinkage is defined as 

$$S_\chi(x_j) = (1-\frac{\chi}{|x_j|}) + x_j$$

Hence we solve the local step. Then, we notice that L1 term $\lambda a_i\|R_i\tilde n_i\|_1$ is independent of the vertex positions $V$. Therefore, the global step is exactly the same as ARAP energy optimization.

<figure markdown>
  ![](assets/lambdas.jpg){ width="1080" }
  <figcaption>Meshes with different cubeness</figcaption>
</figure>

### Implementation
We implement the cubic stylization [@cubic_style] algorithm using Python and [`libigl`](https://github.com/libigl/libigl-python-bindings) [@libigl]. We follow [@cubic_style]'s implementation and set the initial $\rho = 10^{-4}, \mu=50, \tau=2$. In addition, we observe that the local step updates each vertex independently, providing opportunities for parallelization. We use [`Taichi`](https://github.com/taichi-dev/taichi) [@hu2019taichi] to implement a GPU-accelerated version. To maximize parallelism, for each local step, we run a fixed number of ADMM iterations instead of using the stopping criteria. We set the initial ADMM iterations to $50$ and reduce it to $5$ through the steps. From experiments, we found that this strategy is adequate for convergence. 

Compared to the CPU implementation [@cubic_style], our implementation gradually accelerated the local step computation. We tested our implementation on an `AMD R9 5900HS CPU` with a `NVIDIA 3050ti GPU` and listed the performance below. 


| Mesh name | Num. vertices | CPU time (s) | GPU time (s) |
| --------- | ------------- | ------------ | ------------ |
| homer     | 6002          | 10.03        | 3.39         |
| bunny     | 6172          | 27.56        | 3.55         |
| owl       | 39416         | 160.52       | 6.69         |
| horse     | 48485         | 211.62       | 8.25         |
| armadillo | 49990         | 217.96       | 7.80         |
| dragon    | 62472         | 335.71       | 9.16         |

## User Interface

<figure markdown>
  ![](assets/gui.jpg){ width="720" }
  <figcaption>Our GUI, the user can view the mesh deformation progress and change parameters</figcaption>
</figure>

We provide a graphical interface for the users to visualize and easily edit the meshes. The graphical interface is based on the GUI system provided by `Taichi`. Given a triangle mesh, our graphical interface allows the user to change the parameters in the algorithm, visualize the deformations, and save the resulting mesh. 

In our GUI, the user can directly change the `cubeness` parameter and observe different results in real-time. Note that our algorithm only changes the vertex position, hence other local geometric information, such as texture coordinates, is preserved. In addition to the `cubeness` parameter, we notice that cube stylization is orientation dependent. The cubeness is achieved by forcing all vertex normals to align with the three standard axes. If we rotate the input mesh, the output shape will be different. Note that the same effect can be achieved by applying a coordinate transformation on all vertex normals. Therefore, we add the `coordinate rotation` parameters so that users can have different cube orientations. 
<figure markdown>
  ![](assets/orientation.jpg){ width="720" }
  <figcaption>Meshes with different cube orientation</figcaption>
</figure>

Similar to [@arap]'s approach, we can put constraints on vertex positions. Users can utilize our GUI to add handle points and move the handle points to perform as-rigid-as-possible deformation. Note that natural deformations are obtained because the optimization automatically produces the correct local rotations for each vertex. 


## Conclusion
In conclusion, our work presents a powerful tool for cubic stylization that enables 3D artists to create Minecraft-styled objects with ease. Our algorithm, which extends the as-rigid-as-possible energy with an L1 regularization, works seamlessly with ADMM optimization and preserves the underlying geometrical details and topology of the mesh. Furthermore, our implementation with GPU acceleration allows for real-time interactive editing, making the tool both efficient and intuitive to use.

Overall, our work contributes to the growing field of geometry processing by presenting a novel approach to stylization. The ability to manipulate and transform meshes in a cubic style has significant potential for a range of applications, including architectural design, game development, and animation. We believe that our tool will be particularly valuable to 3D artists who wish to create unique and visually striking objects quickly and efficiently.

## References