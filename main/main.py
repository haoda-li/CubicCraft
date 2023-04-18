import numpy as np
import open3d as o3d


import argparse

from utils import normalize_unitbox, cube_style_precomputation, cube_style_single_iteration


parser = argparse.ArgumentParser(description='The algorithm of cubic stylization')
parser.add_argument("--path", type=str, default="../GUI/bunny.ply")
parser.add_argument('--Lambda', type=float, default=0.20)

class cube_style_data():
    def __init__(self, Lambda=0.0, rhoInit=1e-3, ABSTOL=1e-6, RELTOL=1e-3,
        mu=10, tao=2, maxIter_ADMM=100, objVal=0, reldV=float("inf")
                 ):

        # user should tune these parameters
        self.Lambda = Lambda
        self.rhoInit = rhoInit
        self.ABSTOL = ABSTOL
        self.RELTOL = RELTOL

        # usually these parameters do not need to tune
        self.mu = mu
        self.tao = tao
        self.maxIter_ADMM = maxIter_ADMM
        self.objVal = objVal
        self.reldV = reldV

        self.objHis = [] # std::vector<double>
        self.hEList = [] # std::vector<Eigen::MatrixXi>
        self.dVList = [] # std::vector<Eigen::MatrixXd>
        self.UHis = [] # std::vector<Eigen::MatrixXd>
        self.WVecList = [] #std::vector<Eigen::VectorXd>

        self.K = None #Eigen::SparseMatrix<double>
        self.L = None  #Eigen::SparseMatrix<double>
        self.N = None #Eigen::MatrixXd
        self.VA = None  # Eigen::MatrixXd
        self.zAll = None  # Eigen::MatrixXd
        self.uAll = None  # Eigen::MatrixXd
        self.rhoAll = None # Eigen::VectorXd
        self.objValVec = None # Eigen::VectorXd

        self.bc = None # Eigen::MatrixXd
        self.b = None # Eigen::VectorXi

        # TODO: igl::min_quad_with_fixed_data<double> solver_data;

        #for plane constraints
        self.bx = None # Eigen::VectorXi
        self.by = None #Eigen::VectorXi
        self.bz = None # Eigen::VectorXi
        # TODO: igl::min_quad_with_fixed_data<double> solver_data_x, solver_data_y, solver_data_z;
        self.xPlane = 0.0
        self.yPlane = 0.0
        self.zPlane = 0.0

    def reset(self):
        # user should tune these parameters
        self.ABSTOL = 1e-5
        self.rhoInit = 1e-3
        self.RELTOL = 1e-3

        # usually these parameters do not need to tune
        self.mu = 10
        self.tao = 2
        self.maxIter_ADMM = 100
        self.objVal = 0
        self.reldV = float("inf")

        self.objHis = []  # std::vector<double>
        self.hEList = [] # std::vector<Eigen::MatrixXi>
        self.dVList = [] # std::vector<Eigen::MatrixXd>
        self.UHis = [] # std::vector<Eigen::MatrixXd>
        self.WVecList = [] #std::vector<Eigen::VectorXd>

        self.K = None  # Eigen::SparseMatrix<double>
        self.L = None  # Eigen::SparseMatrix<double>
        self.N = None  # Eigen::MatrixXd
        self.VA = None  # Eigen::MatrixXd
        self.zAll = None  # Eigen::MatrixXd
        self.uAll = None  # Eigen::MatrixXd
        self.rhoAll = None  # Eigen::VectorXd
        self.objValVec = None  # Eigen::VectorXd

        # TODO: igl::min_quad_with_fixed_data<double> solver_data;


def main():
    # get parameter from args
    args = parser.parse_args()
    obj_path = args.path
    Lambda = args.Lambda

    # load provided mesh

    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
        print("\033[1m[INFO] successfully load your mesh ... \033[0m")
    except:
        print("\033[1m[INFO] your provided path is wrong ...\033[0m")
        print("\033[1m[INFO] Use default mesh instead ...\033[0m")
        bunny = o3d.data.BunnyMesh()
        mesh = o3d.io.read_triangle_mesh(bunny.path)


    # load vertices, triangle and lambda
    V = np.asarray(mesh.vertices).astype(np.float32)
    F = np.asarray(mesh.triangles)

    data = cube_style_data(Lambda=Lambda)

    V = normalize_unitbox(V)

    # Calculate mean of columns of V
    meanV = np.mean(V, axis=0)
    V = V - meanV
    U = V

    # Set a constrained point F(0, 0)
    data.bc = V[F[0][0]]
    data.b = F[0][0]

    #precomputation ARAP and initialize ADMM parameters
    print("\033[1m[INFO] precomputation ARAP and initialize ADMM parameters ...\033[0m")
    V, F, data = cube_style_precomputation(V, F, data)

    #cubic stylization
    maxIter = 1000
    stopReldV = 1e-3 # stopping criteria for relative displacement
    print("\033[1m[INFO] cubic stylization start ...\033[0m")
    for i in range(maxIter):
        print(f"iteration: {i}\n")
        cube_style_single_iteration(V, U, data)
        if data.reldV < stopReldV:
            break
    print("\033[1m[INFO] cubic stylization end ...\033[0m")

    # create the mesh after cubic_stylization
    mesh_ac = o3d.geometry.TriangleMesh() # ac means after cubic
    mesh_ac.vertices = o3d.utility.Vector3dVector(U)
    mesh_ac.triangles = o3d.utility.Vector3iVector(F)
    mesh_ac.compute_vertex_normals()
    mesh_ac.compute_triangle_normals()

    # visualize our cubic mesh
    o3d.visualization.draw([mesh_ac])

    # write output mesh
    o3d.io.write_triangle_mesh("result.ply", mesh)

    pass


if __name__ == "__main__":
    main()