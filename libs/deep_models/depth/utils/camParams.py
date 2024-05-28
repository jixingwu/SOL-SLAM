import numpy as np

def get_intrinsics_params(dataset_name):
    K, inv_K = np.eye(4), np.eye(4)
    if dataset_name == 'kitti_odom':
        K = np.array([7.070912e+02, 0.000000e+00, 6.018873e+02,
                      0.000000e+00, 7.070912e+02, 1.831104e+02,
                      0.000000e+00, 0.000000e+00, 1.000000e+00])
        K = K.reshape(3,3).astype(np.float32)
        inv_K = np.linalg.inv(K).astype(np.float32)
    elif dataset_name == 'rw':
        K = np.array([992.27732, 0.0, 1050.63971, 0.0, 986.94773, 755.0132, 0.0, 0.0, 1.0])
        K = K.reshape(3,3).astype(np.float32)
        inv_K = np.linalg.inv(K).astype(np.float32)
    elif dataset_name == 'tum-2':
        K = np.array([520.908620, 0.000000e+00, 325.141442,
                      0.000000e+00, 521.007327, 249.701764,
                      0.000000e+00, 0.000000e+00, 1.000000e+00])
        K = K.reshape(3,3).astype(np.float32)
        inv_K = np.linalg.inv(K).astype(np.float32)
    elif dataset_name == 'tum-3':
        K = np.array([535.4, 0.000000e+00, 320.1,
                        0.000000e+00, 539.2, 247.6,
                        0.000000e+00, 0.000000e+00, 1.000000e+00])
        K = K.reshape(3,3).astype(np.float32)
        inv_K = np.linalg.inv(K).astype(np.float32)
    else:
        assert False and 'unknown dataset type'

    return K, inv_K

def get_coeff_params(dataset_name):
    if dataset_name == 'tum-2':
        coeff = np.array([0.231222, -0.784899, -0.003257, -0.000105, 0.917205])
    else:# kitti tum3
        coeff = np.array([0,0,0,0,0])
        
    return coeff