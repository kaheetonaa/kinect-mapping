import freenect
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from PIL import Image
import cv2
import time
DepthCamParams = {
        "fx": 5.9421434211923247e+02,
        "fy": 5.9104053696870778e+02,
        "cx": 3.3930780975300314e+02,
        "cy": 2.4273913761751615e+02,
        "k1": -2.6386489753128833e-01,
        "k2": 9.9966832163729757e-01,
        "p1": -7.6275862143610667e-04,
        "p2": 5.0350940090814270e-03,
        "k3": -1.3053628089976321e+00,
        "a": -0.0030711016,
        "b": 3.3309495161,
    }

    # RGB cam parameters
RGBCamParams = {
        "fx": 5.2921508098293293e+02,
        "fy": 5.2556393630057437e+02,
        "cx": 3.2894272028759258e+02,
        "cy": 2.6748068171871557e+02,
        "k1": 2.6451622333009589e-01,
        "k2": -8.3990749424620825e-01,
        "p1": -1.9922302173693159e-03,
        "p2": 1.4371995932897616e-03,
        "k3": 9.1192465078713847e-01,
        "rot": np.array([[9.9984628826577793e-01,1.2635359098409581e-03,-1.7487233004436643e-02],
                         [-1.4779096108364480e-03,9.9992385683542895e-01, -1.2251380107679535e-02],
                         [1.7470421412464927e-02, 1.2275341476520762e-02,
9.9977202419716948e-01]]),
        "trans": np.array([[ 1.9985242312092553e-02, -7.4423738761617583e-04,
-1.0916736334336222e-02]])
    }
#parameters taken from https://nicolas.burrus.name/oldstuff/kinect_calibration/
def depth_cam_mat():
        '''
        Returns camera matrix, including the transformation values of depth to meters
        :return: camera (intrisec) matrix
        '''
        mat = np.array([[1 / DepthCamParams['fx'], 0, 0, -DepthCamParams['cx'] / DepthCamParams['fx']],
                        [0, 1 / DepthCamParams['fy'], 0, -DepthCamParams['cy'] / DepthCamParams['fy']],
                        [0, 0, 0, 1],
                        [0, 0, DepthCamParams['a'], DepthCamParams['b']]])

        print(mat)
        return mat

def get_registred_depth_rgb(depth,rgb):
        '''
        Returns the registred pointclaud and image with transforming the cameras position in world coordinate system
        :return: registred point cloud and image
        '''
        rgb=rgb
        depth = np.array(depth, dtype=np.float32)
        h,w=rgb.shape[:2]
        # project points to 3D space
        points = cv2.reprojectImageTo3D(depth, depth_cam_mat())

        # transform coordinates to RGB camera coordinates
        points = np.dot(points, RGBCamParams['rot'].T)
        points = np.add(points, RGBCamParams['trans'])

        # handle invalid values
        points[depth >= depth.max()] = 0

        points = points.reshape(-1, 640, 3)
        print('point_coord',points[:, :, 0].max(),points[:, :, 1].max(),points[:, :, 2].max())
        # project 3D points back to image plain
        x = np.array((points[:, :, 0] * (RGBCamParams['fx'] / points[:, :, 2]) + RGBCamParams['cx']),
                     dtype=int).clip(0, w - 1)
        y = np.array((points[:, :, 1] * (RGBCamParams['fy'] / points[:, :, 2]) + RGBCamParams['cy']),
                     dtype=int).clip(0, h - 1)

        return points, rgb[y, x]

def capture():
    # Capture a depth image
    depth,timestamp = freenect.sync_get_depth()
    rgb = freenect.sync_get_video()[0]
    depth_norm=(depth-depth.min())/(depth.max()-depth.min())
    depth_norm=depth_norm*255
    # Capture an RGB image
    points,color=get_registred_depth_rgb(depth,rgb)
    img_depth = Image.fromarray(depth_norm).convert('RGB')
    img_depth.save(str(timestamp)+'-depth.jpg')
    img_rgb = Image.fromarray(rgb)
    img_rgb.save(str(timestamp)+'-rgb.jpg')
    print(f"RGB shape: {rgb.shape}")
    #reconstruction
    color=np.reshape(color,(-1,3))
    points=np.reshape(points,(-1,3))
    pc=pv.PolyData(points)
    pc.point_data['color']=color
    pc.save(str(timestamp)+'-.ply',texture='color')
    print('capture !')

capture()
time.sleep(1)
capture()
time.sleep(1)
capture()
#plotter = pv.Plotter()
#plotter.add_points(pc,scalars='color',rgb=True)
#plotter.show()
