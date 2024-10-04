# Two_View_Stereo_Reconstruction
This project implements a two-view stereo algorithm to convert multiple 2D viewpoints into a 3D reconstruction of a scene. The implementation is structured around a Jupyter notebook (two_view.ipynb) that integrates with various Python functions in two_view_stereo.py.

Key tasks include rectifying two camera views, calculating disparity maps, and producing 3D depth maps and point clouds. Below is an outline of the main components of the project, including code functionality and relevant concepts.

## Implementation Overview

**Rectify Two Views**\
-`compute_right2left_transformation()`: Compute the transformation between the right and left camera frames 
-`compute_rectification_R()`: Compute the rectification rotation matrix to transform the left camera view to a rectified frame
-`rectify_2view()`: Warp the images using the computed homographies to obtain rectified images.

**Disparity Map Calculation**
- `image2patch()`: Convert each pixel into a patch for comparison. 
- `ssd_kernel(), sad_kernel(), zncc_kernel()`: Implement three matching metrics—Sum of Squared Differences (SSD), Sum of Absolute Differences (SAD), and Zero-mean Normalized Cross-Correlation (ZNCC).
- `compute_disparity_map()`: Using the matching metrics and consistency checks, compute the disparity map between the two views.

**Depth Map and Point Cloud Generation**
- `compute_dep_and_pcl()`: Convert the disparity map into a depth map and back-project the points to obtain a point cloud in 3D space.

**Post-processing**
- `postprocess()`: Remove outliers and transform the point cloud from the camera frame to the world frame.

