# Two_View_Stereo_Reconstruction
This project implements a two-view stereo algorithm to convert multiple 2D viewpoints into a 3D reconstruction of a scene. The implementation is structured around a Jupyter notebook (two_view.ipynb) that integrates with various Python functions in two_view_stereo.py.

Key tasks include rectifying two camera views, calculating disparity maps, and producing 3D depth maps and point clouds. Below is an outline of the main components of the project, including code functionality and relevant concepts.

## Implementation Overview

**Rectify Two Views**
- `compute_right2left_transformation()`: Compute the transformation between the right and left camera frames 
- `compute_rectification_R()`: Compute the rectification rotation matrix to transform the left camera view to a rectified frame
- `rectify_2view()`: Warp the images using the computed homographies to obtain rectified images.

**Disparity Map Calculation**
- `image2patch()`: Convert each pixel into a patch for comparison. 
- `ssd_kernel(), sad_kernel(), zncc_kernel()`: Implement three matching metrics—Sum of Squared Differences (SSD), Sum of Absolute Differences (SAD), and Zero-mean Normalized Cross-Correlation (ZNCC).
- `compute_disparity_map()`: Using the matching metrics and consistency checks, compute the disparity map between the two views.

**Depth Map and Point Cloud Generation**
- `compute_dep_and_pcl()`: Convert the disparity map into a depth map and back-project the points to obtain a point cloud in 3D space.

**Post-processing**
- `postprocess()`: Remove outliers and transform the point cloud from the camera frame to the world frame.

## Visualization
We use k3d and plotly to visualize the reconstructed point cloud. In the Jupyter notebook, you can interact with the visualizations, including camera poses and point clouds.

## Key Concepts
- Rectification: Aligning stereo images so that corresponding points lie on the same horizontal line.
- Disparity Map: A map that measures the difference in position of the same point in the left and right images.
- Depth Map: A representation of the distance of points in the scene from the camera.
- Point Cloud: A set of data points in space representing the 3D structure of the scene.

## Results

![Cost-Disparity](https://github.com/user-attachments/assets/e59bdafc-dfbc-4b05-a34e-b91e8f9727d3)
![disparity and depth map](https://github.com/user-attachments/assets/2aa37357-2f4d-4f64-8d5e-1cffa097df38)
![LR Consistency Check Mask](https://github.com/user-attachments/assets/788a451e-5ad8-48ec-8e95-8f521004117c)
![postprocessed depth and disparity map](https://github.com/user-attachments/assets/5851e058-1b8b-4933-8efd-5329e1dbd348)
![Rectified Image](https://github.com/user-attachments/assets/26dda2d6-1d17-4d4e-a459-dbb362448ca5)
![Right and Left View](https://github.com/user-attachments/assets/d21325df-dc69-490f-8cef-077524b8aaca)



