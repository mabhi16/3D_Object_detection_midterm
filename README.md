
# SDCND : Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking. 

In this project, you'll fuse measurements from LiDAR and camera and track vehicles over time. You will be using real-world data from the Waymo Open Dataset, detect objects in 3D point clouds and apply an extended Kalman filter for sensor fusion and tracking.

<img src="img/img_title_1.jpeg"/>

The project consists of two major parts: 
1. **Object detection**: In this part, a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. 
2. **Object tracking** : In this part, an extended Kalman filter is used to track vehicles over time, based on the lidar detections fused with camera detections. Data association and track management are implemented as well.

The following diagram contains an outline of the data flow and of the individual steps that make up the algorithm. 

<img src="img/img_title_2_new.png"/>

Also, the project code contains various tasks, which are detailed step-by-step in the code. More information on the algorithm and on the tasks can be found in the Udacity classroom. 

## Project File Structure

📦project<br>
 ┣ 📂dataset --> contains the Waymo Open Dataset sequences <br>
 ┃<br>
 ┣ 📂misc<br>
 ┃ ┣ evaluation.py --> plot functions for tracking visualization and RMSE calculation<br>
 ┃ ┣ helpers.py --> misc. helper functions, e.g. for loading / saving binary files<br>
 ┃ ┗ objdet_tools.py --> object detection functions without student tasks<br>
 ┃ ┗ params.py --> parameter file for the tracking part<br>
 ┃ <br>
 ┣ 📂results --> binary files with pre-computed intermediate results<br>
 ┃ <br>
 ┣ 📂student <br>
 ┃ ┣ association.py --> data association logic for assigning measurements to tracks incl. student tasks <br>
 ┃ ┣ filter.py --> extended Kalman filter implementation incl. student tasks <br>
 ┃ ┣ measurements.py --> sensor and measurement classes for camera and lidar incl. student tasks <br>
 ┃ ┣ objdet_detect.py --> model-based object detection incl. student tasks <br>
 ┃ ┣ objdet_eval.py --> performance assessment for object detection incl. student tasks <br>
 ┃ ┣ objdet_pcl.py --> point-cloud functions, e.g. for birds-eye view incl. student tasks <br>
 ┃ ┗ trackmanagement.py --> track and track management classes incl. student tasks  <br>
 ┃ <br>
 ┣ 📂tools --> external tools<br>
 ┃ ┣ 📂objdet_models --> models for object detection<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┣ 📂darknet<br>
 ┃ ┃ ┃ ┣ 📂config<br>
 ┃ ┃ ┃ ┣ 📂models --> darknet / yolo model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here<br>
 ┃ ┃ ┃ ┃ ┗ complex_yolov4_mse_loss.pth<br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┗ 📂resnet<br>
 ┃ ┃ ┃ ┣ 📂models --> fpn_resnet model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here <br>
 ┃ ┃ ┃ ┃ ┗ fpn_resnet_18_epoch_300.pth <br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┗ 📂waymo_reader --> functions for light-weight loading of Waymo sequences<br>
 ┃<br>
 ┣ basic_loop.py<br>
 ┣ loop_over_dataset.py<br>



## Installation Instructions for Running Locally
### Cloning the Project
In order to create a local copy of the project, please click on "Code" and then "Download ZIP". Alternatively, you may of-course use GitHub Desktop or Git Bash for this purpose. 

### Python
The project has been written using Python 3.7. Please make sure that your local installation is equal or above this version. 

### Package Requirements
All dependencies required for the project have been listed in the file `requirements.txt`. You may either install them one-by-one using pip or you can use the following command to install them all at once: 
`pip3 install -r requirements.txt` 

### Waymo Open Dataset Reader
The Waymo Open Dataset Reader is a very convenient toolbox that allows you to access sequences from the Waymo Open Dataset without the need of installing all of the heavy-weight dependencies that come along with the official toolbox. The installation instructions can be found in `tools/waymo_reader/README.md`. 

### Waymo Open Dataset Files
This project makes use of three different sequences to illustrate the concepts of object detection and tracking. These are: 
- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

To download these files, you will have to register with Waymo Open Dataset first: [Open Dataset – Waymo](https://waymo.com/open/terms), if you have not already, making sure to note "Udacity" as your institution.

Once you have done so, please [click here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) to access the Google Cloud Container that holds all the sequences. Once you have been cleared for access by Waymo (which might take up to 48 hours), you can download the individual sequences. 

The sequences listed above can be found in the folder "training". Please download them and put the `tfrecord`-files into the `dataset` folder of this project.


### Pre-Trained Models
The object detection methods used in this project use pre-trained models which have been provided by the original authors. They can be downloaded [here](https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing) (darknet) and [here](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing) (fpn_resnet). Once downloaded, please copy the model files into the paths `/tools/objdet_models/darknet/pretrained` and `/tools/objdet_models/fpn_resnet/pretrained` respectively.

### Using Pre-Computed Results

In the main file `loop_over_dataset.py`, you can choose which steps of the algorithm should be executed. If you want to call a specific function, you simply need to add the corresponding string literal to one of the following lists: 

- `exec_data` : controls the execution of steps related to sensor data. 
  - `pcl_from_rangeimage` transforms the Waymo Open Data range image into a 3D point-cloud
  - `load_image` returns the image of the front camera

- `exec_detection` : controls which steps of model-based 3D object detection are performed
  - `bev_from_pcl` transforms the point-cloud into a fixed-size birds-eye view perspective
  - `detect_objects` executes the actual detection and returns a set of objects (only vehicles) 
  - `validate_object_labels` decides which ground-truth labels should be considered (e.g. based on difficulty or visibility)
  - `measure_detection_performance` contains methods to evaluate detection performance for a single frame

In case you do not include a specific step into the list, pre-computed binary files will be loaded instead. This enables you to run the algorithm and look at the results even without having implemented anything yet. The pre-computed results for the mid-term project need to be loaded using [this](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7?usp=sharing) link. Please use the folder `darknet` first. Unzip the file within and put its content into the folder `results`.

- `exec_tracking` : controls the execution of the object tracking algorithm

- `exec_visualization` : controls the visualization of results
  - `show_range_image` displays two LiDAR range image channels (range and intensity)
  - `show_labels_in_image` projects ground-truth boxes into the front camera image
  - `show_objects_and_labels_in_bev` projects detected objects and label boxes into the birds-eye view
  - `show_objects_in_bev_labels_in_camera` displays a stacked view with labels inside the camera image on top and the birds-eye view with detected objects on the bottom
  - `show_tracks` displays the tracking results
  - `show_detection_performance` displays the performance evaluation based on all detected 
  - `make_tracking_movie` renders an output movie of the object tracking results

Even without solving any of the tasks, the project code can be executed. 

The final project uses pre-computed lidar detections in order for all students to have the same input data. If you use the workspace, the data is prepared there already. Otherwise, [download the pre-computed lidar detections](https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing) (~1 GB), unzip them and put them in the folder `results`.

## External Dependencies
Parts of this project are based on the following repositories: 
- [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader)
- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)


## License
[License](LICENSE.md)

# 3D Object detection

## 1. Compute Lidar Point-Cloud from Range Image

This task is about extracting two of the data channels within the range image, which are "range" and "intensity", and convert the floating-point data to an 8-bit integer value range. Later crop range image to +/- 90 deg. left and right of the forward-facing x-axis then stack cropped range and intensity image vertically and visualize the result. Following result is shown below

![range_int_img](https://user-images.githubusercontent.com/49077871/196755672-4c2e707c-84a9-4ba1-a52d-3cc5ba2ff132.PNG)

## Visualize lidar point-cloud

The goal of this task is to use the Open3D library to display the lidar point-cloud in a 3d viewer in order to develop a feel for the nature of lidar point-clouds. Below the are few results of this exercise.

![3d_pcl](https://user-images.githubusercontent.com/49077871/196756614-5e377e06-94f0-44fe-936e-ceba76905c3f.png)
![3d_pcl_2](https://user-images.githubusercontent.com/49077871/196756653-dd04d041-dd01-4225-b7b9-668c5ff094a4.png)
![3d_pcl_3](https://user-images.githubusercontent.com/49077871/196762729-c41ad843-9116-4dbc-a8d8-455fe078168d.png)
![3d_pcl_4](https://user-images.githubusercontent.com/49077871/196762760-e809bdc5-e501-46d9-a266-b48e8465bbde.png)
![3d_pcl_5](https://user-images.githubusercontent.com/49077871/196762783-7e86335c-7d6a-4fbe-952c-b2cab1262171.png)
![3d_pcl_7](https://user-images.githubusercontent.com/49077871/196762798-1a6ccb51-8426-4546-b77a-9201d04f4796.png)
![3d-pcl_6](https://user-images.githubusercontent.com/49077871/196763423-40c1014d-4b87-41af-9779-e8d1333a9831.png)
![3d_pcl_8](https://user-images.githubusercontent.com/49077871/196763481-1d5174ed-e6c0-4146-838c-642a0e9dd9ba.png)
![3d_pcl_9](https://user-images.githubusercontent.com/49077871/196763537-d927e792-02fd-4894-8c92-ab771f141841.png)
![3d_pcl_10](https://user-images.githubusercontent.com/49077871/196763591-9b2846e1-426c-4e03-9bbd-d32c601f4353.png)

From the above images it can be clearly observed that the dominant parts that appear in the LIDAR point cloud are tail lamps, bumper, front light. Sometimes when the cars are at an angle side mirrors also appear clearly. 

## 2. Create Birds-Eye View from Lidar PCL

In first step is to create a birds-eye view (BEV) perspective of the lidar point-cloud. Based on the (x,y)-coordinates in sensor space, respective coordinates within the BEV coordinate space are computed the the actual BEV map can be filled with lidar data from the point-cloud. once this task is done the respestice height and intensity map of the same is computed as well, which are shown below

![bev_from_pcl](https://user-images.githubusercontent.com/49077871/196767226-61027753-5efc-415e-b994-97e9e4b3553d.png)

Height Map

![bev_hei](https://user-images.githubusercontent.com/49077871/196767304-19bbb818-f99f-4f23-85ff-e2468559fb8a.png)

Intensity Map

![bev_int](https://user-images.githubusercontent.com/49077871/196767374-c59e25d6-f642-4f30-8be0-d95ef52d98b9.png)

## 3. Model-based Object Detection in BEV Image

The goal of this task is to illustrate how a new model can be integrated into an existing framework. for this following steps are followed:

1. The fpn_resnet is instantiated by adding configs from cloned repository
2. 3D bounding boxes are extracted from the results
3. Output of the model is made to give out bounding box format [class-id, x, y, z, h, w, l, yaw]

Output of the above task is as shown below:

![detection](https://user-images.githubusercontent.com/49077871/197003999-b7d53761-3daa-4477-a723-3538f50dde4d.png)

## 4. Performance detection for 3D Object Detection

The goal of this task is to find pairings between ground-truth labels and detections, so that we can determine if an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). For this geometrical overlap is computed between the bounding boxes of labels and detected objects and determine the percentage of this overlap in relation to the area of the bounding boxes. For multiple matches objects/detections pair with maximum IOU are kept, later false negatives and false positives are computed to calculate precision and recall. After processing all the frames of a sequence, the performance of the object detection algorithm is evaluated. 

![det_perf](https://user-images.githubusercontent.com/49077871/197009240-5bc98595-629d-4f7e-a7f3-59b17442557f.png)

precision = 0.9506578947368421, recall = 0.9444444444444444

To make sure that the code produces plausible results, the flag configs_det.use_labels_as_objects should be set to True in a second run and this should produce 
precision = 1.0, recall = 1.0, as labels are evaluated against themselves

![use_labels_as_objects](https://user-images.githubusercontent.com/49077871/197012500-b664caa4-b66a-495c-a2aa-f4130cbc9e8f.png)



