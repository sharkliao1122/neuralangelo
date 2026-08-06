<!-- PAGE 1 -->
Construction and Building Materials 412 (2024) 134837
Contents lists available at ScienceDirect
Construction and Building Materials
journal homepage: www.elsevier.com/locate/conbuildmat
Evaluation of asphalt pavement texture using multiview stereo
reconstruction based on deep learning
Han-Cheng Dan a,b, Bingjie Lu c,*, Mengyu Li d
a School of Civil Engineering, Central South University, Changsha 410075, Hunan, China
b National Engineering Research Center for High-speed Railway Construction Technology, Central South University, Changsha, Hunan 410075, China
c Postgraduate, School of Civil Engineering, Central South University, Changsha 410075, Hunan, China
d Undergraduate, School of Civil Engineering, Central South University, Changsha 410075, Hunan, China
A R T I C L E I N F O A B S T R A C T
Keywords: To quickly obtain texture information and accurately evaluate the skid resistance of asphalt pavement, a mul-
Deep learning tiview stereo reconstruction method based on deep learning is proposed for evaluating the texture depth of
Multiview images asphalt pavement. First, a depth camera and a digital camera are used to collect RGB-D (i.e., Red, Green, Blue,
Pavement texture reconstruction
Depth) and RGB (i.e., Red, Green, Blue) multiview image datasets on different types of pavements for model
Image processing
training and validation respectively. Then, the model precision is validated by calculating the overlap (inter-
Mean texture depth
section over union, IoU) between the ground truth point cloud and the reconstructed point cloud. The model
Mean profile depth
performances under different training strategies are compared to obtain the best model, and the effects of image
resolution, number of views, and types of pavement material on the model precision are analyzed. Finally, image
processing methods and texture depth characterization indexes are proposed to obtain pavement texture infor-
mation from the reconstructed depth map, thus validating the effectiveness of the model. The results show that
the trained model using the transfer learning strategy achieves a reconstruction precision of 0.77 when the image
resolution is 3024 × 3024 and the number of views is 7. Furthermore, the reconstruction performance remains
stable across different pavement materials, suggesting the model is suitable for accurately reconstructing depth
maps for asphalt pavements. The errors between the predicted texture depth value calculated based on the
volume-based index (MTDp) and the profile-based index (MPDp) using the depth maps after image processing and
the measured texture depth value (MTDe) using the sand patch method are 11.72% and 18.85%, respectively. It
is believed that the pavement texture depth can be effectively evaluated from the reconstructed depth map using
the volume-based metric (MTDp). In contrast to traditional testing methods, this approach requires using only a
digital camera and a personal computer, making it a lightweight and intelligent analysis method for obtaining
pavement texture depth information.
1. Introduction areas [4–8]. There are numerous factors contributing to its emissions
[9], including engines, traffic composition, acoustic impedance, and
The surface texture of asphalt pavement is in direct contact with pavement age [10–13]. Texture, which has only been investigated in
vehicle tires and crucially impacts the skid resistance function of the recent years alongside mixture properties, plays a pivotal role in this
pavement [1]. Based on different wavelengths (λ) and amplitudes (A), context [14–22]. Consequently, the measurement of macrotexture is
pavement textures are commonly classified into microtexture, macro- crucial for the assessment and prediction of pavement skid resistance
texture, megatexture, or unevenness [2]. Macrotexture on pavement is and noise emissions.
characterized by wavelengths between 0.5 mm and 50 mm and ampli- Traditional methods for measuring pavement macrotexture, such as
tudes between 0.1 mm and 20 mm; macrotexture is closely related to the the sand patch method (SPM) for calculating the mean texture depth
skid resistance of the pavement during high-speed vehicle travel [3]. (MTD) indicator [23], are subject to human factors and require traffic
Furthermore, road traffic stands out as the most influential noise source, control during point measurements, which can disrupt normal road
significantly affecting modern human lifestyles, especially in urban usage. These traits limit the accuracy and practicality of the traditional
* Corresponding author.
E-mail address: bingjie_lu@csu.edu.cn (B. Lu).
https://doi.org/10.1016/j.conbuildmat.2023.134837
Received 10 September 2023; Received in revised form 9 December 2023; Accepted 28 December 2023
Available online 5 January 2024
0950-0618/© 2024 Elsevier Ltd. All rights reserved.


<!-- PAGE 2 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
methods. Additionally, the mean profile depth (MPD) indicator is typi- barriers to obtain texture information. For example, using the AMES
cally measured using the laser scanning method [24]. This method re- profiler (Ames Engineering, Ames, Iowa) to acquire macrotexture data
quires specialized equipment and expertise, and the measurement and predict surface friction levels based on CNNs [40], utilizing syn-
process may be affected by environmental conditions and equipment thetic aperture radar data and a proposed framework to predict surface
precision limitations. Additionally, the MPD indicator does not reliably roughness [41], obtaining 3D surface texture data using laser scanners
predict pavement texture skid resistance [25]. and recognizing pavement textures based on the Siamese network [42],
To overcome the limitations of traditional measurement methods, or collecting texture data using a 3D Safety Sensor and employing the
methods related to three-dimensional reconstruction for evaluating Pavement Texture Super Resolution Generative Adversarial Network
pavement texture depth have become a growing interest in recent years. (PT-SRGAN) for texture measurement [43]. These methods may not be
In general, three-dimensional reconstruction methods of pavement as practical in engineering applications as those based on image
texture can be divided into two main categories: active-based and reconstruction.
passive-based methods. Active-based methods for pavement texture The multiview stereo algorithm can infer 3D geometry from a set of
reconstruction primarily include laser scanning and computed tomog- images with known positions and perspectives, enabling highly detailed
raphy (CT) scanning. Laser scanning refers to the use of a high-precision 3D models to be reconstructed based solely on images [44]. This algo-
laser scanner to capture point cloud data of the pavement, which is then rithm is typically utilized in depth map, point cloud, mesh, and
employed for the 3D reconstruction of pavement texture and for per- voxel-based reconstruction methods. Meanwhile, deep learning-based
formance evaluation [26]. This approach offers greater stability and multiview stereo reconstruction methods, which are primarily catego-
accuracy than traditional MTD measurement methods in various rized into depth map-based and voxel-based approaches [45], can
real-world scenarios. However, it requires 3D data as input and is sup- overcome the incompleteness of traditional methods. Unlike direct
ported by the scanner, which has relatively high usage costs. Further- voxel-based reconstruction, which requires high memory consumption
more, the complex processing of 3D data reduces the reconstruction [46], depth map-based methods leverage geometric information be-
efficiency [27]. CT scanning refers to the use of industrial CT scanning tween views to predict a 2.5D depth map for each view. In this context, a
technology to obtain three-dimensional digital information on pave- depth map is defined where each pixel value in the image represents the
ment textures. However, this testing method can only scan specimens distance from a point in the scene to the camera’s xy-plane. The 3D
indoors [28]. Passive-based methods typically involve using model is then reconstructed through depth map fusion and filtering
high-definition cameras to capture images, which are then reconstructed using 3D fusion techniques. The computational cost depends on only the
into pavement textures using digital image processing techniques. number and resolution of the input images [45].
Close-range photogrammetry (CRP) technology, for example, captures The pioneering work in this domain, MVSNet, is an end-to-end deep
multiple surrounding images of pavement measurement points from learning network for depth map inference [47]. It computes only one
different angles. Through digital feature matching between these im- depth map at a time rather than the entire 3D scene, takes a reference
ages, a three-dimensional model of the pavement texture is recon- image and several source images as inputs, and infers the depth map of
structed. The resulting point cloud model of the pavement texture is the reference image. PatchmatchNet [48], which is used in this article, is
highly accurate. However, this method requires using a series of post- an improvement of the plane-sweeping algorithm [49] used in MVSNet.
processing software to reconstruct the 3D model [29]. Multiple photos Since the reference image for depth map prediction is captured parallel
must be captured by circling the target object to ensure sufficient to the pavement, the predicted depth is roughly concentrated on a
overlap between the captured images [30], reducing the efficiency of certain depth plane, without excessive foreground or background
this method. While the close-range photogrammetry method based on scenes. Hence, utilizing the plane-sweeping algorithm for estimating
binocular vision can improve measurement efficiency, it suffers from pavement texture depth is deemed appropriate. This article further ex-
fine texture information loss [31,32]. The digital grayscale imaging plores this depth map-based multiview stereo reconstruction method for
method involves vertically capturing images of asphalt pavement using depth texture evaluation.
a camera. The grayscale values of each pixel in the processed grayscale
image can reflect the actual roughness of the pavement. Based on this 2. The Objective and roadmap of the work
principle, the grayscale values are converted into texture elevation in-
formation, and a three-dimensional pavement model is obtained The objective of this study is to replace the traditional sand patch
through 3D reconstruction [33]. However, the computational accuracy method with multiview shooting technology and deep learning net-
of this method is influenced by factors such as lighting conditions, works to evaluate the MTD of pavement, shorten the measurement time
pavement color, and camera position [34]. for pavement texture detection, prevent human error, and achieve ac-
Due to their relatively simple data structure [27], computation curate and stable predictions for various pavement materials. To achieve
methods based on image processing are popular in practical applica- fast measurement, PatchmatchNet, a deep learning-based 3D recon-
tions. Deep learning [35] methods for pavement texture reconstruction struction model, is used to reconstruct the depth map of the pavement
have advantages in overcoming the limitations of purely visual tech- texture. Image processing and texture depth calculation are then ach-
niques, which are susceptible to lighting conditions and background ieved without requiring time-consuming dense point cloud reconstruc-
noise. In existing studies, pavement textures have been reconstructed tion, providing a new method for evaluating pavement skid resistance.
from monocular images using a deep convolutional neural network Specifically, the evaluation of pavement texture depth can be ach-
(CNN) or from multiview images with the assistance of the proposed ieved through the following method (red dashed box in Fig. 1): first, an
multi-view combination module [36,37]. Furthermore, to address issues image perpendicular to the pavement is selected from a set of multiview
such as imbalanced datasets, Generative Adversarial Network (GAN) images as the reference view, and the remaining views are used as
models are employed for data augmentation and intelligent recognition source views. Then, sparse reconstruction (also known as Structure-
[38]. These studies demonstrated that deep learning-based reconstruc- from-Motion) is performed to estimate the camera poses of this set of
tion methods exhibit significant stability and accuracy, despite the in- images. Next, multiview images with known camera parameters are
fluence of pavement materials and other confounding factors. In used to reconstruct the depth map of the reference view. Finally, the
particular, the reconstruction accuracy of using multiview images is depth value of the pavement texture is obtained through image pro-
better than that of using monocular images because multiple views cessing and relevant metric calculations. After multiview images are
provide more texture and illumination information [39], which can captured with a camera, relevant information about the pavement
compensate for the limitations and uncertainties of a single view. Some texture depth can be obtained solely through computer-assisted
other deep learning-based methods require the use of sensors with entry calculations.
2


<!-- PAGE 3 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 1. Roadmap of the research.
The research framework of this article comprises three parts: dataset calibration board measuring 100 mm × 100 mm with a thickness of
establishment, model training and validation, and pavement texture 3 mm was used (Fig. 2(c)). A total of 10 different asphalt pavements
depth evaluation. The research routes for the latter two parts are shown with varying service times in Chongqing and Changsha were selected.
in Fig. 1. In Section 3.1, the methods for capturing multiview images Among these, six were chosen to create the RGB-D dataset, while the
using the depth camera and the digital camera are determined, and RGB- remaining four were used to create the RGB dataset. Some example road
D and RGB multiview image datasets are collected on different types of landscapes are shown in Fig. 3(a). Fig. 4 illustrates the shooting view-
outdoor roads. The dataset is then postprocessed and divided for the points of the reference image and some partial source images, with the
subsequent training and validation of the network. In Section 3.2, the requirement that the overlap area between adjacent frames should be no
precision of the model is verified by calculating the intersection over less than 70% during shooting. This criterion is crucial for multiview
union (IoU) of the ground truth point cloud and the reconstructed point depth estimation, as it relies on the overlapping areas between multiple
cloud. The model performances under different training strategies are images. To ensure that the dataset covers most common asphalt pave-
compared on the RGB validation set to obtain the best model, and the ment materials, four common pavement types were selected, including
effects of image resolution, number of views, and pavement types on asphalt concrete pavement (AC-13 and AC-16), stone mastic asphalt
model precision are analyzed. In Section 3.3, a method is proposed to pavement (SMA-13), and open-graded friction course pavement (OGFC-
calculate the absolute depth values for the depth map, and an image 16). The dataset was captured during daylight with sufficient light to
processing method is proposed to correct tilt errors. Pavement texture avoid external shadows. In Table 1, the characteristics of the pavements
depth information is then obtained from the depth map on the RGB test from which data were collected for the study are presented. This table
set based on the characterization metrics MTDp and MPDp. The model provides information on pavement types, ages, and the number of RGB
effectiveness is verified by comparison with the measured values of the and RGB-D images captured from each type of pavement. These details
sand spread method MTDe. are essential for gaining insight into the diversity of the dataset and its
relevance to the research objectives.
3. Methodology
3.1.1. RGB-D dataset
3.1. Dataset establishment Since the performance of deep learning networks largely depends on
the quantity and quality of training samples, having as much high-
The dataset consists of two parts: an RGB-D dataset and an RGB quality training data as possible is essential. However, large-scale pub-
dataset built from scratch. To facilitate fixed-point shooting and the lic datasets such as the DTU [50], BlendedMVS [51], and ScanNet [52]
subsequent recovery of absolute depth values from the depth map, a datasets include captures of indoor and outdoor macro objects, which do
3


<!-- PAGE 4 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 2. (a) RGB-D dataset acquisition facilities; (b) RGB data capture equipments; (c) Calibration plate.
not match the scale of the pavement texture depth to be evaluated. To computer vision methods was used to complete the depth map by filling
improve the accuracy and scene specificity of the model’s predictions, holes and removing noise [54], making the depth map complete and
an RGB-D dataset was collected for asphalt pavements. The equipment continuous and providing a relatively accurate initial depth value for
used to capture the RGB-D dataset is shown in Fig. 2(a). The Intel training. In subsequent research, it was found that models trained on a
RealSense D405 is a depth camera based on Intel’s depth sensing tech- higher resolution RGB-D dataset can more accurately reconstruct the
nology that simultaneously captures three-dimensional depth data and depth map. Therefore, the RGB-D dataset was upsampled to a resolution
RGB data using active infrared stereo technology, thus providing a pair of 1000 × 1000 using the bilinear interpolation algorithm [55]. Then,
of RGB-D training samples. The camera provides up to 1280 × 720 pixel the Open3D library was used for pose estimation of the RGB-D dataset,
resolution and a 60 frames per second frame rate; it can achieve accu- and 2200 pairs of RGB-D images were obtained for subsequent model
racy within 1 mm in close-range shooting, which essentially meets the training. Examples of RGB-D image pairs are shown in Fig. 3(b).
requirements of pavement texture depth calculation. The camera was
mounted on a compatible bracket and controlled by a computer program 3.1.2. RGB dataset
to capture images from multiple viewpoints at a distance of approxi- The RGB-D dataset is not used to evaluate the model performance
mately 15 cm from the ground, while avoiding image blur caused by because the depth map captured by the depth camera may have missing
unnecessary shaking. To ensure the accuracy of the depth map infor- depth values. Even after depth completion, this depth map cannot be
mation, overexposure and underexposure were avoided. Approximately used as the ground truth for validation. In multiview stereo recon-
30 pairs of RGB-D images with a fixed resolution of 640 × 640 were struction based on depth maps, the quality of the depth map directly
captured for each calibration board. Postprocessing techniques such as affects the accuracy of the reconstructed point cloud. Therefore, the
hole filling and depth value smoothing [53] were used during shooting, accuracy of the depth map reconstruction can be indirectly evaluated by
and alignment scripts were used to register image pairs, resulting in an analyzing the similarity between the reconstructed point cloud and the
RGB-D image set with the same resolution and a one-to-one corre- real scene. Previous studies [30] have shown that traditional multiview
spondence between pixel points. stereo reconstruction techniques based on photogrammetry exhibit
The depth map captured by a consumer-grade RGB-D camera can satisfactory accuracy, and the predicted pavement texture depth
provide depth information to some extent, but its measurements exhibit strongly correlates with the measured values. Therefore, the ground
errors and uncertainties, leading to the loss of some depth details. truth for validation in this study is taken from the point cloud accurately
Additionally, the depth capture range of the sensor is limited to a certain reconstructed from the RGB dataset using specialized photogrammetry
distance, which hinders its use in close-range scenarios. Furthermore, software, PhotoScan [56], rather than the depth map, and the validation
due to the influence of ambient light during the collection process, the object is the point cloud obtained by fusing the predicted depth maps
obtained pavement texture depth map obtained is poor quality, so from multiple views of a scene. A strong precision validation indicates
postprocessing is necessary before the map is input into the network for that the model can reconstruct high-quality depth maps. Additionally,
training. Most depth completion algorithms based on deep learning are the RGB dataset is also used for the effectiveness evaluation of the
designed for transparent and reflective objects, so they are not suitable model, so building a well-constructed RGB dataset is crucial.
for this dataset. Therefore, the IP-Basic algorithm based on traditional To meet the development needs of subsequent mobile testing
4


<!-- PAGE 5 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 3. The process of data collection.
Table 1
Pavement data characteristics.
RGB-D dataset RGB dataset
Type Age Number of image Type Age Number of image
pairs sets
AC-13 5 260 AC-13 4 600
years years
AC-13 2 290 AC-16 3 600
years years
AC-16 7 230 SMA-13 5 600
years years
AC-16 2 310 OGFC- 2 600
years 16 years
SMA-13 4 550
years
OGFC- 1 year 560
16
Total 2200 2400
Fig. 4. Camera poses during image acquisition.
affect the precision of the depth map reconstruction. The camera’s pa-
devices, an iPhone 12 was used to capture the RGB dataset, and a phone rameters, such as aperture and shutter speed, should be controlled to
holder was used to assist with the shooting (Fig. 2(b)). Similarly, images ensure consistent exposure between different images. A total of 240 sets
were captured from multiple viewpoints at a distance of approximately of RGB datasets of different types of roads were captured, with a fixed
15 cm from the ground, and Bluetooth remote control was used for number of 10 images per group and a resolution of 3024 × 3024 for
capturing the images to avoid motion blur. When shooting the pavement each image.
on the calibration board, the camera’s viewpoint in the reference image The RGB dataset was divided into validation and test sets in a 1:1
should be parallel to the pavement, and the camera’s viewpoint in the ratio. These sets were used to evaluate the model’s precision and
source image should be evenly distributed around the reference image to effectiveness, respectively. To evaluate the model’s precision at different
provide sufficient multiview texture information (Fig. 3(c)). The number image resolutions, the RGB validation set was downsampled twice and
of images in each group of data should not be overly large, as this would upsampled once while the original resolution of 3024 × 3024 was
result in a long data collection time not suitable for efficient engineering maintained, resulting in resolutions of 2048 × 2048, 2560 × 2560, and
applications; this number should also not be overly small, as this would 3400 × 3400.
5


<!-- PAGE 6 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
3.2. Model training and validation different hyperparameter settings were conducted. The training pa-
rameters resulting in the best model performance are shown in Table 2.
3.2.1. Precision evaluation metric During training, the data loading module randomly scaled and cropped
To evaluate the precision of the model on the RGB validation set, the the input images. To enhance the robustness of the model, a training
IoU [57] evaluation metric was introduced to quantitatively analyze the strategy was adopted that randomly selects images from the dataset to
overlap between the predicted point cloud and the ground truth point match the reference image, rather than selecting the source image that
cloud. The IoU is defined by Eq. (1): best matches the reference image. In addition, segmented learning rate
decay was used to stabilize the model as it approaches the optimal so-
IoU = V GT∩Pre (1) lution. The initial learning rate was set to 0.0001, and the learning rate
V GT∪Pre
was reduced by half after the 10th, 12th, and 14th training epochs.
where GT refers to the point cloud reconstructed using PhotoScan soft- Weight decay regularization was also used to reduce the overfitting risk,
ware, which is considered as the ground truth. Pre refers to the point with the parameter set to 0.0005 to penalize larger weight values and
cloud reconstructed using the model, which is used as the evaluation avoid overly fit and constrained models. The Adam optimization algo-
object for precision. Calculating the IoU involves converting the point rithm with an adaptive learning rate was used to accelerate the training
cloud into a three-dimensional grid representation with voxels as the process, enabling the model to converge more quickly to a local optimal
basic unit. The voxelization process assigns the points in the point cloud solution and avoiding unstable training, gradient vanishing, or explo-
to corresponding voxels and calculates the ratio of the intersection of the sion caused by a learning rate that is overly large or small for gradient
two point cloud voxels to their union. IoU values range from 0 to 1. The descent algorithms.
closer IoU value is to 1, the more similar the spatial shapes of the two 3D
models are, indicating that the precision of the predicted point cloud is 3.2.2.2. Training by the transfer learning method. When obtaining the
better and that the model can perform good quality depth map recon- training dataset for a neural network is difficult, transfer learning [58]
struction. The specific process can be seen in Fig. 5. In the first stage, the can be used for model training. Transfer learning applies knowledge
reference image along with its N-1 source images form the inputs of the acquired from one task to other related tasks to improve performance on
network and the reconstructed depth maps are obtained by using our the target task, especially when data for the target task is limited or hard
model. Then, all the depth maps are fused to create a point cloud. In the to obtain. It helps models adapt to new tasks more quickly, reducing
second stage, the same set of images is input into the PhotoScan software training time and resource costs. Due to time and labor limitations, the
and reconstructed to a point cloud, which is used as the ground truth amount of RGB-D data collected in this study is smaller than that in
point cloud. In the third stage, the IoU metric is used to calculate the
overlap area between two point clouds.
Table 2
Parameter settings for training on the RGB-D dataset.
3.2.2. Training strategies
Parameter Value Parameter Value
3.2.2.1. Training on the established pavement texture RGB-D dataset image_max_dim 1000 × 1000 n_views 7
directly. The model was trained directly on the RGB-D dataset on an batch_size 1 epoch 16
leaning rate 0.0001 lrepochs 10,12,14:2
NVIDIA GeForce RTX 3080 laptop device with 16.0 GB of GPU memory.
weight decay 0.0005 robust_train True
To obtain the best-performing model, multiple experiments with
Fig. 5. The process of model precision validation.
6


<!-- PAGE 7 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
publicly available datasets. To overcome this problem, we consider reconstruction models. The second model, JDACS-MS [60], is an unsu-
using the large-scale public dataset DTU for pretraining the model, using pervised depth estimation model that provides reliable guidance for
the pretrained model’s weights as the initialization parameters, and then depth estimation through additional priors of semantic consistency and
fine-tuning the model on the RGB-D dataset [59]. This approach can use data augmentation consistency. The third model, Fast-MVSNet [61],
the features and parameter initialization values learned by the pre- improves the reconstruction speed by using a prediction framework
trained model to accelerate model training and improve performance from sparse to dense. The fourth model, COLMAP [62,63], is a tradi-
while reducing the data requirements and training time. The model tional 3D reconstruction method that restores point clouds through
further learns the pavement texture features from the RGB-D dataset sparse reconstruction, depth map estimation, and dense reconstruction.
based on the pretrained weights and is expected to achieve good For fairness, the same batch of RGB validation sets was used for com-
reconstruction results. parison, and the same image resolution and number of viewpoints were
The DTU dataset [50] contains 128 controlled laboratory environ- used in each model.
ments, covering various objects and materials. Each scene was scanned
using a structured light scanner at 49 camera positions under 7 different 3.3. Pavement texture depth evaluation
lighting conditions, resulting in a total of 43904 pairs of RGB-D data
with a resolution of 1200 × 1600 pixels. When the DTU dataset was To validate the effectiveness of the model, a method for analyzing
used for pretraining, the maximum image size of the input image was set and comparing the depth values of reconstructed textures and on-site
to 640 × 512 in consideration of computational resources and model measured values through image processing and metric calculation is
training efficiency. Four pairs of data were sent to the neural network for proposed in this section. Evaluation was conducted on the RGB test set
training at each iteration. Because of the large amount of data, the containing different pavement materials, with a fixed multiview image
learning rate was set to 0.001 to accelerate the model’s convergence, resolution of 3024 × 3024 for the input model and a fixed number of 7
and segmented learning rate decay was also used. The weight decay views. Notably, the RGB test set used in this section was collected from
value was set to 0.005 to limit the model’s complexity and improve its different pavements than the RGB validation set discussed earlier.
generalizability. The specific training parameters are shown in Table 3.
The model was then fine-tuned using the pavement texture RGB-D 3.3.1. Image processing
dataset with the same process as described in Section 3.2.2.1. Fig. 6 shows the specific process of the image processing stage, which
aims to obtain a depth map that can be directly used for metric calcu-
3.2.3. Analysis of factors affecting model precision lation. Each step is discussed in detail in this section.
The completeness and accuracy of the reconstructed point clouds
both affect the degree of overlap (IoU) with the ground truth point 3.3.1.1. Scale restoration and ROI selection. Generally, to uniquely
cloud. Accuracy refers to the degree to which the reconstructed model describe the coordinates of each spatial point and the position of the
matches the texture features of the real object. Highly accurate 3D camera, a world coordinate system must first be defined. Then, to
reconstruction requires high-resolution, distortion-free input images. establish the mapping relationship between 3D space points and the
Completeness refers all objects and details in the scene being included in camera plane, as well as the relative relationship between multiple
the reconstructed result, without omissions or information loss. High cameras, a local camera coordinate system is defined for each camera.
completeness in 3D reconstruction requires comprehensive data and To establish geometric constraints, the structure-from-motion (SFM)
complete viewing angle information. However, due to hardware limi- technique [62] is used to recover the camera poses from multiple views,
tations, a trade-off between accuracy and completeness often exists. so that the MVS algorithm can infer the depth information of the scene
Pursuing higher accuracy may require more views to provide texture based on the geometric relationship between images. In this paper,
information; otherwise, the reconstruction completeness will be COLMAP open-source software is used as a mature SFM technique,
impacted. Similarly, pursuing higher completeness limits the model’s which can accurately estimate the camera’s intrinsic and extrinsic pa-
ability to process high-resolution images, thereby affecting the accuracy rameters. COLMAP defaults to using the camera coordinate system of
of reconstruction. Therefore, evaluating the impact of different numbers the first image in the multiview image set as the world coordinate sys-
of views and image resolutions on the model reconstruction quality is tem, with the camera facing the negative Z-axis. Points closer to the
necessary. camera have smaller depth values (Fig. 7). Therefore, the reference
To further evaluate the reconstruction precision of the model on image should be captured first, and the optical axis of the reference view
different pavement materials, the data in the RGB validation set, given should be as perpendicular to the pavement as possible to make the
an image resolution of 3024 × 3024, were classified according to depth values in the depth map correspond to the depth of the pavement.
pavement materials, and the set was reconstructed with a fixed number However, the reconstructed depth map cannot be directly used to
of 7 viewpoints. Both quantitative analysis of the point cloud recon- calculate the depth values of the pavement because without calibration
struction results and qualitative analysis of the depth map reconstruc- information, the depth values in the reconstructed depth map are
tion results were conducted. defined as the Z values of the camera coordinate system rather than the
absolute depth values in the real world. Therefore, to obtain absolute
3.2.4. Comparative studies of different models depth values, additional information and constraints are required for
A comparison study was conducted to demonstrate the advantages of scale calibration. Four pairs of calibration points (marked with blue and
the proposed framework on the 3D reconstruction of pavement texture. red dots) are taken at the upper and lower sides of the four corners on the
Four additional models were selected in the study. The first model, calibration board with a known thickness of 3 mm (Fig. 8(b)). The scale
MVSNet [47], represents the initial generation of learning-based 3D factor can be calculated, and the absolute depth values can be obtained
using Eq. (2):
Table 3 1 ∑4 3mm
Parameter settings for training on the DTU dataset. Z abs = Z rel ⋅scale = Z rel ⋅ 4 Z (cid:0) Z (2)
i=1 blue i red i
Parameter Value Parameter Value
image_max_dim 640 × 512 n_views 5 where Zabs represents the absolute depth value, Zrel represents the rela-
batch_size 4 epoch 16 tive depth value, scale is the scale factor, and Zblue_i and Zred_i are the
leaning rate 0.001 lrepochs 10,12,14:2 relative depth values of each pair of calibration points. For ease of
weight decay 0.005 robust_train True
writing, the term " Z " is used instead of " Zabs ", and the term "depth
7


<!-- PAGE 8 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 6. The workflow of image processing.
value" is used instead of "absolute depth value" in the following text.
After the scale-recovered depth map is obtained, the image must be
cropped according to the region of interest (Fig. 8(c)) defined by the
calibration board (Fig. 8(d)). Cropping the image as close as possible to
the 100 mm × 100 mm border inside the calibration board is important
for ensuring the real size of each pixel in the depth map is reliably and
accurately calculated.
3.3.1.2. Tilt error correction. Due to the possibility of the camera being
tilted or in an abnormal position during shooting, and the presence of
road slopes, the optical axis of the reference image cannot be guaranteed
to be completely perpendicular to the pavement. Therefore, the incli-
nation of the pavement normal vector must be corrected, and the
pavement normal vector must be rotated to align with the Z-axis of the
camera coordinate system. Specifically, the RANSAC algorithm [64] is
first used to fit a plane to the depth map and obtain the normal vector n
of the fitted plane. Then, a coordinate transformation matrix T is con-
structed to project the points in the depth map to the rotated coordinate
system, thus obtaining the depth map with the corrected pavement
normal vector (Eq. (3)):
Fig. 7. Coordinate system definition.
Fig. 8. Depth map scale recovery and cropping: (a) Reference image; (b) Selection of 4 pairs of calibration points marked with red and blue colors; (c) ROI selection;
(d) Depth map after border cropping.
8


<!-- PAGE 9 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
⎛ ⎛ ⎞⎞ ⎛ ⎛ ⎞⎞
X ( ) X calibration board range, an improved sand patch method (Fig. 10) is
Z′ = ⎜ ⎜ ⎝T⋅ ⎜ ⎜ ⎝ Y Z ⎟ ⎟ ⎠ ⎟ ⎟ ⎠ = ⎜ ⎜ ⎝ 0 R T 3 1 t ⋅ ⎜ ⎜ ⎝ Y Z ⎟ ⎟ ⎠ ⎟ ⎟ ⎠ (3) u co se n d d u to c t m ed e a o s n u r d e r t y h e a n te d x t c u le r a e n d e a p re th a s i n o t f h t e h fi e e p ld av [ e 3 m 6] e . n T t, e s a t n in d g d i i s f f p e r r e e f n e t r a t b e l s y t
1 2 1 2 points are selected for each road. 25 cm3 of standard sand, measured by
a measuring cylinder, is gradually poured into the target fixed area.
where Z′ represents the corrected depth value and X, Y, Z are the pixel Simultaneously, standard sand is spread in the sample area using a tool,
coordinates and the corresponding depth valu⎛e in ⎞the original depth and pouring is stopped when the standard sand fills the calibration
0 board square. The final reading v′ of the remaining standard sand in the
map, respectively. If the rotation axis r = n × ⎝ 0 ⎠ is a zero vector, measuring cylinder is recorded using a graduated cylinder, and the
(cid:0) 1
then the rotation matrixRis the ide⎛ntit⎛y ma⎞tr
/
ix. Ot⎞herwise, R =
measure
4
d
V
mean texture depth MTDe is obtained using Eq. (5).
(cid:0) ) 0 MTD = (4)
Rodrigues ‖ r r‖ ⋅θ [0], where θ = arccos⎝ n⋅⎝ 0 ⎠ ‖n‖⎠ is the angle πD2
(cid:0) 1 (cid:0) )
25 (cid:0) v′ ⋅103
between n and the Z-axis and t is the three-dimensional translation MTD e = 1002 (5)
vector.
Fig. 9(a) shows the depth value changes of a certain profile before
and after the correction of tilt error, and Fig. 9(b and c) show the cor- 3.3.2.2. Volume-based metric. To accurately characterize the depth
responding depth maps. In the same depth map, the darker the color of a values of the reconstructed texture, appropriate calculation indexes
region is, the smaller the depth value of the points in that region, which must be selected. The depth information of each pixel in the depth map
indicates that the texture is protruding from the surface. The lighter the is the depth value relative to the imaging plane of the camera, but what
color of a region is, the larger the depth value of the points in that region, needs to be calculated is the depth value of each pixel relative to the
which indicates the texture is sinking into the surface. The color dif- texture reference plane [30]. To avoid the influence of extremely small
ference of the depth map after the correction of tilt error is smaller than depth values (elevation peaks) on the calculation, the plane containing
that of the depth map before correction, indicating that the absolute the p % smallest depth percentile value in the depth map is selected as
difference of the depth value is reduced. The overall slope of the depth the texture reference plane (Fig. 11). The mean depth value MTDp
value profile is also reduced, indicating that the tilt error of the pave- relative to the texture reference plane is then calculated for the
ment has been effectively corrected. remaining 100-p % pixels using Eq. (6) to reflect the overall trend of the
pavement texture variation.
3.3.2. Effectiveness evaluation metrics ∑M ∑N (cid:0) )
Z (cid:0) Z ⋅s
To determine the effectiveness of the model, whether the recon- mn p
MTD = m=1 n=1
structed texture features are consistent with the corresponding actual p S
pavement texture must be determined. If a significant correlation ex-
hibits between the two, the reconstructed texture can be considered 1002
s = mm2 (6)
effective [65]. M⋅N
3.3.2.1. Benchmark metric. MTD is a three-dimensional analysis index where M and N are the number of pixels in the depth map in the lon-
that considers the spatial characteristics of pavement texture. It calcu- gitudinal and transverse directions, respectively; Zmn represents the
lates the mean texture depth of the pavement by gradually spreading depth value of the pixel in the m-th row and n-th column; Zp represents
fine particles with a volume of Vonto the pavement, forming a circle the depth value of the selected texture reference plane; S= 1002 mm2
with a diameter of D, as shown in Eq. (4). To measure MTD within the represents the area of the calibration board; and s represents the size of
each pixel in the depth map.
Fig. 9. (a) Profile depth values of the 1060th row; (b) Original depth map; (c) Depth map after tilt error correction.
9


<!-- PAGE 10 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 10. Measurement of MTDe using sand patch method: (a) Experimental equipments; (b) Measuring range; (c) Fixed-point measurement.
Fig. 11. Reference surface selection of the pavement.
Regarding different selections of p values for calculating MTDp, the values of p. For rough pavements such as SMA-13 with exposed stones
MAPE statistical index (Eq. (7)) is used to measure the average relative and OGFC-16 with large voids between aggregates, a larger p value
error percentage between the predicted depth values and the measured should be selected to exclude more extreme elevation points. For AC
values. pavements with relatively fine textures and smoother surfaces, a smaller
MAPE = 1 ∑N ⃒ ⃒MTD e (cid:0) MTD p ⃒ ⃒ ⋅100% (7) p p l v a a n l e u w e s h h e o re u l t d h e b e 5 t s h e l p e e c r t c e e d n t t o il m e d o e re p t a h c q cu u r a a n t t e i l l y e p va re lu d e ic i t s M lo T c D at p e . d F i i s n a se ll l y e , c t t h ed e
N MTD
i=1 e as the texture reference plane; at this point, the prediction errors for
each type of pavement are close to the minimum.
where N is the sample size. Fig. 12 shows the results of error for different
3.3.2.3. Profile-based metric. The mean profile depth (MPD) (Eq. (8)) is
another widely used characterization index. The MPD divides the length
of the pavement texture curve into two segments and calculates the
difference between the average elevation value (h) and the average of
the two peak elevation values (h1 and h2). Following this idea, the
predicted value of the texture depth of the reconstructed pavement can
be characterized by Eq. (9).
h + h
MPD = 1 2 (cid:0) h (8)
2
1 ∑N 1 ∑N
MPD p = N MSD = N (Z Ali (cid:0) Z Pli ) (9)
i=1 i=1
where the mean section depth (MSD) is the depth value of the average
elevation of a row in the depth map (ZAli ) minus the depth value of the
peak elevation (ZPli ). N is the total number of rows in the depth map.
Fig. 13 shows a schematic diagram of the calculation process of MPDp.
There are protruding aggregates above the average elevation plane.
These angled particle aggregates come into contact with tires and
generate friction force, thereby serving as an important impact on
Fig. 12. The effect of p-value selection under different pavement materials.
10


<!-- PAGE 11 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 13. Calculation diagram of MPDp.
pavement anti-skid performance [66]. meet the requirements of the target task. Moreover, transfer learning
usually requires a smaller dataset for the target task, which may be more
4. Results and discussion limited than the source task. As a result, the model requires more iter-
ations to find the optimal solution in the target task training process,
4.1. Comparison between different training strategies leading to a slower convergence speed. Additionally, the model preci-
sion of the strategy without transfer learning decreased in the later
The model’s reconstruction precision was compared on the RGB stages of training. This is because the pavement texture RGB-D dataset
validation set with a resolution of 3024 × 3024 and 7 views for different has a small amount of data, and multiple rounds of training may cause
training strategies. Fig. 14 shows the precision of the model’s recon- the model to overfit the training set, resulting in poor performance on
structed point cloud after each training epoch, with and without transfer the validation set. After training by transfer learning for 14 epochs, the
learning. The training was conducted for a total of 16 epochs, and each model’s performance also exhibited a decreasing trend. To avoid
epoch took approximately 1.5 h to complete on the GPU. Regardless of possible overfitting, the best model was selected at the 14th epoch,
whether transfer learning was used, the IoU increased rapidly during the where the model achieved the highest IoU of 0.77 for point cloud
initial training period. When training was conducted directly on the reconstruction precision. This model was used in the subsequent
pavement texture RGB-D dataset without transfer learning, the peak experiments.
performance of the model was achieved at approximately the 7th epoch Table 4 further compares the point cloud reconstruction precision of
with an IoU of 0.51. However, when transfer learning was used, the the model utilizing the DTU dataset for pretraining, trained directly on
model’s performance gradually increased after rapidly increasing during the RGB-D dataset, and trained by transfer learning. The pretrained
the early training phase. The IoU reached 0.77 at approximately the model performs well because it was trained on a large-scale general
14th epoch, which was slower than the training convergence speed dataset, which results in strong feature extraction and generalization
without transfer learning. This is because the pretrained model’s fea- capabilities, enabling it to perform well when handling new tasks or
tures may not be completely applicable to the target task, as the data specific domain data. The model trained directly on the RGB-D dataset
distributions and feature representations of the two tasks may differ. performs poorly because the dataset is small, making it difficult for the
Therefore, the existing features must be adjusted for a longer time to model to learn and generalize to other road scenes. In addition, the
dataset may exhibit issues such as noise, sample imbalance, or missing
depth values, which can negatively impact the model’s performance.
The model trained by transfer learning achieves the best performance by
combining the powerful feature representation capabilities of the pre-
trained model and the specific information of the RGB-D dataset. Fine-
tuning the pretrained model with the RGB-D dataset allows the model
to be adapted to the data of the specific task, improving its performance
in this field. Fig. 15 depicts the depth map reconstruction results of the
model under different training strategies. The model trained with
transfer learning can more clearly reconstruct the pavement texture
contour and details and can more accurately capture the key features
and shapes than the other models. This model is considered capable of
effectively handling multiview image inputs, achieving high-quality
depth map reconstruction results, and accurately estimating scene
Table 4
Model precision under different training strategies.
Strategies Pre-trained Without transfer With transfer
model learning learning
IoU 0.65 0.51 0.77
Fig. 14. Model precision after each epoch under different training strategies.
11


<!-- PAGE 12 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 15. Depth map reconstruction results under different training strategies.
distance. 3024 × 3024, respectively.
4.2.2. Pavement materials
4.2. Factors influencing model accuracy Fig. 17 shows the precision of the reconstructed point cloud under
different pavement materials. The boxplot shows that the model’s
4.2.1. View numbers and image resolution reconstruction precision varies for different types of pavements: the
Table 5 quantitatively compares the overlap between the recon- model’s reconstruction performance on AC-13 pavements with smaller
structed point cloud and the ground truth point cloud using Eq. (1), and particle size is worse than its performance on AC-16 pavements with
Fig. 16 shows the qualitative comparison of the reconstructed depth larger particles because the model cannot reconstruct fine textures well
map. As the image resolution increased from 2048 × 2048 to enough due to the potential accuracy limitations. However, the preci-
3024 × 3024, the model reconstruction quality exhibited an upward sion of the model is still within the expected range. The model’s
trend because a higher image resolution can provide more accurate reconstruction performance on SMA pavements is better because SMA
texture details and enable fine textures to be more accurately recon- pavements have larger particle size aggregates that have more diverse
structed. However, the reconstruction quality slightly decreases at the shapes and textures than continuous graded aggregates of AC pave-
higher resolution of 3400 × 3400. This is likely because the higher- ments, resulting in higher reconstruction precision. The reconstruction
resolution reference view requires more source views to provide error on OGFC pavements arises from the complex texture structure and
texture information for matching and avoid impacting the completeness large number of existing holes, which make depth prediction difficult.
of the model reconstruction. Similarly, when the number of views The model tends to extract texture features of coarse aggregates that
increased from 5 to 7, the model reconstruction quality showed an up- occupy a larger proportion of the pavement; as a result, some texture
ward trend because the increase in the number of views alleviated oc- details are ignored [36], decreasing the reconstruction precision. Fig. 18
clusion problems and provided more texture information, thereby shows the depth map reconstruction results for different types of pave-
making the reconstruction more complete. However, when the number ments, where pixels closer to black in color are closer to the camera
of views increased to 10, the model reconstruction quality decreased imaging plane. Qualitative analysis shows that the model can accurately
somewhat, which is likely due to potential errors in feature matching reconstruct the areas that protrude or are sunken relative to the road
between the additional views, which impact the model reconstruction surface for pavements with a high proportion of coarse aggregates, but it
accuracy. In summary, the experiments showed that the optimal number ignores some finer texture details. In contrast, the reconstructed depth
of views and image resolution for model reconstruction were 7 and map provides more comprehensive detail information for pavements
with smoother textures. Overall, the model’s reconstruction perfor-
Table 5 mance is stable on various pavement materials, indicating the materials
Reconstruction quality under different view numbers and resolutions. have a relatively small impact on the model’s reconstruction precision.
The method exhibits a satisfactory level of precision under real-world
Image resolution View numbers
conditions.
5 7 10
2048 × 2048 0.52 0.54 0.53
2560 × 2560 0.69 0.72 0.70 4.3. Model performance comparison
3024 × 3024 0.74 0.77 0.75
3400 × 3400 0.73 0.76 0.74
Table 6 shows the point cloud reconstruction results of different 3D
12


<!-- PAGE 13 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 16. Reconstruction results on the RGB validation dataset with different parameters. Top: whole depth map. Middle and bottom: zoomed local region of
rectangle. N is the numbers of view. Res is the image resolution. As N increases, the reconstruction becomes more complete. As Res increases, the reconstruction
becomes more accurate.
model’s learning and results in unsatisfactory reconstruction precision
for detection needs. Fast-MVSNet [61] constructs a sparse cost volume to
predict high-resolution sparse depth maps, resulting in depth estimation
errors. These errors subsequently impact the generation and optimiza-
tion of dense depth maps, leading to a decrease in reconstruction pre-
cision. COLMAP [62,63] obtains more accurate depth estimation values,
but it still exhibits some problems in depth map completeness and depth
continuity, and its depth estimation efficiency is lower than that of deep
learning methods. PatchmatchNet, a supervised deep learning model
that differs from the MVSNet pipeline network design, reduces the use of
3D cost volume regularization and integrates the Patchmatch algorithm
[67] based on the assumption of dense parallel depth layers in the
plane-sweeping algorithm [49] for random initialization and iterative
propagation. This model achieves a good balance between reconstruc-
tion efficiency and precision. Therefore, the performance of the model
selected in this paper is considered acceptable.
Fig. 17. Reconstruction precision on different pavement materials. 4.4. Effectiveness evaluation results
reconstruction models for pavement textures. MVSNet [47] performs Partial calculation results of MTDe, MTDp and MPDp are presented in
poorly, not only because the simple plane-sweeping algorithm cannot Table 7, where the MTDe value measured by the sand patch method is
effectively handle detailed textures but also because the generated depth considered the benchmark value. The MTDp index has good character-
map resolution is lower than the input image resolution, significantly ization ability for four different types of road materials, with an overall
reducing the precision of the reconstructed point cloud. JDACS-MS [60] mean relative error of 11.72%. The absolute errors calculated based on
uses only unsupervised-generated target signals for training. The the MTDp index are generally less than 0.1 mm, excluding that of the
self-generated depth maps usually have errors, which impacts the OGFC pavement. These errors are close to the benchmark value, meeting
the accuracy requirements for pavement texture depth detection [68].
13


<!-- PAGE 14 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
Fig. 18. Examples of reconstructed depth map for different pavement types.
SMA and AC pavements, with larger absolute errors between the pre-
Table 6
dicted index and the benchmark value. This is because measuring the
Comparison of point cloud reconstruction quality using different models.
texture depth of porous pavements using the sand patch method is not
Model MVSNet JDACS-MS Fast-MVSNet COLMAP Ours reliable; some gravel may fall into the pores, resulting in a high value of
IoU 0.48 0.54 0.60 0.64 0.77 MTDe [70]. Moreover, due to the obvious nonplanarity, large voids
between aggregates, and rough surface of the OGFC pavement, the
predicted depth values vary greatly between different positions in the
However, the texture depth calculated based on the MPDp index is depth map, affecting the accuracy of the model’s depth value prediction
significantly smaller than the benchmark value, with an overall mean
when using the plane-sweeping algorithm. For some measured points on
relative error of 18.85%. For all types of pavements, the average abso-
AC-13 and SMA-13 with smaller texture depths, the error between the
lute errors of the MPDp results are greater than 0.1 mm, and the average
predicted value and the benchmark value is mainly because the sand
relative error is significantly larger than that of MTDp. This is because
patch method is suitable for asphalt pavements with a texture depth
the index calculated based on the two-dimensional profile is too simple
greater than 0.25 mm [70]. The 0.3 mm clean sand cannot be fully
to accurately characterize different textures [69]. Moreover, the calcu-
embedded in those pavements, resulting in inaccurate MTDe
lation error of the MPDp index is larger on SMA and OGFC pavements
measurements.
than on the AC pavement. This is because the aggregates in SMA and
Regardless of the selected calculation index or pavement material
OGFC pavements are more uneven, and the profile texture fluctuations
type, the prediction error is also impacted by the erroneous estimation of
are larger. The assumed average elevation plane differs greatly from the
the depth information of the calibration points. This error increases to
texture at the bottom of the valley, resulting in a smaller MPDp calcu-
the depth values of other points due to the scale recovery process,
lation compared to the benchmark value.
decreasing the overall prediction accuracy. Therefore, using more cali-
However, regardless of which index is selected, the prediction per-
bration points and accurate calibration methods is important. The error
formance of the model for the OGFC pavement is worse than that for the
also comes from the difference between the plane model fitted by the
Table 7
Partial calculation results of three indicators.
Materials MTDe (mm) MTDp (mm) AE (mm) MAE (mm) RE (%) MRE (%) MPDp (mm) AE (mm) MAE (mm) RE (%) MRE (%)
AC-13 0.57 0.65 0.08 0.086 14.04 13.48 0.46 0.11 0.118 19.30 18.57
0.59 0.50 0.09 15.25 0.45 0.14 23.73
0.62 0.54 0.08 12.90 0.53 0.09 14.52
0.69 0.62 0.07 10.14 0.56 0.13 18.84
0.73 0.84 0.11 15.07 0.61 0.12 16.44
AC-16 0.90 0.83 0.07 0.072 7.78 7.35 0.77 0.13 0.126 14.44 12.88
0.93 0.99 0.06 6.45 0.82 0.11 11.83
0.96 1.04 0.08 8.33 0.84 0.12 12.50
1.02 1.08 0.06 5.88 0.90 0.12 11.76
1.08 0.99 0.09 8.33 0.93 0.15 13.89
SMA-13 0.63 0.72 0.09 0.100 14.28 14.41 0.48 0.15 0.172 23.81 24.73
0.67 0.56 0.11 16.42 0.49 0.18 26.87
0.69 0.77 0.08 11.59 0.55 0.14 20.29
0.73 0.61 0.12 16.44 0.54 0.19 26.03
0.75 0.65 0.10 13.33 0.55 0.20 26.67
OGFC-16 2.16 1.96 0.20 0.274 9.26 11.63 1.79 0.37 0.452 17.13 19.22
2.26 2.02 0.24 10.62 1.84 0.42 18.58
2.37 2.05 0.32 13.50 1.93 0.44 18.57
2.43 2.10 0.33 13.58 1.88 0.55 22.63
2.50 2.22 0.28 11.20 2.02 0.48 19.20
14


<!-- PAGE 15 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
algorithm and the true plane of the pavement, which leads to cumulative
calculation errors.
Note：AE: Absolute error; RE: Relative error; MAE: Mean absolute
error; MRE: Mean relative error.
Fig. 19 shows the fitting line graph of the data in Table 7, where R2
represents the coefficient of determination, which is used to measure the
degree of fit of the model. The value of R2 ranges from 0 to 1. The closer
it is to 1, the better the fit between the measured results and the
reconstructed results. Fig. 19 illustrates that although MTD is relatively
scattered, the model’s predicted MTDp and MPDp achieved results
highly correlated with the measured MTDe. However, as discussed
earlier, the error between the texture depth value calculated based on
the MPDp index and the measured benchmark value is large, and the
reconstructed texture depth value based on the MPDp calculation is not
very reliable. Therefore, although a linear correlation exists between
MPDp and MTDe, the MTDp index is recommended for calculating the
reconstructed texture depth to obtain results closer to actual
measurements.
To further validate the reliability of the method, 30 measurement Fig. 20. MTD results of using the SPM and proposed method.
points were selected as samples for on-site testing on an AC-16 pave-
ment. Fig. 20 shows the experimental results of each measurement point
Table 8
using the sand patch method and the proposed method. Table 8 is a
Comparisons of the measurement and proposed methods.
comparison of the statistical data of the measurement results. Although
the texture depths significantly differ between the test areas of the Max MTD (mm) Min MTD (mm) Avg MTD (mm)
pavement itself, the mean values of MTD calculated using the two MTDe 1.15 0.81 0.9923
methods are similar. Therefore, when using the proposed method, MTDp MTDp 1.18 0.87 0.9903
can be used as an approximate reference value for MTDe. Error 0.03 0.06 0.0020
4.5. Comparisons among different texture measurement methods
Table 9
Comparison of different measurement methods.
The time required to gather texture information using our method,
photogrammetry, and the sand patch testing method was subjected to Methods Testing Requisite equipment
time
statistical analysis. Each method measured 10 sample points, and
Table 9 records the average time and necessary measuring instruments. Proposed method 5 min Camera, computer
Following the proposed pipeline, our model takes 7 images for depth Photogrammetry 15 min Photographic workstation, commercial
map reconstruction with a resolution of 3024 × 3024. Photogrammetry software
Sand patch method 10 min Specialized instruments
is employed using PhotoScan software for point cloud reconstruction.
From the Table 9, it is evident that our method takes significantly less
time to acquire asphalt pavement surface texture information compared
to photogrammetry and the traditional sand patch testing method. This
indicates that our proposed method efficiently performs texture mea-
surement for asphalt pavement. Moreover, our method does not require
Fig. 19. Correlation between (a) MTDe and MTDp (b) MTDe and MPDp.
15


<!-- PAGE 16 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
the additional use of a photographic workstation or commercial soft- number 202104), and Science and Technology Research and Develop-
ware for texture reconstruction, nor does it necessitate the preparation ment Program of China Railway Group Limited (Grant No: 2022-ZD-13).
of standard sand or experimental materials. It only requires easily
accessible cameras and computers for the measurements. References
5. Conclusion and future works [1] J.J. Henry, Evaluation of pavement friction characteristics, Transportation
Research Board 2000.
[2] P.I.A.o.R. Congresses, Report of the committee on surface characteristics, XVIII
In this study, a deep learning-based multiview stereo reconstruction World Road Congress, World Road Association Paris, 1987.
method is applied for the analysis of pavement texture depth. The model [3] B. Guan, J. Wu, C. Xie, J. Fang, H. Zheng, H. Chen, Influence of macrotexture and
microtexture on the skid resistance of aggregates, Adv. Mater. Sci. Eng. 2018
performance in pavement texture reconstruction based on multiple
(2018), https://doi.org/10.1155/2018/1437069.
views is discussed in detail, and a method for image processing and [4] A. Ruiz-Padillo, D.P. Ruiz, A.J. Torija, A ´ . Ramos-Ridao, Selection of suitable
metric calculation of depth maps is proposed. Users can obtain infor- alternatives to reduce the environmental impact of road traffic noise using a fuzzy
multi-criteria decision model, Environ. Impact Assess. Rev. 61 (2016) 8–18,
mation about pavement texture depth using a digital camera and a
https://doi.org/10.1016/j.eiar.2016.06.003.
personal computer, providing a new method for measuring pavement [5] E. Ascari, M. Cerchiai, L. Fredianelli, D. Melluso, F. Rampino, G. Licitra, Decision
texture depth. trees and labeling of low noise pavements as support for noise action plans,
The experimental results show that the well-trained deep learning Environ. Pollut. 337 (2023) 122487, https://doi.org/10.1016/j.
envpol.2023.122487.
network can reconstruct high-quality pavement texture models using a [6] G. Licitra, F. Artuso, M. Bernardini, A. Moro, F. Fidecaro, L. Fredianelli, Acoustic
transfer learning strategy, with validation results indicating an beamforming algorithms and their applications in environmental noise, Curr.
improved point cloud reconstruction precision of 0.77. Additionally, the Pollut. Rep. (2023) 1–24, https://doi.org/10.1007/s40726-023-00264-9.
[7] E. Ascari, M. Cerchiai, L. Fredianelli, G. Licitra, Statistical pass-by for unattended
reconstruction results from various viewpoints and image resolutions
road traffic noise measurement in an urban environment, Sensors 22 (22) (2022)
suggest that the model performs best with input images at 3024 × 3024 8767, https://doi.org/10.3390/s22228767.
resolution and using 7 multiview images. The model also exhibits [8] L. Gaetano, B. Marco, M. Ricardo, B. Francesco, F. Luca, CNOSSOS-EU coefficients
for electric vehicle noise emission, Appl. Acoust. 211 (2023) 109511, https://doi.
consistent reconstruction precision across different pavement materials,
org/10.1016/j.apacoust.2023.109511.
making it suitable for pavement texture depth evaluation on various [9] U. Sandberg, J. Ejsmont, Tyre/road noise. Reference book, (2002).
roads. Furthermore, the effectiveness evaluation results reveal that the [10] J.M.Alves Filho, A. Lenzi, P.H.T. Zannin, Effects of traffic composition on road
noise: a case study, Transp. Res. Part D: Transp. Environ. 9 (1) (2004) 75–80,
errors in calculating texture depth using the 3D indicator MTDp are
https://doi.org/10.1016/j.trd.2003.08.001.
generally within 0.1 mm, which is smaller than those obtained using the [11] F. Bianco, L. Fredianelli, F. Lo Castro, P. Gagliardi, F. Fidecaro, G. Licitra,
2D indicator MPDp. Choosing MTDp as the calculation metric meets the Stabilization of ap-u sensor mounted on a vehicle for measuring the acoustic
impedance of road surfaces, Sensors 20 (5) (2020) 1239, https://doi.org/10.3390/
standards for MTD evaluation in highway engineering testing. Finally, a
s20051239.
comparison with two other commonly used texture measurement [12] F.G. Pratico`, R. Fedele, G. Pellicano, Monitoring road acoustic and mechanical
methods highlights the superior efficiency and convenience of the pro- performance, in: European Workshop on Structural Health Monitoring, Springer,,
posed approach. 2020, pp. 594–602, https://doi.org/10.1007/978-3-030-64594-6_58.
[13] L. Teti, G. de Leo´n, L.G. Del Pizzo, A. Moro, F. Bianco, L. Fredianelli, G. Licitra,
Due to the difficulty of collecting RGB-D datasets and their suscep- Modelling the acoustic performance of newly laid low-noise pavements, Constr.
tibility to environmental factors, future work will focus on improving Build. Mater. 247 (2020) 118509, https://doi.org/10.1016/j.
the collection methods of training datasets. Additionally, using multiple conbuildmat.2020.118509.
[14] L. Del Pizzo, L. Teti, A. Moro, F. Bianco, L. Fredianelli, G. Licitra, Influence of
cameras for fixed-angle shooting to derive camera poses with real-scale
texture on tyre road noise spectra in rubberized pavements, Appl. Acoust. 159
information and facilitate the direct reconstruction of depth maps with (2020) 107080, https://doi.org/10.1016/j.apacoust.2019.107080.
absolute depth values will be considered. Finally, the network must be [15] F.G. Pratico`, On the dependence of acoustic performance on pavement
characteristics, Transp. Res. Part D: Transp. Environ. 29 (2014) 79–87, https://doi.
improved to extract multiscale features from nonuniform mixtures and
org/10.1016/j.trd.2014.04.004.
better predict the texture depth of different pavement types. [16] G. de Leo´n, L.G. Del Pizzo, L. Teti, A. Moro, F. Bianco, L. Fredianelli, G. Licitra,
Evaluation of tyre/road noise and texture interaction on rubberised and
conventional pavements using CPX and profiling measurements, Road. Mater.
CRediT authorship contribution statement Pavement Des. 21 (sup1) (2020) S91–S102, https://doi.org/10.1080/
14680629.2020.1735493.
Dan Han-Cheng: Writing – review & editing, Supervision, Re- [17] L.G. Del Pizzo, F. Bianco, A. Moro, G. Schiaffino, G. Licitra, Relationship between
tyre cavity noise and road surface characteristics on low-noise pavements, Transp.
sources, Project administration, Funding acquisition, Conceptualization.
Res. Part D: Transp. Environ. 98 (2021) 102971, https://doi.org/10.1016/j.
Lu Bingjie: Writing – original draft, Visualization, Validation, Meth- trd.2021.102971.
odology, Formal analysis, Data curation, Conceptualization. Li Mengyu: [18] A. Del Pizzo, F. Bianco, L. Teti, A. Moro, G. Licitra, A new approach for the
Writing – review & editing, Software. evaluation of the relationship between road texture and rolling noise, 25th
International Congress on Sound and Vibration (ICSV25), Hiroshima, Japan, 2018.
[19] D. Chen, C. Ling, T. Wang, Q. Su, A. Ye, Prediction of tire-pavement noise of porous
Declaration of Competing Interest asphalt mixture based on mixture surface texture level and distributions, Constr.
Build. Mater. 173 (2018) 801–810, https://doi.org/10.1016/j.
conbuildmat.2018.04.062.
The authors declare that they have no known competing financial [20] S.J. Hong, S.-W. Park, S.W. Lee, Tire-pavement noise prediction using asphalt
interests or personal relationships that could have appeared to influence pavement texture, KSCE J. Civ. Eng. 22 (9) (2018) 3358–3362, https://doi.org/
the work reported in this paper. 10.1007/s12205-018-9501-3.
[21] M. Miljkovi´c, M. Radenberg, C. Gottaut, Characterization of noise-reducing
capacity of pavement by means of surface texture parameters, J. Mater. Civ. Eng.
Data Availability 26 (2) (2014) 240–249, https://doi.org/10.1061/(ASCE)MT.1943-5533.0000821.
[22] F.G. Pratico`, F. Anfosso-L´ed´ee, Trends and issues in mitigating traffic noise through
quiet pavements, Procedia-Soc. Behav. Sci. 53 (2012) 203–212, https://doi.org/
The self-created dataset in this article can provide samples upon
10.1016/j.sbspro.2012.09.873.
reasonable request, but the code will not be shared due to intellectual [23] ASTM, Standard test method for measuring pavement macrotexture depth using a
property considerations. volumetric technique, Designation: E 965–96 (2006). https://www.astm.org/e09
65–15r19.html.
[24] M.-T. Do, V. Cerezo, Road surface texture and skid resistance, Surf. Topogr.:
Acknowledgments Metrol. Prop. 3 (4) (2015) 043001, https://doi.org/10.1088/2051-672x/3/4/
043001.
[25] S. Torbruegge, B. Wies, Characterization of pavement texture by means of height
H. D. wants to thank the support from the National Natural Science
difference correlation and relation to wet skid resistance, J. Traffic Transp. Eng.
Foundation of China (grant numbers 52278468 and U22A20235), the (Engl. Ed. ) 2 (2) (2015) 59–67, https://doi.org/10.1016/j.jtte.2015.02.001.
Hunan Transportation Science and Technology Foundation (CN) (grant
16


<!-- PAGE 17 -->
H.-C. Dan et al. C o n s t r u c t i o n a n d B u i l d i n g M a t e r i a l s 412 (2024) 134837
[26] Z. Tong, J. Gao, A. Sha, L. Hu, S. Li, Convolutional neural network for asphalt [49] Q. Zhu, C. Min, Z. Wei, Y. Chen, G. Wang, Deep learning for multi-view stereo via
pavement surface texture analysis, Comput. Civ. Infrastruct. Eng. 33 (12) (2018) plane sweep: a survey, arXiv Prepr. arXiv 2106 (2021) 15328, https://doi.org/
1056–1072, https://doi.org/10.1111/mice.12406. 10.48550/arXiv.2106.15328.
[27] A. Ioannidou, E. Chatzilari, S. Nikolopoulos, I. Kompatsiaris, Deep learning [50] H. Aanæs, R.R. Jensen, G. Vogiatzis, E. Tola, A.B. Dahl, Large-scale data for
advances in computer vision with 3d data: A survey, ACM Comput. Surv. (CSUR) multiple-view stereopsis, Int. J. Comput. Vis. 120 (2016) 153–168, https://doi.
50 (2) (2017) 1–38, https://doi.org/10.1145/3042064. org/10.1007/s11263-016-0902-9.
[28] L. Gao, M. Liu, Z. Wang, J. Xie, S. Jia, Correction of texture depth of porous asphalt [51] Y. Yao, Z. Luo, S. Li, J. Zhang, Y. Ren, L. Zhou, T. Fang, L. Quan, Blendedmvs: a
pavement based on CT scanning technique, Constr. Build. Mater. 200 (2019) large-scale dataset for generalized multi-view stereo networks, Proc. IEEE/CVF
514–520, https://doi.org/10.1016/j.conbuildmat.2018.12.154. Conf. Comput. Vis. Pattern Recognit. (2020) 1790–1799, https://doi.org/10.1109/
[29] S. Granshaw, Close range photogrammetry: principles, methods and applications, cvpr42600.2020.00186.
Wiley Online Libr. (2010), https://doi.org/10.1111/j.1477-9730.2010.00574_1.x. [52] A. Dai, A.X. Chang, M. Savva, M. Halber, T. Funkhouser, M. Nießner, Scannet:
[30] H.-C. Dan, G.-W. Bai, Z.-H. Zhu, X. Liu, W. Cao, An improved computation method richly-annotated 3d reconstructions of indoor scenes, Proc. IEEE Conf. Comput.
for asphalt pavement texture depth based on multiocular vision 3D reconstruction Vis. Pattern Recognit. (2017) 5828–5839, https://doi.org/10.1109/cvpr.2017.261.
technology, Constr. Build. Mater. 321 (2022) 126427, https://doi.org/10.1016/j. [53] A. Grunnet-Jepsen, D. Tong, Depth post-processing for intel® realsense™ d400
conbuildmat.2022.126427. depth cameras, N. Technol. Group, Intel. Corp. (3) (2018).
[31] A. El Gendy, A. Shalaby, M. Saleh, G.W. Flintsch, Stereo-vision applications to [54] J. Ku, A. Harakeh, S.L. Waslander, In defense of classical image processing: fast
reconstruct the 3D texture of pavement surface, Int. J. Pavement Eng. 12 (03) depth completion on the cpu, in: 15th Conference on Computer and Robot Vision
(2011) 263–273, https://doi.org/10.1080/10298436.2010.546858. (CRV), 2018, IEEE,, 2018, pp. 16–22, https://doi.org/10.1109/crv.2018.00013.
[32] M.S. Medeiros, B.S. Underwood, C. Castorena, T. Rupnow, M. Rawls, 3D [55] E.J. Kirkland, E.J. Kirkland, Bilinear interpolation, Adv. Comput. Electron Microsc.
measurement of pavement macrotexture using digital stereoscopic vision, 2016. (2010) 261–263, https://doi.org/10.1007/978-1-4419-6533-2_12.
https://trid.trb.org/view/1394028. [56] L. Agisoft, Metashape python Ref., Release 1 (0) (2020) 1–199. https://www.
[33] D. Chen, Evaluating asphalt pavement surface texture using 3D digital imaging, Int. agisoft.com/pdf/metashape_python_api_2_0_2.pdf.
J. Pavement Eng. 21 (4) (2020) 416–427, https://doi.org/10.1080/ [57] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, S. Savarese, Generalized
10298436.2018.1483503. intersection over union: a metric and a loss for bounding box regression, Proc.
[34] X. Sun, J. Huang, W. Liu, M. Xu, Pavement crack characteristic detection based on IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2019) 658–666, https://doi.org/
sparse representation, EURASIP J. Adv. Signal Process. 2012 (1) (2012) 1–11, 10.1109/cvpr.2019.00075.
https://doi.org/10.1186/1687-6180-2012-191. [58] F. Zhuang, Z. Qi, K. Duan, D. Xi, Y. Zhu, H. Zhu, H. Xiong, Q. He, A comprehensive
[35] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, nature 521 (7553) (2015) 436–444, survey on transfer learning, Proc. IEEE 109 (1) (2020) 43–76, https://doi.org/
https://doi.org/10.1038/nature14539. 10.1109/JPROC.2020.3004555.
[36] S. Dong, S. Han, C. Wu, O. Xu, H. Kong, Asphalt pavement macrotexture [59] P. Peng, J. Wang, How to fine-tune deep neural networks in few-shot learning?,
reconstruction from monocular image based on deep convolutional neural arXiv preprint arXiv:2012.00204 (2020). https://doi.org/10.48550/arXiv.2012.
network, Comput. Civ. Infrastruct. Eng. 37 (13) (2022) 1754–1768, https://doi. 00204.
org/10.1111/mice.12878. [60] H. Xu, Z. Zhou, Y. Qiao, W. Kang, Q. Wu, Self-supervised multi-view stereo via
[37] C. Liu, J. Li, J. Gao, D. Yuan, Z. Gao, Z. Chen, Three-dimensional texture effective co-segmentation and data-augmentation, Proc. AAAI Conf. Artif. Intell.
measurement using deep learning and multi-view pavement images, Measurement (2021) 3030–3038, https://doi.org/10.1609/aaai.v35i4.16411.
172 (2021) 108828, https://doi.org/10.1016/j.measurement.2020.108828. [61] Z. Yu, S. Gao, Fast-mvsnet: Sparse-to-dense multi-view stereo with learned
[38] N. Chen, Z. Xu, Z. Liu, Y. Chen, Y. Miao, Q. Li, Y. Hou, L. Wang, Data augmentation propagation and gauss-newton refinement, Proc. IEEE/CVF Conf. Comput. Vis.
and intelligent recognition in pavement texture using a deep learning, IEEE Trans. Pattern Recognit. (2020) 1949–1958, https://doi.org/10.1109/
Intell. Transp. Syst. 23 (12) (2022) 25427–25436, https://doi.org/10.1109/ cvpr42600.2020.00202.
TITS.2022.3140586. [62] J.L. Schonberger, J.-M. Frahm, Structure-from-motion revisited, Proc. IEEE Conf.
[39] M. Emoto, Depth perception and induced accommodation responses while Comput. Vis. Pattern Recognit. (2016) 4104–4113, https://doi.org/10.1109/
watching high spatial resolution two-dimensional TV images, Displays 60 (2019) cvpr.2016.445.
24–29, https://doi.org/10.1016/j.displa.2019.08.005. [63] J.L. Scho¨nberger, E. Zheng, J.-M. Frahm, M. Pollefeys, Pixelwise view selection for
[40] G. Yang, Q.J. Li, Y. Zhan, Y. Fei, A. Zhang, Convolutional neural network–based unstructured multi-view stereo, Computer Vision–ECCV 2016: 14th European
friction model using pavement texture data, J. Comput. Civ. Eng. 32 (6) (2018) Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part
04018052, https://doi.org/10.1061/(ASCE)CP.1943-5487.0000797. III 14, Springer, 2016, pp. 501–518. https://doi.org/10.1007/978–3-319–4648
[41] M.Z. Bashar, C. Torres-Machi, Deep learning for estimating pavement roughness 7-9_31.
using synthetic aperture radar data, Autom. Constr. 142 (2022) 104504, https:// [64] K.G. Derpanis, Overv. RANSAC Algorithm, Image Rochester NY 4 (1) (2010) 2–3.
doi.org/10.1016/j.autcon.2022.104504. https://rmozone.com/snapshots/2015/07/cdg-room-refs/ransac.pdf.
[42] S. Pan, H. Yan, Z. Liu, N. Chen, Y. Miao, Y. Hou, Automatic pavement texture [65] J. Chen, X. Huang, B. Zheng, R. Zhao, X. Liu, Q. Cao, S. Zhu, Real-time
recognition using lightweight few-shot learning, Philos. Trans. R. Soc. A 381 identification system of asphalt pavement texture based on the close-range
(2254) (2023) 20220166, https://doi.org/10.1098/rsta.2022.0166. photogrammetry, Constr. Build. Mater. 226 (2019) 910–919, https://doi.org/
[43] G. Wang, K.C. Wang, G. Yang, Deep learning based image reconstruction at any 10.1016/j.conbuildmat.2019.07.321.
speeds for faster pavement texture measurement using 3D laser technology, Int. J. [66] L. Hu, D. Yun, Z. Liu, S. Du, Z. Zhang, Y. Bao, Effect of three-dimensional
Pavement Eng. 24 (2) (2023) 2269461, https://doi.org/10.1080/ macrotexture characteristics on dynamic frictional coefficient of asphalt pavement
10298436.2023.2269461. surface, Constr. Build. Mater. 126 (2016) 720–729, https://doi.org/10.1016/j.
[44] Y. Furukawa, C. Herna´ndez, Multi-view stereo: a tutorial, Found. Trends® Comput. conbuildmat.2016.09.088.
Graph. Vis. 9 (1-2) (2015) 1–148, https://doi.org/10.1561/0600000052. [67] C. Barnes, E. Shechtman, A. Finkelstein, D.B. Goldman, PatchMatch: A randomized
[45] X. Wang, C. Wang, B. Liu, X. Zhou, L. Zhang, J. Zheng, X. Bai, Multi-view stereo in correspondence algorithm for structural image editing, in: ACM Trans. Graph, 28,
the deep learning era: a comprehensive review, Displays 70 (2021) 102102, .,, 2009, https://doi.org/10.1145/3596711.3596777.
https://doi.org/10.1016/j.displa.2021.102102. [68] M. China, Highway Performance Assessment Standards: JTG H20–2007, Standard
[46] M. Ji, J. Gall, H. Zheng, Y. Liu, L. Fang, Surfacenet: an end-to-end 3d neural Prees of China, Beijing, China, 2007.
network for multiview stereopsis, Proc. IEEE Int. Conf. Comput. Vis. (2017) [69] S. Chen, X. Liu, H. Luo, J. Yu, F. Chen, Y. Zhang, T. Ma, X. Huang, A state-of-the-art
2307–2315, https://doi.org/10.1109/iccv.2017.253. review of asphalt pavement surface texture and its measurement techniques,
[47] Y. Yao, Z. Luo, S. Li, T. Fang, L. Quan, Mvsnet: depth inference for unstructured J. Road. Eng. 2 (2) (2022) 156–180, https://doi.org/10.1016/j.jreng.2022.05.003.
multi-view stereo, Proc. Eur. Conf. Comput. Vis. (ECCV (2018) 767–783, https:// [70] F. Pratico`, R. Vaiana, A study on the relationship between mean texture depth and
doi.org/10.1007/978-3-030-01237-3_47. mean profile depth of asphalt pavements, Constr. Build. Mater. 101 (2015) 72–79,
[48] F. Wang, S. Galliani, C. Vogel, P. Speciale, M. Pollefeys, Patchmatchnet: learned https://doi.org/10.1016/j.conbuildmat.2015.10.021.
multi-view patchmatch stereo, Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (2021) 14194–14203, https://doi.org/10.1109/cvpr46437.2021.01397.
17
