# Gipuma


Source code for the paper:

S. Galliani, K. Lasinger and K. Schindler, [Massively Parallel Multiview Stereopsis by Surface Normal Diffusion](http://www.prs.igp.ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/galliani-lasinger-iccv15.pdf), ICCV 2015

## Authors
- [Silvano Galliani](silvano.galliani@geod.baug.ethz.ch)
- [Katrin Lasinger](katrin.lasinger@geod.baug.ethz.ch)


**IMPORTANT**: If you use this software please cite the following in any resulting publication:
```
@inproceedings{galliani15_iccv,
    Title={Massively Parallel Multiview Stereopsis by Surface Normal Diffusion},
    Author={Galliani, Silvano and Lasinger, Katrin and Schindler, Konrad},
    Booktitle = {Proc. of the IEEE International Conference on Computer Vision (ICCV)},
    Year = {2015},
    Month = {December}
}
```

## Requirements:
 - [Cuda](https://developer.nvidia.com/cuda-downloads) >= 6.0
 - Nvidia video card with compute capability at least 3.0, see https://en.wikipedia.org/wiki/CUDA#Supported_GPUs
 - [Opencv](http://opencv.org) >= 2.4
 - [cmake](http://cmake.org)
 
## Tested Operating Systems
 - Ubuntu GNU/Linux 14.04
 - Windows with Visual Studio 2012/2013 (it's working directly with cmake)

## How to compile it
Just use cmake for both Windows and Linux.
For linux it gets as easy as:
```bash
cmake .
make
```

## How does it work?
Gipuma itself is only a matcher. It will compute a depthmap with respect to the specified reference camera.

For each camera gipuma need to compute the _noisy_ depthmap. The final fusion of depthmap is obtained with [fusibile](https://github.com/kysucix/fusibile)

## Faq
 TODO
## Examples
 TODO
### Middlebury
 TODO

### DTU dataset
http://roboimagedata.compute.dtu.dk/?page_id=24
### Strecha dataset
## Configuration

### Necessary Parameters
The minimum information Gipuma needs is camera information and image list
Gipuma relies on known camera information. You can provide this information in 3 different ways:

| Parameter Name | Syntax | Comment |
| -------------- | --------------------------------- | --------------- |
| pmvs_folder | -pmvs_folder \<folder\> | The easiest way is to point gipuma to the output of VisualSFM for pmvs. Images will be taken from \<pmvs\>/visualize and cameras from \<pmvs\>/txt/ Additionally 3d points in \<pmvs\>/bundle.rd.out |
| krt_file | -krt_file \<file\> | In this way camera information is read from a file as specified by Middlebury benchmark http://vision.middlebury.edu/mview/data/ |
| p_folder | -p_folder \<folder\> | This parameter expects a folder with a list of textfiles containing the P matrix on 3 lines and the same filename as the images but with ".P." appended |

To specify an image list in case a pmvs folder is not specify a list of filename is needed with an image folder.
For example:
```
./gipuma image1.jpg image2.jpg -img_folder images/ -krt_file Temple.txt
```

### Important Parameters
 **gipuma** comes with a good-enough parameter sets, but to obtain best results some setting can be optimized


| Parameter Name | Syntax and default Value | Comment |
| -------------- | ------------------------ | --------------- |
| camera_idx | --camera_idx=00 | This value set the reference camera to be used. The resulting depth map will be computed with respect to this camera |
| blocksize | --blocksize=19 | It's an important value that affects the patch size used for cost computation. Its value is highly dependent on the image resolution and the object size. Suggested value range from 9 for Middlebury size image (640x480) to 25 for DTU dataset (1600x1200) |
| iterations | --iterations=8 | This parameter controls the amount of normal diffusion employed for the reconstruction. A value bigger than 8 rarely improves the reconstruction. Recuding its value trades-off runtime for quality of reconstruction |
| min_angle and max_angle | --min_angle=5 --max_angle=45 | The reference camera will be matched with respect to other cameras that have a intersection angle of the principal ray withing the specified range. For datasets with many images a range 10-30 degree is suggested. For dataset with a sparse set of images (as middlebury SparseRing) a bigger range is needed |
| max_views | --max_views=9 | In case more than max_views cameras survive the angle selection, a random set of max_views cameras is considered. |
| depth_min and depth_max | --depth_min=-1 --depth_max=-1 | This value set the minimum depth for the reconstruction in world coordinate. In case it is not set it is computed as the minimum and maximum range for all the camera from the 3d points inside the specified pmvs directory. In case only a list of P matrices is given, it is computed from the minimum and maximum range of depth obtained when setting the viewing angle to the minimum and maximum specified |

### Optional Parameters
 The following parameters _can_ be tweaked at will but in our experience they do not affect the overall reconstruction
 TODO
