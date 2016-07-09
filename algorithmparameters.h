#pragma once
#include "managed.h"

//cost function
enum {
    PM_COST = 0,
    CENSUS_TRANSFORM = 1,
    ADAPTIVE_CENSUS = 2,
    CENSUS_SELFSIMILARITY = 3,
    PM_SELFSIMILARITY = 4,
    ADCENSUS = 5,
    ADCENSUS_SELFSIMILARITY = 6,
    SPARSE_CENSUS = 7
};

//cost combination
enum { COMB_ALL = 0, COMB_BEST_N = 1, COMB_ANGLE = 2, COMB_GOOD = 3 };

struct AlgorithmParameters : public Managed{

    AlgorithmParameters():
        algorithm          (0), // algorithm cost type
        max_disparity      (256.0f), // maximal disparity value CUDA
        min_disparity      (0.0f), // minimum disparity value (default 0) CUDA
        box_hsize          (19), // filter kernel width CUDA
        box_vsize          (19), // filter kernel height CUDA
        tau_color          (10), // PM_COST max. threshold for color CUDA
        tau_gradient       (2.0f), // PM_COST max. threshold for gradient CUDA
        alpha              (0.9f), // PM_COST weighting between color and gradient CUDA
        gamma              (10.0f), // parameter for weight function (used e.g. in PM_COST) CUDA
        border_value       (-1), // what value should pixel at extended border get (constant or replicate -1)
        iterations         (8), // number of iterations
        color_processing   (false), // use color processing or not (otherwise just grayscale processing)
        dispTol            (1.0), //PM Stereo: 1, PM Huber: 0.5
        normTol            (0.1f), // 0.1 ... about 5.7 degrees
        census_epsilon     (2.5f), //for census transform
        self_similarity_n  (50), // number of pixels considered for self similarity
        cam_scale          (1.0f), //used to rescale K in case of rescaled image size
        num_img_processed  (1), //number of images that are processed as reference images
        costThresh         (40.0f), // threshold to decide whether disparity/depth is valid or not
         good_factor       (1.5f), // for cost aggregation/combination good: factor for truncation CUDA
        n_best             (2), // CUDA
        cost_comb          (1), // CUDA
        viewSelection      (true), // Default has to be false, or user has no way to disable view selection
        depthMin           (-1.0f), // CUDA
        depthMax           (-1.0f), // CUDA
        min_angle          (5.0f), // CUDA
        max_angle          (45.0f), // CUDA
        no_texture_sim     (0.9f), // CUDA
        no_texture_per     (0.6f), // CUDA
        max_views          (9) {}
    int algorithm; // algorithm cost type
    float max_disparity; // maximal disparity value CUDA
    float min_disparity; // minimum disparity value (default 0) CUDA
    int box_hsize; // filter kernel width CUDA
    int box_vsize; // filter kernel height CUDA
    float tau_color; // PM_COST max. threshold for color CUDA
    float tau_gradient; // PM_COST max. threshold for gradient CUDA
    float alpha; // PM_COST weighting between color and gradient CUDA
    float gamma; // parameter for weight function (used e.g. in PM_COST) CUDA
    int border_value; // what value should pixel at extended border get (constant or replicate -1)
    int iterations; // number of iterations
    bool color_processing; // use color processing or not (otherwise just grayscale processing)
    float dispTol; //PM Stereo: 1, PM Huber: 0.5
    float normTol; // 0.1 ... about 5.7 degrees
    float census_epsilon; //for census transform
    int self_similarity_n; // number of pixels considered for self similarity
    float cam_scale; //used to rescale K in case of rescaled image size
    int num_img_processed; //number of images that are processed as reference images
    float costThresh; // threshold to decide whether disparity/depth is valid or not
    float good_factor; // for cost aggregation/combination good: factor for truncation CUDA
    int n_best; // CUDA
    int cost_comb; // CUDA
    bool viewSelection;
    float depthMin; // CUDA
    float depthMax; // CUDA
    float min_angle; // CUDA
    float max_angle; // CUDA
    float no_texture_sim; // CUDA
    float no_texture_per; // CUDA
    unsigned int max_views;
    // hack XXX
    int cols;
    int rows;
};
