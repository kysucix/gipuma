#include "main.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#endif

#include <vector>
#include <string>
#include <iostream>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir
#include <dirent.h> // opendir()

#include "algorithmparameters.h"
#include "globalstate.h"
#include "gipuma.h"

#include "fileIoUtils.h"
#include "cameraGeometryUtils.h"
#include "mathUtils.h"
#include "displayUtils.h"
#include "groundTruthUtils.h"


// file format from https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
static void parse_bundler_3d_points ( const char *filename, std::vector<Vec3f> &point_list)
{
    unsigned int num_cameras, num_points;

    ifstream myfile;
    myfile.open(filename, ifstream::in);
    char line[512];
    if(myfile.peek()=='#')
        myfile.getline(line,512); // skip comment
    myfile.getline(line,512); //     <num_cameras> <num_points>   [two integers]
    sscanf ( line, "%u %u", &num_cameras, &num_points);
    for (size_t i=0; i< num_cameras; i++)
    {
        myfile.getline(line,512); //         <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]

        myfile.getline(line,512); //     <R>             [a 3x3 matrix representing the camera rotation]
        myfile.getline(line,512); //     <R>             [a 3x3 matrix representing the camera rotation]
        myfile.getline(line,512); //     <R>             [a 3x3 matrix representing the camera rotation]
        myfile.getline(line,512); //     <t>             [a 3-vector describing the camera translation]
    }
    // Now parse point list
    /*
     * Each point entry has the form:
     *
     *
     *    <position>      [a 3-vector describing the 3D position of the point]
     *    <color>         [a 3-vector describing the RGB color of the point]
     *    <view list>     [a list of views the point is visible in]
     */
    point_list.resize (num_points);
    for (size_t i=0; i< num_points; i++)
    {
        Vec3f X;
        myfile.getline(line, 512); //    <position>      [a 3-vector describing the 3D position of the point]
        sscanf ( line, "%f %f %f", &X[0], &X[1], &X[2]);
        myfile.getline(line, 512); //    <color>         [a 3-vector describing the RGB color of the point]
        myfile.getline(line, 512); //    <view list>     [a list of views the point is visible in]
        //if (i<10)
            //printf("3d point is %f %f %f\n", X[0], X[1], X[2]);
        point_list[i] = X;
    }
}

static void from_bundler_get_range (CameraParameters &cameraParams,
                                    AlgorithmParameters &algParams,
                                    const char *filename)
{
    std::vector<Vec3f> point_list;
    vector<Camera> &cameras = cameraParams.cameras;
    parse_bundler_3d_points(filename, point_list);
    float min_depth = 9999;
    float max_depth = 0;
    // For each camera
    for ( size_t i = 1; i < cameras.size (); i++ ) {
        // For each point
        for (auto X : point_list)
        {
            // Compute euclidean distance to camera center
            float depth = cv::norm(X - cameras[i].C);
            min_depth = std::min(depth, min_depth);
            max_depth = std::max(depth, max_depth);
        }
    }
    if (algParams.depthMin == -1)
        algParams.depthMin = min_depth - min_depth*0.4;
    if (algParams.depthMax == -1)
        algParams.depthMax = max_depth + max_depth*0.2;

    // For each camera compute minimum and maximum depth
}


static void print_help (char **argv)
{
    printf ( "\nUsage: %s <im1> <im2> ... [--parameter=<parameter>]\n", argv[0] );
}

static void get_directory_entries(
                           const char *dirname,
                           vector<string> &directory_entries)
{
    DIR *dir;
    struct dirent *ent;

    // Open directory stream
    dir = opendir (dirname);
    if (dir != NULL) {
        //cout << "Dirname is " << dirname << endl;
        //cout << "Dirname type is " << ent->d_type << endl;
        //cout << "Dirname type DT_DIR " << DT_DIR << endl;

        // Print all files and directories within the directory
        while ((ent = readdir (dir)) != NULL) {
            //cout << "INSIDE" << endl;
            //if(ent->d_type == DT_DIR)
            {
                char* name = ent->d_name;
                if(strcmp(name,".") == 0 || strcmp(ent->d_name,"..") == 0)
                    continue;
                //printf ("dir %s/\n", name);
                directory_entries.push_back(string(name));
            }
        }

        closedir (dir);

    } else {
        // Could not open directory
        printf ("Cannot open directory %s\n", dirname);
        exit (EXIT_FAILURE);
    }
    sort ( directory_entries.begin (), directory_entries.end () );
}

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, - algorithm parameters
 */
static int getParametersFromCommandLine ( int argc,
                                          char** argv,
                                          InputFiles &inputFiles,
                                          OutputFiles &outputFiles,
                                          AlgorithmParameters &algParams,
                                          GTcheckParameters &gt_parameters
                                        )
{
    int camera_idx = 0;
    const char* algorithm_opt     = "--algorithm=";
    const char* maxdisp_opt       = "--max-disparity=";
    const char* blocksize_opt     = "--blocksize=";
    const char* cost_tau_color_opt   = "--cost_tau_color=";
    const char* cost_tau_gradient_opt   = "--cost_tau_gradient=";
    const char* cost_alpha_opt   = "--cost_alpha=";
    const char* cost_gamma_opt   = "--cost_gamma=";
    const char* disparity_tolerance_opt = "--disp_tol=";
    const char* normal_tolerance_opt = "--norm_tol=";
    const char* border_value = "--border_value=";
    const char* gtDepth_divFactor_opt = "--gtDepth_divisionFactor=";
    const char* gtDepth_tolerance_opt = "--gtDepth_tolerance=";
    const char* gtDepth_tolerance2_opt = "--gtDepth_tolerance2=";
    const char* colorProc_opt = "-color_processing";
    const char* num_iterations_opt = "--iterations=";
    const char* self_similariy_n_opt = "--ss_n=";
    const char* ct_epsilon_opt = "--ct_eps=";
    const char* cam_scale_opt = "--cam_scale=";
    const char* num_img_processed_opt = "--num_img_processed=";
    const char* n_best_opt = "--n_best=";
    const char* cost_comb_opt = "--cost_comb=";
    const char* cost_good_factor_opt = "--good_factor=";
    const char* depth_min_opt = "--depth_min=";
    const char* depth_max_opt = "--depth_max=";
    //    const char* scale_opt         = "--scale=";
    const char* outputPath_opt = "-output_folder";
    const char* calib_opt = "-calib_file";
    const char* gt_opt = "-gt";
    const char* gt_nocc_opt = "-gt_nocc";
    const char* occl_mask_opt = "-occl_mask";
    const char* gt_normal_opt = "-gt_normal";
    const char* images_input_folder_opt = "-images_folder";
    const char* p_input_folder_opt = "-p_folder";
    const char* krt_file_opt = "-krt_file";
    const char* camera_input_folder_opt = "-camera_folder";
    const char* bounding_folder_opt = "-bounding_folder";
    const char* viewSelection_opt = "-view_selection";
    const char* initial_seed_opt = "--initial_seed";
    const char* min_angle_opt = "--min_angle=";
    const char* max_angle_opt = "--max_angle=";
    const char* no_texture_sim_opt = "--no_texture_sim";
    const char* no_texture_per_opt = "--no_texture_per";
    const char* max_views_opt = "--max_views=";
    const char* pmvs_folder_opt = "--pmvs_folder";
    const char* camera_idx_opt = "--camera_idx=";

    //read in arguments
    for ( int i = 1; i < argc; i++ ) {
        if ( argv[i][0] != '-' )
        {
            inputFiles.img_filenames.push_back ( argv[i] );
        }
        else if ( strncmp ( argv[i], algorithm_opt, strlen ( algorithm_opt ) ) == 0 )
        {
            char* _alg = argv[i] + strlen ( algorithm_opt );
            algParams.algorithm = strcmp ( _alg, "pm" ) == 0 ? PM_COST :
            strcmp ( _alg, "ct" ) == 0 ? CENSUS_TRANSFORM :
            strcmp ( _alg, "sct" ) == 0 ? SPARSE_CENSUS :
            strcmp ( _alg, "ct_ss" ) == 0 ? CENSUS_SELFSIMILARITY :
            strcmp ( _alg, "adct" ) == 0 ? ADCENSUS :
            strcmp ( _alg, "adct_ss" ) == 0 ? ADCENSUS_SELFSIMILARITY :
            strcmp ( _alg, "pm_ss" ) == 0 ? PM_SELFSIMILARITY : -1;
            if ( algParams.algorithm < 0 )
            {
                printf ( "Command-line parameter error: Unknown stereo algorithm\n\n" );
                print_help (argv);
                return -1;
            }
        }
        else if ( strncmp ( argv[i], cost_comb_opt, strlen ( cost_comb_opt ) ) == 0 )
        {
            char* _alg = argv[i] + strlen ( algorithm_opt );
            algParams.cost_comb = strcmp ( _alg, "all" ) == 0 ? COMB_ALL :
            strcmp ( _alg, "best_n" ) == 0 ? COMB_BEST_N :
            strcmp ( _alg, "angle" ) == 0 ? COMB_ANGLE :
            strcmp ( _alg, "good" ) == 0 ? COMB_GOOD : -1;
            if ( algParams.cost_comb < 0 )
            {
                printf ( "Command-line parameter error: Unknown cost combination method\n\n" );
                print_help (argv);
                return -1;
            }
        }
        else if ( strncmp ( argv[i], maxdisp_opt, strlen ( maxdisp_opt ) ) == 0 )
        {
            if ( sscanf ( argv[i] + strlen ( maxdisp_opt ), "%f", &algParams.max_disparity ) != 1 ||
                 algParams.max_disparity < 1  )
            {
                printf ( "Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer \n" );
                print_help (argv);
                return -1;
            }
        }
        else if ( strncmp ( argv[i], blocksize_opt, strlen ( blocksize_opt ) ) == 0 )
        {
            int k_size;
            if ( sscanf ( argv[i] + strlen ( blocksize_opt ), "%d", &k_size ) != 1 ||
                 k_size < 1 || k_size % 2 != 1 )
            {
                printf ( "Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n" );
                return -1;
            }
            algParams.box_hsize = k_size;
            algParams.box_vsize = k_size;
        }
        else if ( strncmp ( argv[i], cost_good_factor_opt, strlen ( cost_good_factor_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_good_factor_opt ), "%f", &algParams.good_factor );
        }
        else if ( strncmp ( argv[i], cost_tau_color_opt, strlen ( cost_tau_color_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_tau_color_opt ), "%f", &algParams.tau_color );
        }
        else if ( strncmp ( argv[i], cost_tau_gradient_opt, strlen ( cost_tau_gradient_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_tau_gradient_opt ), "%f", &algParams.tau_gradient );
        }
        else if ( strncmp ( argv[i], cost_alpha_opt, strlen ( cost_alpha_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_alpha_opt ), "%f", &algParams.alpha );
        }
        else if ( strncmp ( argv[i], cost_gamma_opt, strlen ( cost_gamma_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_gamma_opt ), "%f", &algParams.gamma );
        }
        else if ( strncmp ( argv[i], border_value, strlen ( border_value ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( border_value ), "%d", &algParams.border_value );
        }
        else if ( strncmp ( argv[i], num_iterations_opt, strlen ( num_iterations_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( num_iterations_opt ), "%d", &algParams.iterations );
        }
        else if ( strncmp ( argv[i], disparity_tolerance_opt, strlen ( disparity_tolerance_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( disparity_tolerance_opt ), "%f", &algParams.dispTol );
        }
        else if ( strncmp ( argv[i], normal_tolerance_opt, strlen ( normal_tolerance_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( normal_tolerance_opt ), "%f", &algParams.normTol );
        }
        else if ( strncmp ( argv[i], self_similariy_n_opt, strlen ( self_similariy_n_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( self_similariy_n_opt ), "%d", &algParams.self_similarity_n );
        }
        else if ( strncmp ( argv[i], ct_epsilon_opt, strlen ( ct_epsilon_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( ct_epsilon_opt ), "%f", &algParams.census_epsilon );
        }
        else if ( strncmp ( argv[i], cam_scale_opt, strlen ( cam_scale_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cam_scale_opt ), "%f", &algParams.cam_scale );
        }
        else if ( strncmp ( argv[i], num_img_processed_opt, strlen ( num_img_processed_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( num_img_processed_opt ), "%d", &algParams.num_img_processed );
        }
        else if ( strncmp ( argv[i], n_best_opt, strlen ( n_best_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( n_best_opt ), "%d", &algParams.n_best );
        }
        else if ( strncmp ( argv[i], gtDepth_divFactor_opt, strlen ( gtDepth_divFactor_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( gtDepth_divFactor_opt ), "%f", &gt_parameters.divFactor );
        }
        else if ( strncmp ( argv[i], gtDepth_tolerance_opt, strlen ( gtDepth_tolerance_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( gtDepth_tolerance_opt ), "%f", &gt_parameters.dispTolGT );
        }
        else if ( strncmp ( argv[i], gtDepth_tolerance2_opt, strlen ( gtDepth_tolerance2_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( gtDepth_tolerance2_opt ), "%f", &gt_parameters.dispTolGT2 );
        }
        else if ( strncmp ( argv[i], depth_min_opt, strlen ( depth_min_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( depth_min_opt ), "%f", &algParams.depthMin );
        }
        else if ( strncmp ( argv[i], depth_max_opt, strlen ( depth_max_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( depth_max_opt ), "%f", &algParams.depthMax );
        }
        else if ( strncmp ( argv[i], min_angle_opt, strlen ( min_angle_opt ) ) == 0 )
            sscanf ( argv[i] + strlen ( min_angle_opt ), "%f", &algParams.min_angle );
        else if ( strncmp ( argv[i], max_angle_opt, strlen ( max_angle_opt ) ) == 0 ) {
            sscanf ( argv[i] + strlen ( max_angle_opt ), "%f", &algParams.max_angle );
        }
        else if ( strncmp ( argv[i], pmvs_folder_opt, strlen ( pmvs_folder_opt ) ) == 0 ) {
            inputFiles.pmvs_folder = argv[++i];
        }
        else if ( strncmp ( argv[i], max_views_opt, strlen ( max_views_opt ) ) == 0 )
            sscanf ( argv[i] + strlen ( max_views_opt ), "%u", &algParams.max_views );
        else if ( strncmp ( argv[i], no_texture_sim_opt, strlen ( no_texture_sim_opt ) ) == 0 )
            sscanf ( argv[i] + strlen ( no_texture_sim_opt ), "%f", &algParams.no_texture_sim );
        else if ( strncmp ( argv[i], no_texture_per_opt, strlen ( no_texture_per_opt ) ) == 0 )
            sscanf ( argv[i] + strlen ( no_texture_per_opt ), "%f", &algParams.no_texture_per );
        else if ( strcmp ( argv[i], viewSelection_opt ) == 0 )
            algParams.viewSelection = true;
        else if ( strcmp ( argv[i], colorProc_opt ) == 0 )
            algParams.color_processing = true;
        else if ( strcmp ( argv[i], "-o" ) == 0 )
            outputFiles.disparity_filename = argv[++i];
        else if ( strcmp ( argv[i], outputPath_opt ) == 0 )
            outputFiles.parentFolder = argv[++i];
        else if ( strcmp ( argv[i], calib_opt ) == 0 )
            inputFiles.calib_filename = argv[++i];
        else if ( strcmp ( argv[i], gt_opt ) == 0 )
            inputFiles.gt_filename = argv[++i];
        else if ( strcmp ( argv[i], gt_nocc_opt ) == 0 )
            inputFiles.gt_nocc_filename = argv[++i];
        else if ( strcmp ( argv[i], occl_mask_opt ) == 0 )
            inputFiles.occ_filename = argv[++i];
        else if ( strcmp ( argv[i], gt_normal_opt ) == 0 )
            inputFiles.gt_normal_filename = argv[++i];
        else if ( strcmp ( argv[i], images_input_folder_opt ) == 0 )
            inputFiles.images_folder = argv[++i];
        else if ( strcmp ( argv[i], p_input_folder_opt ) == 0 )
            inputFiles.p_folder = argv[++i];
        else if ( strcmp ( argv[i], krt_file_opt ) == 0 )
            inputFiles.krt_file = argv[++i];
        else if ( strcmp ( argv[i], camera_input_folder_opt ) == 0 )
            inputFiles.camera_folder = argv[++i];
        else if ( strcmp ( argv[i], initial_seed_opt ) == 0 )
            inputFiles.seed_file = argv[++i];
        else if ( strcmp ( argv[i], bounding_folder_opt ) == 0 )
            inputFiles.bounding_folder = argv[++i];
        else if ( strncmp ( argv[i], camera_idx_opt, strlen( camera_idx_opt) ) == 0 ){
            sscanf ( argv[i] + strlen ( camera_idx_opt ), "%d", &camera_idx);
        }
        else
        {
            printf ( "Command-line parameter warning: unknown option %s\n", argv[i] );
            //return -1;
        }
    }
    //cout << "Seed file is " << inputFiles.seed_file  << endl;
    //cout << "Min angle is " << algParams.min_angle  << endl;
    if (inputFiles.pmvs_folder.size()>0)
    {
        cout << "Using pmvs information inside directory " << inputFiles.pmvs_folder  << endl;
        inputFiles.images_folder = inputFiles.pmvs_folder + "/visualize/";

        inputFiles.img_filenames.clear();
        get_directory_entries(inputFiles.images_folder.c_str(), inputFiles.img_filenames);

        inputFiles.p_folder = inputFiles.pmvs_folder + "/txt/";

        cout << "Using image " << inputFiles.img_filenames[camera_idx] << " as reference camera" << endl;
        std::swap( inputFiles.img_filenames[0], inputFiles.img_filenames[camera_idx]);
    }
    cout << "Input files are: ";
    for (const auto i: inputFiles.img_filenames)
        cout <<  i  << " ";
    cout <<  endl;

    return 0;
}

static void selectViews (CameraParameters &cameraParams, int imgWidth, int imgHeight, AlgorithmParameters &algParams ) {
    vector<Camera> &cameras = cameraParams.cameras;
    Camera ref = cameras[cameraParams.idRef];

    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cameraParams.viewSelectionSubset.clear ();

    Vec3f viewVectorRef = getViewVector ( ref, x, y);

    // TODO hardcoded value makes it a parameter
    float minimum_angle_degree = algParams.min_angle;
    float maximum_angle_degree = algParams.max_angle;

    unsigned int maximum_view = algParams.max_views;
    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    float min_depth = 9999;
    float max_depth = 0;
    if ( algParams.viewSelection )
        printf("Accepting intersection angle of central rays from %f to %f degrees, use --min_angle=<angle> and --max_angle=<angle> to modify them\n", minimum_angle_degree, maximum_angle_degree);
    for ( size_t i = 1; i < cameras.size (); i++ ) {
        //if ( !algParams.viewSelection ) { //select all views, dont perform selection
            //cameraParams.viewSelectionSubset.push_back ( i );
            //continue;
        //}

        Vec3f vec = getViewVector ( cameras[i], x, y);

        float baseline = norm (cameras[0].C, cameras[i].C);
        float angle = getAngle ( viewVectorRef, vec );
        if ( angle > minimum_angle_radians &&
             angle < maximum_angle_radians ) //0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
        {
            if ( algParams.viewSelection ) {
                cameraParams.viewSelectionSubset.push_back ( i );
                //printf("\taccepting camera %ld with angle\t %f degree (%f radians) and baseline %f\n", i, angle*180.0f/M_PI, angle, baseline);
            }
            float min_range = (baseline/2.0f) / sin(maximum_angle_radians/2.0f);
            float max_range = (baseline/2.0f) / sin(minimum_angle_radians/2.0f);
            min_depth = std::min(min_range, min_depth);
            max_depth = std::max(max_range, max_depth);
            //printf("Min max ranges are %f %f\n", min_range, max_range);
            //printf("Min max depth are %f %f\n", min_depth, max_depth);
        }
        //else
            //printf("Discarding camera %ld with angle\t %f degree (%f radians) and baseline, %f\n", i, angle*180.0f/M_PI, angle, baseline);
    }

    if (algParams.depthMin == -1)
        algParams.depthMin = min_depth;
    if (algParams.depthMax == -1)
        algParams.depthMax = max_depth;

    if (!algParams.viewSelection) {
        cameraParams.viewSelectionSubset.clear();
        for ( size_t i = 1; i < cameras.size (); i++ )
            cameraParams.viewSelectionSubset.push_back ( i );
        return;
    }
    if (cameraParams.viewSelectionSubset.size() >= maximum_view) {
        printf("Too many camera, randomly selecting only %d of them (modify with --max_views=<number>)\n", maximum_view);
        std::srand ( unsigned ( std::time(0) ) );
        std::random_shuffle( cameraParams.viewSelectionSubset.begin(), cameraParams.viewSelectionSubset.end() ); // shuffle elements of v
        cameraParams.viewSelectionSubset.erase (cameraParams.viewSelectionSubset.begin()+maximum_view,cameraParams.viewSelectionSubset.end());
    }
    //for (auto i : cameraParams.viewSelectionSubset )
        //printf("\taccepting camera %d\n", i);
}

static void delTexture (int num, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (int i=0; i<num; i++) {
        cudaFreeArray(cuArray[i]);
        cudaDestroyTextureObject(texs[i]);
    }
}

static void addImageToTextureUint (vector<Mat_<uint8_t> > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with uint8_t point type
        cudaChannelFormatDesc channelDesc =
        //cudaCreateChannelDesc (8,
                               //0,
                               //0,
                               //0,
                               //cudaChannelFormatKindUnsigned);
        cudaCreateChannelDesc<char>();
        // Allocate array with correct size and number of channels
        checkCudaErrors(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<uint8_t>(),
                                              imgs[i].step[0],
                                              cols*sizeof(uint8_t),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}
static void addImageToTextureFloatColor (vector<Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        //cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float)*4,
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
    }
    return;
}

static void addImageToTextureFloatGray (vector<Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc (32,
                               0,
                               0,
                               0,
                               cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        checkCudaErrors(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}

static void selectCudaDevice ()
{
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "There is no cuda capable device!\n");
        exit(EXIT_FAILURE);
    } 
    cout << "Detected " << deviceCount << " devices!" << endl;
    std::vector<int> usableDevices;
    std::vector<std::string> usableDeviceNames;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 3 && prop.minor >= 0) {
                usableDevices.push_back(i);
                usableDeviceNames.push_back(string(prop.name));
            } else {
                cout << "CUDA capable device " << string(prop.name)
                     << " is only compute cabability " << prop.major << '.'
                     << prop.minor << endl;
            }
        } else {
            cout << "Could not check device properties for one of the cuda "
                    "devices!" << endl;
        }
    }
    if(usableDevices.empty()) {
        fprintf(stderr, "There is no cuda device supporting gipuma!\n");
        exit(EXIT_FAILURE);
    }
    cout << "Detected gipuma compatible device: " << usableDeviceNames[0] << endl;;
    checkCudaErrors(cudaSetDevice(usableDevices[0]));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*128);
}

static int runGipuma ( InputFiles &inputFiles,
                                 OutputFiles &outputFiles,
                                 AlgorithmParameters &algParams,
                                 GTcheckParameters &gtParameters,
                                 Results &results
                                 )
{

    // create folder to store result images
    time_t timeObj;
    time ( &timeObj );
    tm *pTime = localtime ( &timeObj );

#if defined(_WIN32)
    _mkdir ( outputFiles.parentFolder );
#else
    mkdir ( outputFiles.parentFolder, 0777 );
#endif
    char outputFolder[256];
    if(inputFiles.img_filenames.empty())
    {
      throw std::runtime_error("There was a problem finding the input files!");
    }
    string ref_name = inputFiles.img_filenames[0].substr ( 0, inputFiles.img_filenames[0].length() - 4 );
    sprintf ( outputFolder, "%s/%04d%02d%02d_%02d%02d%02d_%s", outputFiles.parentFolder, pTime->tm_year + 1900, pTime->tm_mon + 1, pTime->tm_mday, pTime->tm_hour, pTime->tm_min, pTime->tm_sec, ref_name.c_str () );
#if defined(_WIN32)
    _mkdir ( outputFolder );
#else
    mkdir ( outputFolder, 0777 );
#endif

    // store results to file
    char resultsFile[256];
    sprintf ( resultsFile, "%s/results.txt", outputFolder );

    // load images
    if ( inputFiles.img_filenames.size () < 2 )
    {
        printf ( "Command-line parameter error: at least 2 images must be specified\n" );
        return -1;
    }

    size_t numImages = inputFiles.img_filenames.size ();
    algParams.num_img_processed = min ( ( int ) numImages, algParams.num_img_processed );

    vector<Mat_<Vec3b> > img_color(numImages); // imgLeft_color, imgRight_color;
    vector<Mat_<uint8_t> > img_grayscale(numImages);
    for ( size_t i = 0; i < numImages; i++ ) {
        img_grayscale[i] = imread ( ( inputFiles.images_folder + inputFiles.img_filenames[i] ), IMREAD_GRAYSCALE );
        if ( algParams.color_processing ) {
            img_color[i] = imread ( ( inputFiles.images_folder + inputFiles.img_filenames[i] ), IMREAD_COLOR );
        }

        if ( img_grayscale[i].rows == 0 ) {
            printf ( "Image seems to be invalid\n" );
            return -1;
        }
    }

    uint32_t rows = img_grayscale[0].rows;
    uint32_t cols = img_grayscale[0].cols;
    uint32_t numPixels = rows * cols;

    Mat_<float> groundTruthDisp;
    Mat_<float> groundTruthDispNocc;
    Mat_<Vec3f> groundTruthNormals;
    if ( !inputFiles.gt_filename.empty () ) {
        gtParameters.gtCheck = true;
        printf ( "Opening GT image %s\n", inputFiles.gt_filename.c_str () );
        string ext = inputFiles.gt_filename.substr ( inputFiles.gt_filename.find_last_of ( "." ) + 1 );
        if ( ext.compare ( "pfm" ) == 0 ) {
            long nx, ny;
            readPfm ( inputFiles.gt_filename.c_str (), groundTruthDisp, &nx, &ny );
        } else if ( ext.compare ( "dmb" ) == 0 ) {
            readDmb ( inputFiles.gt_filename.c_str (), groundTruthDisp );
        } else {
            Mat gtImg = imread ( inputFiles.gt_filename, -1 );
            gtImg.convertTo ( groundTruthDisp, CV_32F );
        }
        cout << "gt: " << groundTruthDisp.rows << " " << groundTruthDisp.cols << " " << groundTruthDisp.channels () << " " << groundTruthDisp.depth () << endl;
        double minVal, maxVal;
        minMaxLoc ( groundTruthDisp, &minVal, &maxVal );
        //cout << "depth min max: " << minVal << " " << maxVal << endl;
    }
    if ( !inputFiles.gt_nocc_filename.empty () ) {
        if ( !gtParameters.gtCheck ) {
            printf ( "Command-line parameter error: Ground truth image (-gt) must be specified for use of nocc GT\n" );
            return -1;
        }
        gtParameters.noccCheck = true;
        printf ( "Opening nocc GT image %s\n", inputFiles.gt_nocc_filename.c_str () );
        Mat gtImg = imread ( inputFiles.gt_nocc_filename, -1 );
        gtImg.convertTo ( groundTruthDispNocc, CV_32F );
    } else if ( !inputFiles.occ_filename.empty () ) {
        if ( !gtParameters.gtCheck ) {
            printf ( "Command-line parameter error: Ground truth image (-gt) must be specified for use of occlusion mask\n" );
            return -1;
        }
        //Mat occlusionImg; //(imgLeft.rows, imgLeft.cols, CV_16UC1,Scalar(255));
        printf ( "Opening Occlusion image %s\n", inputFiles.occ_filename.c_str () );
        Mat occlusionImg = imread ( inputFiles.occ_filename, IMREAD_GRAYSCALE );
        getNoccGTimg ( groundTruthDisp, occlusionImg, groundTruthDispNocc );
    } else {
        groundTruthDispNocc = groundTruthDisp;
    }
    if ( !inputFiles.gt_normal_filename.empty () ) {
        cout << inputFiles.gt_normal_filename << endl;
        Mat gtNormImg = imread ( inputFiles.gt_normal_filename, -1 );
        cvtColor ( gtNormImg, gtNormImg, COLOR_BGR2RGB );
        //gtNormImg.convertTo ( groundTruthNormals, CV_32FC3 );
        groundTruthNormals = Mat::zeros(gtNormImg.rows,gtNormImg.cols, CV_32FC3);
        for(int y = 0; y < gtNormImg.rows; y++) {
            for(int x = 0; x < gtNormImg.cols; x++) {
                Vec3i gtN_int = (Vec3i) gtNormImg.at<Vec<uint16_t, 3> >(y,x);
                gtN_int = gtN_int - Vec3i(32767,32767,32767);

                if(gtN_int(0) == gtN_int(1) && gtN_int(0) == gtN_int(2) && gtN_int(0) == 0){
                    continue;
                    //gtN_int = Vec3i(0,0,0);
                }
                groundTruthNormals(y,x) = normalize((Vec3f)gtN_int); // get rid of scaling
            }
        }
    }

    // Read initial seeds from disk if available
    if (!inputFiles.seed_file.empty())
    {
        // TODO
    }

    size_t avail;
    size_t used;
    size_t total;

    GlobalState *gs = new GlobalState;
    //cudaMemGetInfo( &avail, &total );
    //used = total - avail;
    //printf("Device memory used after GlobalState allocation: %fMB\n", used/1000000.0f);
    CameraParameters cameraParams = getCameraParameters ( *(gs->cameras), inputFiles, algParams.cam_scale);

    writeParametersToFile ( resultsFile, inputFiles, algParams, gtParameters, numPixels );

    //allocation for disparity and normal stores
    vector<Mat_<float> > disp ( algParams.num_img_processed );
    vector<Mat_<uchar> > validCost ( algParams.num_img_processed );
    for ( int i = 0; i < algParams.num_img_processed; i++ ) {
        disp[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32F );
        validCost[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_8U );
    }

        Mat testImg_display;
    //visualize normals on a halfsphere (just for comparision with normal result image)
    {
        int sizeN = cols/8;
        float halfSize = (float)sizeN/2.0f;
        Mat_<Vec3f> normalTestImg = Mat::zeros(sizeN, sizeN, CV_32FC3);
        for(int i=0;i<sizeN;i++){
            for(int j=0;j<sizeN;j++){
                float y = (float)i/halfSize-1.0f;
                float x = (float)j/halfSize-1.0f;
                float xy = pow(x,2)+pow(y,2);
                if(xy <= 1.0f){
                    float z = sqrt(1.0f-xy);
                    normalTestImg(sizeN-1-i,sizeN-1-j) = Vec3f(-x,-y,-z);
                }
            }
        }

        normalTestImg.convertTo(testImg_display,CV_16U,32767,32767);
        cvtColor(testImg_display,testImg_display,COLOR_RGB2BGR);
        //char outputPathTest[256];
        //sprintf(outputPathTest, "%s/normalsTestImg.png", outputFolder);
        //imwrite(outputPathTest,testImg_display);
    }


    selectViews ( cameraParams, cols, rows, algParams);

    if (inputFiles.pmvs_folder.size()>0) {
        cout << "Using bundler file " << inputFiles.pmvs_folder + "/bundle.rd.out" << " to obtain depth range" << endl;
        from_bundler_get_range (cameraParams, algParams, (inputFiles.pmvs_folder + "/bundle.rd.out").c_str());
    }


    //cout << "Range of Minimum/Maximum depth is: " << algParams.depthMin << " " << algParams.depthMax << endl;
    int numSelViews = cameraParams.viewSelectionSubset.size ();

    cout << "Total number of images used: " << numSelViews << endl;
    ofstream myfile;
    myfile.open ( resultsFile, ios::out | ios::app );
    myfile << "\nNumber of selected views: " << numSelViews << endl;
    myfile << "Selected views: ";
    cout << "Selected views: ";
    for ( int i = 0; i < numSelViews; i++ ) {
        myfile << cameraParams.viewSelectionSubset[i] << ", ";
        cout << cameraParams.viewSelectionSubset[i] << ", ";
        gs->cameras->viewSelectionSubset[i] = cameraParams.viewSelectionSubset[i];
    }
    cout << endl;
    myfile << "\n\n";
    myfile.close ();


    for ( int i = 0; i < algParams.num_img_processed; i++ ) {
        cameraParams.cameras[i].depthMin = algParams.depthMin;
        cameraParams.cameras[i].depthMax = algParams.depthMax;

        gs->cameras->cameras[i].depthMin = algParams.depthMin;
        gs->cameras->cameras[i].depthMax = algParams.depthMax;

        algParams.min_disparity = disparityDepthConversion ( cameraParams.f, cameraParams.cameras[i].baseline, cameraParams.cameras[i].depthMax );
        algParams.max_disparity = disparityDepthConversion ( cameraParams.f, cameraParams.cameras[i].baseline, cameraParams.cameras[i].depthMin );


        double minVal, maxVal;
        minMaxLoc ( disp[i], &minVal, &maxVal );
    }
    cout << "Range of Minimum/Maximum depth is: " << algParams.min_disparity << " " << algParams.max_disparity << ", change it with --depth_min=<value> and  --depth_max=<value>" <<endl;

    // run gpu run
    // Init parameters
    gs->params = &algParams;

    gs->cameras->viewSelectionSubsetNumber = numSelViews;

    // Init ImageInfo
    gs->cameras->cols = cols;
    gs->cameras->rows = rows;
    gs->params->cols = cols;
    gs->params->rows = rows;

    // Resize lines
    {
        gs->lines->n = rows * cols;
        gs->lines->resize(rows * cols);
        //gs->lines.s = img_grayscale[0].step[0];
        gs->lines->s = cols;
        gs->lines->l = cols;
    }

    vector<Mat > img_grayscale_float(numImages);
    vector<Mat > img_color_float(numImages);
    vector<Mat > img_color_float_alpha(numImages);
    vector<Mat_<uint16_t> > img_grayscale_uint(numImages);
    for (size_t i = 0; i<numImages; i++)
    {
        img_grayscale[i].convertTo(img_grayscale_float[i], CV_32FC1); // or CV_32F works (too)
        img_grayscale[i].convertTo(img_grayscale_uint[i], CV_16UC1); // or CV_32F works (too)
        if(algParams.color_processing) {
            vector<Mat_<float> > rgbChannels ( 3 );
            img_color_float_alpha[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC4 );
            img_color[i].convertTo (img_color_float[i], CV_32FC3); // or CV_32F works (too)
            Mat alpha( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC1 );
            //Mat out[] = { img_color_float[i], alpha };

            //// add alpha channel
            //int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
            //mixChannels( &img_color_float_alpha[i], 1, out, 2, from_to, 4 );
            split (img_color_float[i], rgbChannels);
            rgbChannels.push_back( alpha);
            merge (rgbChannels, img_color_float_alpha[i]);
        }
    }
    int64_t t = getTickCount ();

    cudaMemGetInfo( &avail, &total );
    used = total - avail;
    //printf("Device memory used: %fMB\n", used/1000000.0f);
    // Copy images to texture memory
    //addImageToTextureUint (img_grayscale, gs->imgs);
    if (algParams.color_processing)
        addImageToTextureFloatColor (img_color_float_alpha, gs->imgs, gs->cuArray);
    else
        addImageToTextureFloatGray (img_grayscale_float, gs->imgs, gs->cuArray);

    cudaMemGetInfo( &avail, &total );
    used = total - avail;
    //printf("Device memory used: %fMB\n", used/1000000.0f);
    runcuda(*gs);
    Mat_<Vec3f> norm0 = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC3 );
    Mat_<float> cudadisp = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC1 );
    for( int i = 0; i < img_grayscale[0].cols; i++ )
        for( int j = 0; j < img_grayscale[0].rows; j++ )
        {
            int center = i+img_grayscale[0].cols*j;
            float4 n = gs->lines->norm4[center];
            norm0 (j, i) = Vec3f ( n.x,
                                   n.y,
                                   n.z);
            cudadisp (j, i) = gs->lines->norm4[i+img_grayscale[0].cols*j].w;
        }

    Mat_<Vec3f> norm0disp = norm0.clone ();
    Mat planes_display, planescalib_display, planescalib_display2;
    getNormalsForDisplay ( norm0disp, planes_display );
    testImg_display.copyTo(planes_display(Rect(cols-testImg_display.cols, 0, testImg_display.cols, testImg_display.rows)));
    writeImageToFile ( "./", "normals", planes_display );
    writeImageToFile ( outputFolder, "normals", planes_display );
    planes_display.release ();

    Mat cost_display;
    normalize ( cudadisp, cost_display, 0, 65535, NORM_MINMAX, CV_16U );
    writeImageToFile ( "./", "cudacost", cost_display );

    Mat_<float> disp0 = cudadisp.clone ();

    char outputPath[256];
    sprintf ( outputPath, "%s/disp.dmb", outputFolder );
    writeDmb ( outputPath, disp0 );
    //sprintf ( outputPath, "%s/d.dmb", outputFolder );
    //writeDmb ( outputPath, planes[0].d );
    //vector<Mat_<float> > channelsNormal ( 3 );
    //split ( norm0, channelsNormal );
    //sprintf ( outputPath, "%s/normalX.dmb", outputFolder );
    //writeDmb ( outputPath, channelsNormal[0] );
    //sprintf ( outputPath, "%s/normalY.dmb", outputFolder );
    //writeDmb ( outputPath, channelsNormal[1] );
    //sprintf ( outputPath, "%s/normalZ.dmb", outputFolder );
    //writeDmb ( outputPath, channelsNormal[2] );
    sprintf ( outputPath, "%s/normals.dmb", outputFolder );
    writeDmbNormal ( outputPath, norm0 );

    Mat_<float> distImg;

    for ( size_t i = 0; i < (size_t) algParams.num_img_processed; i++ ) {
        // store 3D coordinates to file
        CameraParameters camParamsNotTransformed = getCameraParameters ( *(gs->cameras), inputFiles, algParams.cam_scale, false );
        char plyFile[256];
        sprintf ( plyFile, "%s/3d_model%lu.ply", outputFolder, i );

        storePlyFileBinary ( plyFile, disp0, norm0, img_grayscale[i], camParamsNotTransformed.cameras[i], distImg );
        Mat dist_display, dist_display_col;
        getDisparityForDisplay ( distImg, dist_display, dist_display_col, algParams.max_disparity );
        writeImageToFile ( outputFolder, "dist", dist_display );
        writeImageToFile ( "./", "cudadisp", dist_display );
        writeImageToFile ( outputFolder, "dist_col", dist_display_col );
    }

    Mat_<float> disp_occFilling, disp_nocc;
    disp_occFilling = cudadisp;

    Mat_<Vec3f> planes_occFilling, planes_nocc;
    planes_occFilling = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC3 );

    uint32_t numValidPixels = 0;
    if ( algParams.num_img_processed >= 2 ) {
        Mat_<float> disp0Temp = disp[0];
        Mat_<float> disp1Temp = disp[1];
        if ( !cameraParams.rectified ) {
            convertDisparityDepthImage ( disp[0], disp0Temp, cameraParams.f, cameraParams.cameras[1].baseline );
            convertDisparityDepthImage ( disp[1], disp1Temp, cameraParams.f, cameraParams.cameras[1].baseline );
            cameraParams.rectified = true;
        }
        disp_nocc = disp_occFilling.clone ();
        planes_nocc = planes_occFilling.clone ();
    }
    else {
        disp_nocc = disp[0].clone ();
    }
    Mat_<float> disp_median = cudadisp;

    if ( algParams.num_img_processed >= 2 ) {
        Mat disp_display, disp_col, disp_nocc_display, disp_nocc_col;
        getDisparityForDisplay ( disp_occFilling, disp_display, disp_col, algParams.max_disparity );
        writeImageToFile ( outputFolder, "dispOccFilling", disp_col );
        disp_display.release ();
        disp_col.release ();

        getDisparityForDisplay ( disp_nocc, disp_nocc_display, disp_nocc_col, algParams.max_disparity );
        writeImageToFile ( outputFolder, "dispNocc", disp_nocc_display );
        writeImageToFile ( outputFolder, "dispNocc_color", disp_nocc_col );
        disp_nocc_display.release ();
            disp_nocc_col.release ();
    }

    //cout << "before time output" << endl;
    t = getTickCount () - t;
    double rt = ( double ) t / getTickFrequency ();

    results.total_runtime = rt;
    results.runtime_per_pixel = rt / ( float ) ( numPixels * 1000 );
    results.valid_pixels = ( float ) numValidPixels / ( float ) numPixels;

    myfile.open ( resultsFile, ios::out | ios::app );
    myfile << "\n\nTotal runtime: " << rt << " sec ( " << rt / 60.0f << " min)" << endl;
    myfile << "\nRuntime per pixel: " << results.runtime_per_pixel << " ms" << endl;
    myfile << "\nValid pixels: " << results.valid_pixels << endl;
    myfile.close ();

    cout << "Total runtime including disk i/o: " << rt << "sec" << endl;

    //ground truth comparison
    if ( gtParameters.gtCheck ) {

        //postprocessing
        Mat_<uchar> valid = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_8U );
        Mat_<uchar> errorImg = Mat::zeros ( rows, cols, CV_8U );
        Mat_<uchar> errorImgNocc = Mat::zeros ( rows, cols, CV_8U );
        Mat_<uchar> errorImgNocc2 = Mat::zeros ( rows, cols, CV_8U );
        Mat_<uchar> errorImgValid ( rows, cols, 255 );
        Mat_<uchar> nonerrorImg = Mat::zeros ( groundTruthDisp.rows, groundTruthDisp.cols, CV_8U );

        float error = 0, errorNocc = 0, errorValid = 0, errorValidAll = 0, error2 = 0;

        //error after post processing
        float med_error, med_error2;
        computeError ( groundTruthDisp, groundTruthDispNocc, disp_median, errorImg, errorImgNocc, errorImgNocc2, nonerrorImg, med_error, med_error2, errorNocc, errorValid, errorValidAll, gtParameters, valid, errorImgValid );

        //error before post processing
        results.valid_pixels_gt = computeError ( groundTruthDisp, groundTruthDispNocc, disp0, errorImg, errorImgNocc, errorImgNocc2, nonerrorImg, error, error2, errorNocc, errorValid, errorValidAll, gtParameters, valid, errorImgValid );

        float normalError = 0.0f;
        float normalError2 = 0.0f;
        if ( !inputFiles.gt_normal_filename.empty () ) {
            Mat_<float> normalErrorMap = Mat::zeros ( rows, cols, CV_32F );
            normalError = computeNormalError ( norm0, groundTruthNormals, 0.2f, 0.3f, normalErrorMap, normalError2 );
            Mat normalErrorMapDisplay;
            normalize ( normalErrorMap, normalErrorMapDisplay, 0, 65535, NORM_MINMAX, CV_16U );
            writeImageToFile ( outputFolder, "errorNormal", normalErrorMapDisplay );

        }

        results.error_occ = error;
        results.error_noc = errorNocc;
        results.error_valid = errorValid;
        results.error_valid_all = errorValidAll;

        //char outputPath[256];
        sprintf ( outputPath, "%s/finalErrorCont.png", outputFolder );
        imwrite ( outputPath, errorImg );

        char outputName[256];
        sprintf ( outputName, "Error (all): %f, Error (nocc) %f", error, errorNocc );

        sprintf ( outputPath, "%s/finalErrorBin_10.png", outputFolder );
        imwrite ( outputPath, errorImgNocc );

        sprintf ( outputPath, "%s/finalErrorBin_2.png", outputFolder );
        imwrite ( outputPath, errorImgNocc2 );

        sprintf ( outputPath, "%s/finalErrorValidOcclusionCheck.png", outputFolder );
        imwrite ( outputPath, errorImgValid );

        sprintf ( outputPath, "%s/finalNonError.png", outputFolder );
        imwrite ( outputPath, nonerrorImg );

        myfile.open ( resultsFile, ios::out | ios::app );
        myfile << "Valid pixels of GT subset: " << results.valid_pixels_gt << endl << endl;
        myfile << "Error1: " << error << endl;
        myfile << "Correct1: " << 1 - error << endl;
        myfile << "Error2: " << error2 << endl;
        myfile << "Correct2: " << 1 - error2 << endl;
        myfile << "\n\nError (nocc): " << errorNocc << endl;
        myfile << "Error (valid occlusion check): " << errorValid << endl;
        myfile << "Error (valid occlusion check, div by #GT points): " << errorValidAll << endl;
        myfile << "\nError after median filtering: " << med_error << ", thresh 2: " << med_error2 << endl;
        myfile << "Correct %: " << 1 - med_error << " (thresh 2: " << 1 - med_error2 << ")" << endl;
        myfile << "\nNormal error (0.2rad): " << normalError << endl;
        myfile << "\nNormal error2 (0.3rad): " << normalError2 << endl;
        myfile.close ();

        cout << "Error1: " << error << endl;
        cout << "Correct1: " << 1 - error << endl;
        cout << "Error2: " << error2 << endl;
        cout << "Correct2: " << 1 - error2 << endl;
        cout << "Error (nocc): " << errorNocc << endl;
        cout << "Error (valid occlusion check): " << errorValid << endl;
        cout << "Error (valid occlusion check, div by #GT points): " << errorValidAll << endl;
    }

    if ( algParams.num_img_processed >= 2 ) {
        Mat planes_display, planescalib_display, planescalib_display2;

        writeImageToFile ( outputFolder, "normals", planes_display );
        planes_display.release ();
    }

    if ( gtParameters.gtCheck ) {
        Mat gt_col;
        double minVal, maxVal;
        minMaxLoc ( groundTruthDisp, &minVal, &maxVal );
        cout << "minmax" << minVal << " " << maxVal << endl;
        Mat gt8;
        groundTruthDisp.convertTo ( gt8, CV_8U, 255.f / ( float ) ( algParams.max_disparity * gtParameters.divFactor ) );
        minMaxLoc ( gt8, &minVal, &maxVal );
        cout << "minmax" << minVal << " " << maxVal << endl;
        applyColorMap ( gt8, gt_col, COLORMAP_JET );
        //writeImageToFile(outputFolder,"groundTruth_color",gt_col);
    }

    //cudaMemGetInfo( &avail, &total );
    //used = total - avail;
    //printf("Device memory used: %fMB\n", used/1000000.0f);
    // Free memory
    delTexture (algParams.num_img_processed, gs->imgs, gs->cuArray);
    delete gs;
    delete &algParams;
    cudaDeviceSynchronize();

    //cudaMemGetInfo( &avail, &total );
    //used = total - avail;
    //printf("Device memory used: %fMB\n", used/1000000.0f);

    return 0;
}

int main(int argc, char **argv)
{
    if ( argc < 3 )
    {
        print_help (argv);
        return 0;
    }

    InputFiles inputFiles;
    OutputFiles outputFiles;
    AlgorithmParameters* algParams = new AlgorithmParameters;
    GTcheckParameters gtParameters;

    int ret = getParametersFromCommandLine ( argc, argv, inputFiles, outputFiles, *algParams, gtParameters);
    if ( ret != 0 )
        return ret;

    selectCudaDevice();

    Results results;
    ret = runGipuma ( inputFiles, outputFiles, *algParams, gtParameters, results);

    return 0;
}

