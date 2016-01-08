/*
 * utility functions for reading and writing files
 */

#pragma once

#include <iostream>
#include <fstream>

static void getProjectionMatrix(char* line, Mat_<float> &P){
    const char* p;
    int idx = 0;
    for (p = strtok( line, " " );  p;  p = strtok( NULL, " " ))
    {
        if(p[0]=='P' || p[0]=='p')
            continue;

        //float val = stof(p);
        float val = (float)atof(p);
        /*cout << val << endl;*/
        P(idx/4,idx%4)=val;

        idx++;
    }
}

static int read3Dpoint(char* line, Vec3f &pt){
    const char* p;
    int idx = 0;
    for (p = strtok( line, " " );  p;  p = strtok( NULL, " " ))
    {
        if(idx > 2)
            return -1;
        float val = (float)atof(p);
        pt[idx] = val;

        idx++;
    }
    if(idx<2)
        return -1;
    return 0;
}

static void readCalibFileKitti(const string calib_filename, Mat_<float> &P1, Mat_<float> &P2){
    ifstream myfile;
    myfile.open(calib_filename.c_str(),ifstream::in);
    //get first line (containing P0)
    char line[512];
    myfile.getline(line,512);
    getProjectionMatrix(line,P1);
    myfile.getline(line,512);
    getProjectionMatrix(line,P2);
    myfile.close();
}

static void readBoundingVolume(const string filename, Vec3f &ptBL, Vec3f & ptTR){
    ifstream myfile;
    myfile.open(filename.c_str(),ifstream::in);
    char line[512];
    //bottom left point
    myfile.getline(line,512);
    read3Dpoint(line,ptBL);
    //top right point
    myfile.getline(line,512);
    read3Dpoint(line,ptTR);

    myfile.close();
}


static void readCameraFileStrecha(const string camera_filename, float &focalLength){
    // only interested in focal length, but possible to get also other internal and external camera parameters
    // focal length is stored in pixel format as alpha_x and alphy_y, only using alpha_x (which is the very first parameter of the internal camera matrix)
    ifstream myfile;
    myfile.open(camera_filename.c_str(),ifstream::in);
    char line[512];
    myfile.getline(line,512);
    const char* p = strtok( line, " " );
    focalLength = (float)atof(p);
    myfile.close();
}

static void readPFileStrechaPmvs(const string p_filename, Mat_<float> &P){
    ifstream myfile;
    myfile.open(p_filename.c_str(),ifstream::in);

    //cout <<"Opening file " << p_filename << endl;
    for( int i = 0; i < 4; i++){
        if (myfile.eof())
            break;
        char line[512];
        myfile.getline(line,512);
        if (strstr(line,"CONTOUR")!= NULL) {
            //printf("Skipping CONTOUR\n");
            i--;
            continue;
        }

        //cout << "Line is "<< line << endl;
        const char* p;
        int j = 0;
        for (p = strtok( line, " " );  p;  p = strtok( NULL, " " ))
        {
            float val = (float)atof(p);
            P(i,j)=val;
            j++;
        }
    }
    myfile.close();
}
static void readKRtFileMiddlebury(const string filename, vector<Camera> cameras, InputFiles inputFiles)
{
    ifstream myfile;
    myfile.open( filename, ifstream::in );
    string line;

    getline (myfile, line); // throw away first line

    int i=0;
    int truei=-1;
    while( getline( myfile,line) )
    {
        /*cout << "Line is "<< line << endl;*/
        Mat Rt;
        Mat_<float> K = Mat::zeros( 3, 3, CV_32F );
        Mat_<float> R = Mat::zeros( 3, 3, CV_32F );
        Vec3f vt;
        stringstream ss(line);
        string tmp;
        ss >> tmp
        >> K(0,0) >> K(0,1) >> K(0,2) >> K(1,0) >> K(1,1) >> K(1,2) >> K(2,0) >> K(2,1) >> K(2,2) //
        >> R(0,0) >> R(0,1) >> R(0,2) >> R(1,0) >> R(1,1) >> R(1,2) >> R(2,0) >> R(2,1) >> R(2,2) //
        >> vt(0) >> vt(1) >> vt(2);
        /*cout << "K is " << K << endl;*/
        /*cout << "R is " << R << endl;*/
        /*cout << "t is " << vt << endl;*/
        //cout << "Filename is " << tmp << endl;
        //cout << "image Filename is " << inputFiles.img_filenames[i] << endl;
        for( size_t j = 0; j < inputFiles.img_filenames.size(); j++) {
            if( tmp == inputFiles.img_filenames[j]) {
                truei=j;
                break;
            }
        }
        Mat t(vt, false);
        /*Mat t(vt);*/
        hconcat(R, t, Rt);
        cameras[truei].P = K*Rt;
        /*cout << "Rt is " << Rt<< endl;*/
        /*cout << "P is " << P << endl;*/
        /*cout << "P is " << cameras[i].P << endl;*/
        i++;
    }


    /*while (os >> temp)                //the stringstream makes temp a token*/
        /*std::cout <<temp <<std::endl;   //and deletes that token from itself*/
    //the token can now be
    //outputted to console, or put into an array,
    //or whatever you choose to do ith it .
    return;
}

static void readCalibFileDaisy(const string calib_filename, Mat_<float> &P){
    ifstream myfile;
    myfile.open(calib_filename.c_str(),ifstream::in);

    char line[512];
    while (myfile.getline(line, 512)) {
        if(line[0] == 'p')
            getProjectionMatrix(line,P);
    }

    myfile.close();
}

static void writeImageToFile(const char* outputFolder,const char* name,const Mat &img){
    char outputPath[256];
    sprintf(outputPath, "%s/%s.png", outputFolder,name);
    imwrite(outputPath,img);
}

static void writeParametersToFile(char* resultsFile, InputFiles inputFiles, AlgorithmParameters &algParameters, GTcheckParameters &gtParameters, uint32_t numPixels){

    ofstream myfile;
    myfile.open (resultsFile, ios::out);
    myfile << "Number of images: " << inputFiles.img_filenames.size() << endl;
    myfile << "Image folder: " << inputFiles.images_folder << endl;
    myfile << "Images: ";
    for(size_t i=0; i < inputFiles.img_filenames.size(); i++)
        myfile << inputFiles.img_filenames[i] << ", " ;
    myfile << endl;
    if(numPixels != 0)
        myfile << "Num. pixels: " << numPixels << endl;
    myfile << "\nParameters:" << endl;
    myfile << "  Cost function: " ;
    if(algParameters.algorithm == PM_COST)
        myfile << "PatchMatch Cost" << endl;
    else if(algParameters.algorithm == CENSUS_TRANSFORM)
        myfile << "Census Transform" << endl;
    else if(algParameters.algorithm == SPARSE_CENSUS)
        myfile << "Sparse Census Transform" << endl;
    else if(algParameters.algorithm == CENSUS_SELFSIMILARITY)
        myfile << "Census Transform with Self-Similarity Propagation" << endl;
    else if(algParameters.algorithm == PM_SELFSIMILARITY)
        myfile << "Patch Match Cost with Self-Similarity Propagation" << endl;
    else
        myfile << algParameters.algorithm << endl;
    myfile << "  Kernel size: " << algParameters.box_hsize << " x " << algParameters.box_vsize << endl;
    myfile << "  Number of iterations: " << algParameters.iterations << endl;
    myfile << "  Max. disparity: " << algParameters.max_disparity << endl;
    myfile << "  Depth min: " << algParameters.depthMin << endl;
    myfile << "  Depth max: " << algParameters.depthMax << endl;
    myfile << "  gamma: " << algParameters.gamma << endl;
    myfile << "  alpha: " << algParameters.alpha << endl;
    myfile << "  tauCol: " << algParameters.tau_color << endl;
    myfile << "  tauGrad: " << algParameters.tau_gradient << endl;
    myfile << "  border value: " << algParameters.border_value << endl;
    myfile << "  disparity tolerance (occ check): " << algParameters.dispTol << endl;
    myfile << "  normal tolerance (occ check): " << algParameters.normTol << endl;
    myfile << "  census epsilon: " << algParameters.census_epsilon << endl;
    myfile << "  self-similarity n: " << algParameters.self_similarity_n << endl;
    myfile << "  cost good truncation factor: " << algParameters.good_factor << endl;
    myfile << "  cost combination: ";
    if(algParameters.cost_comb == COMB_ALL)
        myfile << "ALL" << endl;
    else if(algParameters.cost_comb == COMB_BEST_N)
        myfile << "BEST_N (n=" << algParameters.n_best << ")" << endl;
    else if(algParameters.cost_comb == COMB_ANGLE)
        myfile << "ANGLE" << endl;
    else if(algParameters.cost_comb == COMB_GOOD)
        myfile << "GOOD" << endl;
    myfile << "  color processing: ";
    if(algParameters.color_processing)
        myfile << "yes" << endl;
    else
        myfile << "no" << endl;
    myfile << "  view selection: ";
    if(algParameters.viewSelection)
        myfile << "yes" << endl;
    else
        myfile << "no" << endl;
    myfile << "  GT disparity tolerance: " << gtParameters.dispTolGT << "\n" << endl;
    myfile.close();
}
// read ground truth depth map file (dmb) (provided by Tola et al. "DAISY: A Fast Local Descriptor for Dense Matching" http://cvlab.epfl.ch/software/daisy)
static int readDmbNormal (const char *filename, Mat_<Vec3f> &img)
{
    FILE *inimage;
    inimage = fopen(filename, "rb");
    if (!inimage){
        printf("Error opening file %s",filename);
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    //only support float
    if(type != 1){
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    float* data;
    data = (float*) malloc (sizeof(float)*dataSize);
    fread(data,sizeof(float),dataSize,inimage);

    img = Mat(h,w,CV_32FC3,data);

    fclose(inimage);
    return 0;

}
// read ground truth depth map file (dmb) (provided by Tola et al. "DAISY: A Fast Local Descriptor for Dense Matching" http://cvlab.epfl.ch/software/daisy)
static int readDmb(const char *filename, Mat_<float> &img)
{
    FILE *inimage;
    inimage = fopen(filename, "rb");
    if (!inimage){
        printf("Error opening file %s",filename);
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    //only support float
    if(type != 1){
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    float* data;
    data = (float*) malloc (sizeof(float)*dataSize);
    fread(data,sizeof(float),dataSize,inimage);

    img = Mat(h,w,CV_32F,data);

    fclose(inimage);
    return 0;

}
static int writeDmbNormal(const char *filename, Mat_<Vec3f> &img){
    FILE *outimage;
    outimage = fopen(filename, "wb");
    if (!outimage)
        printf("Error opening file %s",filename);

    int32_t type = 1; //float
    int32_t h = img.rows;
    int32_t w = img.cols;
    int32_t nb = 3;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)img.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

static int writeDmb(const char *filename, Mat_<float> &img){
    FILE *outimage;
    outimage = fopen(filename, "wb");
    if (!outimage)
        printf("Error opening file %s",filename);

    int32_t type = 1; //float
    int32_t h = img.rows;
    int32_t w = img.cols;
    int32_t nb = 1;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)img.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

static int readPfm( const char *filename,
                    //double ***u, // double matrix image
                    Mat_<float> &img,
                    long *nx, /*image size in x direction */
                    long *ny) /*image size in y direction */
{
    FILE *inimage;              /* input image FILE pointer */
    long   i, j;                 /* loop variable */
    char   row[4096];              /* for reading data */

    /* open input pgm file and read header */
    inimage = fopen(filename, "rb");
    if (!inimage)
        printf("Error opening file %s",filename);

    /* calling it two times because of the P6 header */
    fgets (row, 4096, inimage);
    if (row[0]!='P')
        abort();
    printf("Row is %s\n", row);
    char format = row[1];
    switch (format)  /* which format to deal with */
    {
    case '5': /* P6 format - classic pgm */
      printf("Opening %s P5 file\n", filename );
      fgets (row, 4096, inimage);

      while (row[0]=='#'||row[0]=='\n') fgets(row, 4096, inimage);
      sscanf (row, "%ld %ld", nx, ny);
      fgets (row, 4096, inimage);
      //    fgets (row, 4096, inimage);

      /* allocate storage */
      //alloc_matrix_d (u, *nx, *ny);
      img = Mat::zeros((int)*ny,(int)*nx,CV_32F);

      /* read image data */
      for (j=0; j<*ny; j++)
          for (i=0; i<*nx; i++)
              img(j,i) = (float) getc (inimage);
              //(*u)[i][j] = (double) getc (inimage);
      break;
      /* PF format - pbm HDR format file */
    case 'F':
    case 'f':
      printf("Opening %s PF file\n", filename );
      fgets (row, 4096, inimage);

      while (row[0]=='#'||row[0]=='\n')
          fgets(row, 4096, inimage);

      sscanf (row, "%ld %ld", nx, ny);

//      fgets (row, 4096, inimage);
      double scale;
      fscanf(inimage, "%lf\n", &scale);
//      printf("Scale is %f\n", scale);
      //    fgets (row, 4096, inimage);

      /* allocate storage */
      //alloc_matrix_d (u, (*nx)*4*3, (*ny)*4*3);
      img = Mat::zeros((int)*ny,(int)*nx,CV_32F);

      float tmpfloat;
//      printf("Float is %lu bytes big\n", sizeof(float));
      /* read image data */
      for (j=*ny-1; j>=0; j--) {
          for (i=0; i<*nx; i++) {
              //            (*u)[i][j] = (double) getc (inimage);
              fread((void *)(&tmpfloat), sizeof(float), 1, inimage);
              /* overwrite other 3 channels when they are available */
              if (format=='F') {
                  fread((void *)(&tmpfloat), sizeof(float), 1, inimage);
                  fread((void *)(&tmpfloat), sizeof(float), 1, inimage);
              }
              img(j,i) = tmpfloat;
              //(*u)[i][j] = (double) tmpfloat;

              /* if positive convert to big endian */
              if (scale > 0)
              {
                  char array[4];
                  char tmpbyte;
                  memcpy(array, &tmpfloat, sizeof (float));
                  /* swap 0 3 */
                  tmpbyte = array[0];
                  array[0] = array[3];
                  array[3] = tmpbyte;
                  /* swap 1 2 */
                  tmpbyte = array[1];
                  array[1]=array[2];
                  array[2] = tmpbyte;
                  memcpy(&tmpfloat, array, sizeof (float));

                  img(j,i) = tmpfloat;
                  //(*u)[i][j] = (double) tmpfloat;
              }
          }
      }
      break;
    }

    fclose(inimage);
    return 0;

}
