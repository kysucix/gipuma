/*
 * utility functions for comparison with ground truth data
 */

#pragma once

/* compute error of disparity image based on ground truth image
 * Input: gtDisp  - ground truth disparity image
 *		  occImg  - occlusion mask (for seperate error computation of non occluded pixels)
 *        disp    - disparity image
 *        gtParam - parameters for error computation (e.g. division factor, thresholds, ...)
 *        valid   - mask that defines if pixel passed occlusion check (1) or not (0)
 * Output: errorImgContinuous - shows continuous error (difference between ground truth and actual disparity)
 *         errorImg           - white: no error, gray: error at occluded area, black: error at nonoccluded area
 *         nonErrorImg        - black: no GT data, gray: error, white: no error
 *         error              - error all (sum over all pixels: 0 or 1 (error), divided by num pixels for which GT data is available)
 *         errorNocc		  - error for nonoccluded areas (using occlusion mask)
 *         errorValid         - error of all pixels that passed occlusion check
 *         errorvalidAll      - error of all pixels that passed the occlusion check with all other pixels being counted as error 1
 *         errorImgValid      - black: errorneous pixels that passed the occlusion check
 */
float computeError(const Mat_<float>& gtDisp, const Mat_<uint8_t>& occImg, Mat_<float>& disp, Mat_<uchar>& errorImgContinuous, Mat_<uchar>& errorImg, Mat_<uchar>& errorImg2, Mat_<uchar>& nonErrorImg, float &error, float &error2, float &errorNocc, float &errorValid, float &errorValidAll, GTcheckParameters gtParam, Mat_<uchar>& valid, Mat_<uchar>& errorImgValid){
	//for Middleburry dataset
	error = 0;
	error2 = 0;
	errorNocc = 0;
	errorValid = 0;
	errorValidAll = 0;
	int numGtPixels = 0; //pixels with disp=0 in GT are not counted
	int numNoccPixels = 0; //according to occImg (GT)
	int numValidPixels = 0; //pixels that passed occlusion check
	bool checkValid = false;
	if(valid.rows>0)
		checkValid =true;

	for(int i = 0; i < gtDisp.rows; i++) {
		for(int j = 0; j < gtDisp.cols; j++) {
			float d1 = (float)gtDisp(i,j)/gtParam.divFactor;
			if(d1==0.0f || d1==-1.0f){ //no valid disparity value for ground truth
				errorImgContinuous(i,j) = 255;
				errorImg(i,j) = 255;
				errorImg2(i,j) = 255;
				continue;
			}
			numGtPixels++;
			bool nocc = false;
			bool validPixel = false;
			if(occImg(i,j) != 0){ //check if pixel is occluded
				nocc = true;
				numNoccPixels++;
			}
			nonErrorImg(i,j) = 100;
			if(checkValid && valid(i,j) != 0){
				validPixel = true;
				numValidPixels++;
			}
			float d2 = disp(i,j);
			float diff = abs(d1 - d2);
			errorImgContinuous(i,j) = 255 - min((int)(diff*gtParam.scale),255);
			uint8_t err = 255;
			if (diff >= gtParam.dispTolGT){
				error++;
				err = 150;
				if(nocc){
					errorNocc++;
					err = 0;
				}
				if(validPixel){
					errorValid++;
					errorImgValid(i,j) = 0;
				}


			}else{
				nonErrorImg(i,j) = 255;
			}
			errorImg(i,j) = err;

			uint8_t err2 = 255;
			if(diff >= gtParam.dispTolGT2){
				err2 = 150;
				if(nocc)
					err2 = 0;
				error2++;
			}
			errorImg2(i,j) = err2;
		}
	}
	error = error / (float)numGtPixels;
	error2 = error2 / (float)numGtPixels;
	errorNocc = errorNocc / (float)numNoccPixels;
	errorValidAll = (errorValid + (numGtPixels-numValidPixels)) / (float)numGtPixels;
	errorValid = errorValid / (float)numValidPixels;
	return (float)numValidPixels/(float)numGtPixels;
}

static float computeNormalError(const Mat_<Vec3f> &normals, const Mat_<Vec3f> &gtNormals, float gtNormTol, float gtNormTol2, Mat_<float> &errorMap, float &error2){
	int numGtPixels = 0;
	int error = 0;
	int error2_count = 0;
	for(int y = 0; y < gtNormals.rows; y++) {
		for(int x = 0; x < gtNormals.cols; x++) {
			/*
			Vec3i gtN_int = gtNormals(y,x);
			gtN_int = gtN_int - Vec3i(32767,32767,32767);


			if(gtN_int(0) == gtN_int(1) && gtN_int(0) == gtN_int(2) && gtN_int(0) == 0){
				continue;
			}
			Vec3f gtN = normalize((Vec3f)gtN_int); // get rid of scaling
			*/
			Vec3f gtN = gtNormals(y,x);
			if(gtN(0) + gtN(1) + gtN(2) < 0.1f)
				continue;

			numGtPixels++;

			Vec3f n = normals(y,x);
			float angle = getAngle(gtN,n);

			//cout << "n: " << n << " / " << gtN << " / angle: " << angle << endl;
			errorMap(y,x) = angle;

			if(angle > gtNormTol){
				error++;
			}
			if(angle > gtNormTol2){
				error2_count++;
			}
		}
	}
	error2 = (float)error2_count/ (float) numGtPixels;
	return (float)error / (float) numGtPixels;
}

static void getNoccGTimg(const Mat_<float> &groundTruthDisp, const Mat_<uint8_t> &occlusionImg, Mat_<float> &groundTruthDispNocc){
	bitwise_and(groundTruthDisp,groundTruthDisp,groundTruthDispNocc,occlusionImg);
}
