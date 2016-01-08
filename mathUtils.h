/*
 * some math helper functions
 */

#pragma once

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif
#define M_PI_float    3.14159265358979323846f

//rounding of positive float values
#if defined(_WIN32)
static float roundf ( float val ) {
	return floor ( val+0.5f );
};
#endif

/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
static float getAngle ( Vec3f v1, Vec3f v2 ) {
	float angle = acosf ( v1.dot ( v2 ) );
	//if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
	if ( angle != angle )
		return 0.0f;
	//if ( acosf ( v1.dot ( v2 ) ) != acosf ( v1.dot ( v2 ) ) )
		//cout << acosf ( v1.dot ( v2 ) ) << " / " << v1.dot ( v2 )<< " / " << v1<< " / " << v2 << endl;
	return angle;
}

