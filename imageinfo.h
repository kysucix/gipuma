#pragma once
#include "managed.h"

class __align__(128) ImageInfo : public Managed{
public:
    // Image size
    int cols;
    int rows;

    // Total number of pixels
    int np;

    // Total number of bytes (may be different when padded)
    int nb;
};
