#!/bin/bash
UNZIP_DIR="./data/"

function download_and_unzip() {
    TMPFILE=`mktemp`
    wget "$1" -O $TMPFILE
    unzip -d $UNZIP_DIR $TMPFILE
    rm $TMPFILE
}
mkdir -p $UNZIP_DIR

download_and_unzip http://vision.middlebury.edu/mview/data/data/temple.zip
download_and_unzip http://vision.middlebury.edu/mview/data/data/templeRing.zip
download_and_unzip http://vision.middlebury.edu/mview/data/data/templeSparseRing.zip

download_and_unzip http://vision.middlebury.edu/mview/data/data/dino.zip
download_and_unzip http://vision.middlebury.edu/mview/data/data/dinoRing.zip
download_and_unzip http://vision.middlebury.edu/mview/data/data/dinoSparseRing.zip
