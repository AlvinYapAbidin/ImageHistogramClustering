#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "photohist.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    /* The goal of the project is to group pictures based on the subject, features and objects*/ 

    // Read each picture in the folder

    // Apply histogram to each picture, possibly add  feature detection for better accuracy

    // Experiment with other features that 

    // Use cluster algorithm to group similar i.e. hashing techniques for similar image detection

    // Gui to help choose pictures within clusters



    Photohist::run("photos");
}