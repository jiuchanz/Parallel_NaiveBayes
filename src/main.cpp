#include <iostream>
#include <vector>
#include <map>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "bitmap.hpp"
#include <sstream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <time.h>
#include "sys/time.h"
#define MNIST_DATA_DIR "../mnist_data"

using namespace std;
int main(int argc, char* argv[]) {

    int p,tid,i,j,k,c,f;
    p=atoi(argv[1]);
    omp_set_num_threads(p);
    struct timespec start, stop; 
    double time;
    



    //Read in the data set from the files
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_DIR);
    //Binarize the data set (so that pixels have values of either 0 or 1)
    mnist::binarize_dataset(dataset);
    //There are ten possible digits 0-9 (classes)
    int numLabels = 10;
    //There are 784 features (one per pixel in a 28x28 image)
    int numFeatures = 784;
    //Each pixel value can take on the value 0 or 1
    int numFeatureValues = 2;
    //image width
    int width = 28;
    //image height
    int height = 28;
    //image to print (these two images were randomly selected by me with no particular preference)
    int trainImageToPrint = 50;
    int testImageToPrint = 5434;
    // get training images
    std::vector<std::vector<unsigned char>> trainImages = dataset.training_images;
    // get training labels
    std::vector<unsigned char> trainLabels = dataset.training_labels;
    // get test images
    std::vector<std::vector<unsigned char>> testImages = dataset.test_images;
    // get test labels
    std::vector<unsigned char> testLabels = dataset.test_labels;


   

    //train
    vector<double> ProbC(10);
    vector<int> CountC(10);
    for(int i=0;i<trainLabels.size();i++)
    {
        int label = static_cast<int>(trainLabels[i]);
        CountC[label]++;
    }

   
    for(int i=0;i<10;i++)
    {
        ProbC[i]=((double) CountC[i])/trainLabels.size();
    }

    
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    vector<int> temp(numFeatures,0); 
    vector<vector<int>> CountWhite(numLabels,temp); // row is pixel Fj, col is c

    #pragma omp parallel shared(trainImages,CountWhite,trainLabels) private(i,j,tid)
    {
        tid=omp_get_thread_num();
        #pragma omp for schedule(static,trainImages.size()/p) nowait
        for(i=0;i<trainImages.size();i++)
        {
            for(j=0;j<trainImages[i].size();j++)
            {
                int pixelIntValue = static_cast<int>(trainImages[i][j]);
                if(pixelIntValue==1)
                    CountWhite[trainLabels[i]][j]++;
            }
        }
    }   
    
    ofstream myfile;
    myfile.open("../output/network.txt");
    vector<double> temp1(numFeatures,0);
    vector<vector<double>> ProbWhite(numLabels,temp1);
    #pragma omp parallel for shared(ProbWhite,CountWhite,CountC) private(i,j) schedule(static,numFeatures/p)
    for(j=0;j<numFeatures;j++)
    {
        for(i=0;i<numLabels;i++)
        {
            //tid=omp_get_thread_num();
            //cout<<tid<<" "<<CountWhite[i][j]<<endl;
            ProbWhite[i][j]=((double) CountWhite[i][j]+1)/(CountC[i]+2);
            myfile<<ProbWhite[i][j]<<endl;
        }
    }
    myfile.close();
    
    //evaluation
    #pragma omp parallel for shared(ProbWhite) private(f,c) schedule(static,numFeatures/p)
    for (f=0; f<numFeatures; f++) {
        std::vector<unsigned char> classFs(numFeatures);
        for (c=0; c<numLabels; c++) {
            //TODO: get probability of pixel f being white given class c
            double p = ProbWhite[c][f];
            uint8_t v = 255*p;
            classFs[f] = (unsigned char)v;
        }
        std::stringstream ss;ss << "../output/digit" <<c<<".bmp";
        Bitmap::writeBitmap(classFs, 28, 28, ss.str(), false);
    }
    

    //test

    int countCorrect=0;
    vector<int> temp2(10,0);
    vector<vector<int>> classification(10,temp2);
    #pragma omp parallel for shared(countCorrect,classification) private(i,j,k) schedule(static,testImages.size()/p)
    for(i=0;i<testImages.size();i++)
    {
        vector<double> predict;
        for(k=0;k<10;k++)
            predict.push_back(log(ProbC[k]));
        for(j=0;j<testImages[i].size();j++)
        {
            for(k=0;k<10;k++)
            {
                if(testImages[i][j]==1)
                    predict[k]+=log(ProbWhite[k][j]);
                else
                    predict[k]+=log(1-ProbWhite[k][j]);
            }
        }
        int guessC=distance(predict.begin(), max_element(predict.begin(), predict.end()));
        int realC=testLabels[i];
        classification[realC][guessC]++;
        countCorrect+=(int) (realC==guessC);
    }
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}       
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    cout<<time<<endl; 
    
    myfile.open("../output/classification-summary.txt");
    for(int i=0;i<10;i++)
    {
        for (int j = 0; j < 10; ++j)
        {
            myfile<<setw(5)<<classification[i][j]<<" ";
        }
        myfile<<endl;
    }
    myfile<<((double) countCorrect)/ 10000<<endl;
    myfile.close();


    
    return 0;
}

