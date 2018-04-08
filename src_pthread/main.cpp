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
#include <pthread.h>
#include <time.h>
#include "sys/time.h"
#define MNIST_DATA_DIR "../mnist_data"

using namespace std;

std::vector<std::vector<unsigned char>> trainImages;
std::vector<unsigned char> trainLabels;
std::vector<std::vector<unsigned char>> testImages;
std::vector<unsigned char> testLabels;
vector<double> ProbC(10);
vector<int> CountC(10);
vector<vector<int>> CountWhite;
pthread_barrier_t barr;
vector<vector<double>> ProbWhite;
int p;
//There are 784 features (one per pixel in a 28x28 image)
int numFeatures = 784;
//There are ten possible digits 0-9 (classes)
int numLabels = 10;


void * entry(void *arg)
{
    long rank=(long) arg;
    for(int i=trainImages.size()/p*rank;i<trainImages.size()/p*(rank+1);i++)
    {
        for(int j=0;j<trainImages[i].size();j++)
        {
            int pixelIntValue = static_cast<int>(trainImages[i][j]);
            if(pixelIntValue==1)
                CountWhite[trainLabels[i]][j]++;
        }
    }

    // Synchronization point
    int rc = pthread_barrier_wait(&barr);
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier\n");
        exit(-1);
    }

    ofstream myfile;
    myfile.open("../output/network.txt");
    for(int j=rank*(numFeatures/p);j<(rank+1)*(numFeatures/p);j++)
    {
        for(int i=0;i<numLabels;i++)
        {
            //tid=omp_get_thread_num();
            //cout<<tid<<" "<<CountWhite[i][j]<<endl;
            ProbWhite[i][j]=((double) CountWhite[i][j]+1)/(CountC[i]+2);
            myfile<<i<<" "<<j<<" "<<ProbWhite[i][j]<<endl;
        }
    }
    myfile.close();
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {

    
    p=atoi(argv[1]);
    struct timespec start, stop; 
    double time;
    



    //Read in the data set from the files
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_DIR);
    //Binarize the data set (so that pixels have values of either 0 or 1)
    mnist::binarize_dataset(dataset);
      
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
    trainImages = dataset.training_images;
    // get training labels
    trainLabels = dataset.training_labels;
    // get test images
    testImages = dataset.test_images;
    // get test labels
    testLabels = dataset.test_labels;


   

    //train
    
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
    vector<vector<int>> temp2(numLabels,temp); // row is pixel Fj, col is c
    CountWhite=temp2;
    vector<double> temp1(numFeatures,0);
    vector<vector<double>> temp3(numLabels,temp1);
    ProbWhite=temp3;
    pthread_t threads[p];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    if(pthread_barrier_init(&barr,NULL,p))
    {
        cout<<"Could not create a barrier"<<endl;
        return -1;
    }
    for(int t=0;t<p;t++)
    {
        if(pthread_create(&threads[t],NULL,entry,(void*)t))
        {
            cout<<"Could not create thread "+t<<endl;
            return -1;
        }
    }
    for(int t=0;t<p;t++)
    {
        if(pthread_join(threads[t],NULL))
        {
            cout<<"Could not join thread "+t<<endl;
            return -1;
        }
    }



    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}       
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    cout<<time<<endl; 
    

    //visualization
    for (int c=0; c<numLabels; c++)  
    {
        std::vector<unsigned char> classFs(numFeatures);
        for (int f=0; f<numFeatures; f++)
        {
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
    vector<int> temp4(10,0);
    vector<vector<int>> classification(10,temp4);
    for(int i=0;i<testImages.size();i++)
    {
        vector<double> predict;
        for(int k=0;k<10;k++)
            predict.push_back(log(ProbC[k]));
        for(int j=0;j<testImages[i].size();j++)
        {
            for(int k=0;k<10;k++)
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
    ofstream myfile;
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

