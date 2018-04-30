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
#include <mpi.h>
#include <time.h>
#include "sys/time.h"
#define MNIST_DATA_DIR "../mnist_data"

using namespace std;
int main(int argc, char* argv[]) {

    int p,tid,i,j,k,c,f;
    
    MPI_Init(NULL,NULL);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    int numLabels = 10;
    //There are 784 features (one per pixel in a 28x28 image)
    int numFeatures = 784;
    //Each pixel value can take on the value 0 or 1
    int numFeatureValues = 2;
    //image width
    int width = 28;
    //image height
    int height = 28;
    int trainSize=60000;
    unsigned char trainImages[numFeatures*trainSize];
    unsigned char trainLabels[trainSize];
    std::vector<std::vector<unsigned char>> testImages;
    std::vector<unsigned char> testLabels;
    struct timespec start, stop; 
    double time;
    double ProbC[10]={0,0,0,0,0,0,0,0,0,0};
    int CountC[10]={0,0,0,0,0,0,0,0,0,0};
    if(rank==0)
    {
        //Read in the data set from the files
        mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_DIR);
        //Binarize the data set (so that pixels have values of either 0 or 1)
        mnist::binarize_dataset(dataset);
        
        // get training images
        //std::vector<std::vector<unsigned char>> trainImages = dataset.training_images;
        for(int i=0;i<trainSize;i++)
        {
            for(int j=0;j<numFeatures;j++)
            {
                trainImages[i*numFeatures+j]=dataset.training_images[i][j];
            }
        }
        // get training labels
        //std::vector<unsigned char> trainLabels = dataset.training_labels;
        
        for(int i=0;i<trainSize;i++)
        {
            trainLabels[i]=dataset.training_labels[i];
            //cout<<(int)trainLabels[i]<<endl;
        }
        // get test images
        testImages = dataset.test_images;
        // get test labels
        testLabels = dataset.test_labels;
        //train
      
        for(int i=0;i<trainSize;i++)
        {
            int label = static_cast<int>(trainLabels[i]);
            CountC[label]++;
        }

        for(int i=0;i<10;i++)
        {
            ProbC[i]=((double) CountC[i])/trainSize;
        }

        
        if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    }
    MPI_Bcast(trainImages,(numFeatures*trainSize),MPI::UNSIGNED_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(trainLabels,trainSize,MPI::UNSIGNED_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(CountC,10,MPI::INT,0,MPI_COMM_WORLD);
    int CountWhite[numLabels*numFeatures];
    int CountWhiteLocal[numLabels*numFeatures];
    double ProbWhite[numLabels*numFeatures]; 
    double ProbWhiteLocal[numLabels*numFeatures]; 
    ofstream myfile;
    myfile.open("../output/network.txt");
     
    for(int i=0;i<numFeatures*numLabels;i++)
    {
        ProbWhite[i]=0;
        CountWhiteLocal[i]=0;
    }

    for(int i=trainSize/size*rank;i<trainSize/size*(rank+1);i++)
    {
        for(int j=0;j<numFeatures;j++)
        {
            int pixelIntValue = static_cast<int>(trainImages[i*numFeatures+j]);
            if(pixelIntValue==1){
                CountWhiteLocal[trainLabels[i]+j*numLabels]++;
               // cout<<((int)trainLabels[i])<<endl;
                
            }
        }

    }

    MPI_Allreduce(CountWhiteLocal,CountWhite,numFeatures*numLabels,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    //MPI_Bcast(CountWhite,numFeatures*numLabels,MPI_INT,0,MPI_COMM_WORLD);
    
    
    for(int j=rank*(numFeatures/size);j<(rank+1)*(numFeatures/size);j++)
    {
        for(int i=0;i<numLabels;i++)
        {
            ProbWhiteLocal[j*numLabels+i]=((double) CountWhite[j*numLabels+i]+1)/(CountC[i]+2);
            myfile<<i<<" "<<j<<" "<<ProbWhiteLocal[j*numLabels+i]<<endl;

        }

    }
    MPI_Reduce(ProbWhiteLocal,ProbWhite,numFeatures*numLabels,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    // if(rank!=0)
    //     MPI_Send(ProbWhite+rank*(numFeatures/size)*numLabels,numFeatures/size*numLabels,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    // if(rank==0)
    // {
    //     for(int i=1;i<size;i++)
    //     {
    //         MPI_Recv(ProbWhite+i*(numFeatures/size)*numLabels,numFeatures/size*numLabels,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    //     }
    // }
    
    myfile.close();
    if(rank==0)
    {
        if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}       
        time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
        cout<<"exection time: "<<time<<endl; 

        
        //visualization

        //#pragma omp parallel for shared(ProbWhite) private(f,c) schedule(static,numFeatures/p)
        for (c=0; c<numLabels; c++) {
            std::vector<unsigned char> classFs(numFeatures);
            for (f=0; f<numFeatures; f++)
             {
                //TODO: get probability of pixel f being white given class c
                double p = ProbWhite[f*numLabels+c];
               
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
        for(i=0;i<testImages.size();i++)
        {
            vector<double> predict;
            for(k=0;k<numLabels;k++)
                predict.push_back(log(ProbC[k]));
            for(j=0;j<numFeatures;j++)
            {
                for(k=0;k<numLabels;k++)
                {
                    if(testImages[i][j]==1)
                        predict[k]+=log(ProbWhite[j*numLabels+k]);
                    else
                        predict[k]+=log(1-ProbWhite[j*numLabels+k]);
                }
            }
            int guessC=distance(predict.begin(), max_element(predict.begin(), predict.end()));
            int realC=testLabels[i];
            classification[realC][guessC]++;
            countCorrect+=(int) (realC==guessC);
        }
        
        
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
        cout<<"test accuracy: "<<((double) countCorrect)/ 10000<<endl;
        myfile.close();
    }
    


    MPI_Finalize();
    return 0;
}

