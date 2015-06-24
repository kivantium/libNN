#include <iostream>
#include <cstdio>
#include "svm.h"
using namespace std;

int main(void){
    // number of test data
    const int sample = 100;
    // size (dimension) of input vector
    const int size = 2;
    
    // create SVM (dimension is size)
    svm detector(size);
  
    // train data
    float x[size*sample];
    // label data
    int t[sample];
    
    // load CSV
    FILE *fp = fopen("sample_svm.csv", "r");
    if(fp==NULL) return -1;
    for(int i=0; i<sample; i++){
        // load data
        for(int j=0; j<size; j++) fscanf(fp, "%f,", x+size*i+j);
        // load label
        fscanf(fp, "%d", t+i);
    }

   
    // train SVM 
    detector.train(x, t, sample);
    // show the result
    int correct = 0;
    for(int i=0; i<sample; i++){
       if(detector.predict(x+size*i)>0){
           if(detector.test(i)==1) correct++;
       }else{
           if(detector.test(i)==-1) correct++;
       }
    }
    cout << (correct*100.0/sample) << "%" << endl;
    
    return 0;
}
