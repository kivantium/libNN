#include <iostream>
#include <cstdio>
#include "../mlp.h"
using namespace std;

int main(void){
    // number of test data
    const int sample = 150;
    // size (dimension) of input vector
    const int size = 2;
    // number of labels
    const int label = 3;
    
    // create MLP(input 2, hidden 3, output 3)
    // number of hidden layer is your choice
    mlp net(size,3,label);
  
    // train data
    float x[size*sample];
    // label data
    int t[sample];
    
    // load CSV
    FILE *fp = fopen("sample.csv", "r");
    if(fp==NULL) return -1;
    for(int i=0; i<sample; i++){
        // load label
        fscanf(fp, "%d,", t+i);
        // load data
        for(int j=0; j<size; j++) fscanf(fp, "%f,", x+size*i+j);
    }
   
    // train MLP
    // repeat time is your choice
    const int repeat = 500;
    net.train(x, t, sample, repeat);
    // show the result
    for(int i=0; i<sample; i++){
        cout << net.predict(x+size*i) << endl;
    }
    
    // save and load example
    /*
    net.save("data.csv");
    mlp net2("data.csv");
    for(int i=0; i<sample; i++){
        cout << net2.predict(x+size*i) << endl;
    }
    */
    return 0;
}
