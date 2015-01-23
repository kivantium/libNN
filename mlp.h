#ifndef LIBNN_MLP
#define LIBNN_MLP

#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

class mlp {
private:
    // number of each layer
    int in, hid, out;
    // input layer
    float *xi1, *xi2, *xi3;
    // output layer
    float *o1, *o2, *o3;
    // error value
    float *d2, *d3;
    // wait
    float *w1, *w2;

    //sigmoid function
    float sigmoid(float x){
        return 1/(1+exp(-x));
    }

    //derivative of sigmoid function
    float d_sigmoid(float x){
        return (1-sigmoid(x))*sigmoid(x);
    }

    //caluculate forward propagation of input x
    void forward(float *x){
        //calculation of input layer
        for(int j=0; j<in; j++){
            xi1[j] = x[j];
            xi1[in] = 1;
            o1[j] = xi1[j];
        }
        o1[in] = 1;

        //caluculation of hidden layer
        for(int j=0; j<hid; j++){
            xi2[j] = 0;
            for(int i=0; i<in+1; i++){
                xi2[j] += w1[i*hid+j]*o1[i];
            }
            o2[j] = sigmoid(xi2[j]);
        }
        o2[hid] = 1;

        //caluculation of output layer
        for(int j=0; j<out; j++){
            xi3[j] = 0;
            for(int i=0; i<hid+1; i++){
                xi3[j] += w2[i*out+j]*o2[i];
            }
            o3[j] = xi3[j];
        }
    }

public:
    mlp(const std::string filename){
        std::ifstream ifs(filename.c_str());
        if(!ifs){
            std::cout << "File does not exist" << std::endl;
            exit(1);
        }
        std::string str;
        ifs >> str;
        if(str!="LIBNN_MLP_0"){
            std::cout << "File type error!" << std::endl;
            exit(1);
        }
        ifs >> in >> hid >> out;
        // allocate inputs
        xi1 = new float[in+1];
        xi2 = new float[hid+1];
        xi3 = new float[out];

        // allocate outputs
        o1 = new float[in+1];
        o2 = new float[hid+1];
        o3 = new float[out];

        // allocate errors
        d2 = new float[hid+1];
        d3 = new float[out];

        // allocate memories
        w1 = new float[(in+1)*hid];
        w2 = new float[(hid+1)*out];

        for(int i=0; i<(in+1)*hid; i++)  ifs >> w1[i];
        for(int i=0; i<(hid+1)*out; i++) ifs >> w2[i]; 
    }
    /* constructor
     * n_in:  number of input layer
     * n_hid: number of hidden layer
     * n_out: number of output layer */
    mlp(int n_in, int n_hid, int n_out){
        // unit number
        in = n_in;
        hid = n_hid;
        out = n_out;

        // allocate inputs
        xi1 = new float[in+1];
        xi2 = new float[hid+1];
        xi3 = new float[out];

        // allocate outputs
        o1 = new float[in+1];
        o2 = new float[hid+1];
        o3 = new float[out];

        // allocate errors
        d2 = new float[hid+1];
        d3 = new float[out];

        // allocate memories
        w1 = new float[(in+1)*hid];
        w2 = new float[(hid+1)*out];

        // initialize wait
        // range depends on the research of Y. Bengio et al. (2010)
        srand ((unsigned) (time(0)));
        float range = std::sqrt(6)/std::sqrt(in+hid+2);
        std::srand ((unsigned) (std::time(0)));
        for(int i=0; i<(in+1)*hid; i++) w1[i] = (float) 2*range*std::rand()/RAND_MAX-range;
        for(int i=0; i<(hid+1)*out; i++) w2[i] = (float) 2*range*std::rand()/RAND_MAX-range;
    }

    // deconstructor
    ~mlp(){
        // delete arrays
        delete[] xi1, xi2, xi3;
        delete[] o1, o2, o3;
        delete[] d2, d3;
        delete[] w1, w2; 
    }

    /* train: multi layer perceptron
     * x: train data(number of elements is in*N)
     * t: correct label(number of elements is N)
     * N: data size
     * repeat: repeat times
     * eta: learning rate */
    void train(float x[], int t[], float N, int repeat=1000, float eta=0.1){
        for(int times=0; times<repeat; times++){
            for(int sample=0; sample<N; sample++){
                // forward propagation
                forward(x+sample*in);

                // calculate the error of output layer
                for(int j=0; j<out; j++){
                    if(t[sample] == j) d3[j] = o3[j]-1;
                    else d3[j] = o3[j];
                }
                // update the wait of output layer
                for(int i=0; i<hid+1; i++){
                    for(int j=0; j<out; j++){
                        w2[i*out+j] -= eta*d3[j]*o2[i];
                    }
                }
                // calculate the error of hidden layer
                for(int j=0; j<hid+1; j++){
                    float tmp = 0;
                    for(int l=0; l<out; l++){
                        tmp += w2[j*out+l]*d3[l];
                    }
                    d2[j] = tmp * d_sigmoid(xi2[j]);
                }
                // update the wait of hidden layer
                for(int i=0; i<in+1; i++){
                    for(int j=0; j<hid; j++){
                        w1[i*hid+j] -= eta*d2[j]*o1[i];
                    }
                }
            }
        }
    }

    // return most probable label to the input x
    int predict(float x[]){
        // forward propagation
        forward(x);
        // biggest output means most probable label
        float max = o3[0];
        int ans = 0;
        for(int i=1; i<out; i++){
            if(o3[i] > max){
                max = o3[i];
                ans = i;
            }
        }
        return ans;
    }

     // save mlp to csv file
    void save(const std::string filename){
        std::ofstream ofs(filename.c_str());
        ofs << "LIBNN_MLP_0" << std::endl;
        ofs << in << " " << hid << " " << out << std::endl;
        for(int i=0; i<(in+1)*hid-1; i++) ofs << w1[i] << " ";
        ofs << w1[(in+1)*hid-1] << std::endl;
        for(int i=0; i<(hid+1)*out-1; i++) ofs << w2[i] << " "; 
        ofs << w2[(hid+1)*out-1] << std::endl;
        ofs.close();
    }
};

#endif
