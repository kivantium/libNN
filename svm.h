/* svm.h */
#ifndef LIBNN_SVM
#define LIBNN_SVM

#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

class svm {
    // dimension of vector
    int dim;
    // lagrange multiplier
    float *a;
    // train data
    float *point;
    // train label
    float *target;
    // error cache
    float *E;
    // threshold
    float b;
    float eps;
    float tol;
    float C;
    int N;

    // kernel function (gauss kernel)
    float kernel(float *x1, float *x2, float delta=1.0){
        float tmp = 0;
        for(int i=0; i<dim; ++i){
            tmp += (x1[i]-x2[i])*(x1[i]-x2[i]);
        }
        return exp(-tmp/(2.0*delta*delta));
    }
public:
    float predict(float *x){
        float tmp = 0;
        for(int i=0; i<N; ++i){
            tmp += a[i]*target[i]*kernel(x, point+i*dim);
        }
        return tmp-b;
    }
    int takeStep(int i1, int i2){
        if(i1 == i2) return 0;
        float alph1 = a[i1];
        float alph2 = a[i2];
        int y1 = target[i1];
        int y2 = target[i2];
        float E1 = E[i1];
        float E2 = E[i2];
        int s = y1*y2;
        // Compute L, H
        float L, H;
        if(y1!=y2){
            L = std::max((float)0.0, alph2-alph1);
            H = std::min(C, C+alph2-alph1);
        } else{
            L = std::max((float)0.0, alph1+alph2-C);
            H = std::min(C, alph1+alph2);
        }
        if(L==H) return 0;

        float k11 = kernel(point+i1*dim, point+i1*dim);
        float k12 = kernel(point+i1*dim, point+i2*dim);
        float k22 = kernel(point+i2*dim, point+i2*dim);

        float eta = 2*k12-k11-k22;

        float a1, a2;

        if(eta < 0){
            a2 = alph2 - y2*(E1-E2)/eta;
            if(a2<L) a2 = L;
            else if(a2>H) a2 = H;
        } else{
            a1 = a[i1];
            a2 = a[i2];
            float v1 = predict(point+i1*dim) - b - y1*a1*k11 - y2*a2*k12; // assume K12 = K21
            float v2 = predict(point+i2*dim) - b - y1*a1*k12 - y2*a2*k22;
            float Wconst = 0;
            for(int i=0; i<N; ++i){
                if(i!=i1 && i!=i2) Wconst+=a[i1];
            }
            for(int i=0; i<N; ++i){
                for(int j=0; j<N; ++j){
                    if(i!=i1 && i!=i2 && j!=i1 && j!=i2){
                        Wconst += target[i]*target[j]*kernel(point+i*dim, point+j*dim)*a[i]*a[j]/2.0;
                    }
                }
            }
            a2 = L;
            a1 = y1*a[i1]+y2*a[i2]-y2*L;
            float Lobj = a1+a2-k11*a1*a1/2.0-k22*a2*a2/2.0-s*k12*a1*a2/2.0
                -y1*a1*v1-y2*a2*v2+Wconst;
            a2 = H;
            a1 = y1*a[i1]+y2*a[i2]-y2*H;
            float Hobj = a1+a2-k11*a1*a1/2.0-k22*a2*a2/2.0-s*k12*a1*a2/2.0
                -y1*a1*v1-y2*a2*v2+Wconst;
            if(Lobj > Hobj + eps) a2 = L;
            else if(Lobj < Hobj - eps) a2 = H;
            else a2 = alph2;
        }

        if(a2 < 1e-8) a2 = 0;
        else if(a2 > C-1e-8) a2 = C;

        if(abs(a2-alph2) < eps*(a2+alph2+eps)) return 0;

        a1 = alph1+s*(alph2-a2);

        float b_old = b;
        float b1 = E1 + y1*(a1-a[i1])*k11 + y2*(a2-a[i2])*k12 + b;
        float b2 = E2 + y1*(a1-a[i1])*k12 + y2*(a2-a[i2])*k22 + b;
        if(b1==b2) b = b1;
        else b = (b1+b2)/2;
        float da1 = a1-a[i1];
        float da2 = a2-a[i2];
        for(int i=0; i<N; ++i){
            E[i] = E[i] + y1*da1*kernel(point+i1*dim, point+i*dim)
                +y2*da2*kernel(point+i2*dim, point+i*dim) + b_old - b;
        }

        a[i1] = a1;
        a[i2] = a2;

        return 1;
    }
    int examineExample(int i2){
        float y2 = target[i2];
        float alph2 = a[i2];
        float E2 = E[i2];
        float r2 = E2*y2;
        int i1 = 0;

        if((r2<-tol && alph2<C) || (r2>tol && alph2>0)){
            int number = 0;
            for(int i=0; i<N; ++i){
                if(a[i]!=0 || a[i]!=C) number++;
            }
            if(number > 1){
                float max = 0;
                for(int i=0; i<N; ++i){
                    if(abs(E[i]-E2) > max){
                        max = abs(E[i]-E2);
                        i1 = i;
                    }
                }
                if(takeStep(i1, i2)) return 1;
            }
            srand((unsigned)time(NULL));
            i1 = rand()%N;
            if(takeStep(i1, i2)) return 1;
        }
        return 0;
    }

    /* constructor
     * dimension: dimension of input
     * C: constant
     * */
    svm(int dimension, float argC=1.0){
        dim = dimension;
        C = argC;
        eps = 0.01;
        tol = 0.01;
    }

    // deconstructor
    ~svm(){
        delete[] a;
        delete[] point;
        delete[] target;
        delete[] E;
    }
    int test(int i){
        return target[i];
    }
    /* train svm
     * x: train data(size is dim*N)
     * t: train label(size is N)
     * size: data size
     */
    void train(float x[], int t[], int size){
        N = size;
        // initialize a
        a = new float[N];
        for(int i=0; i<N; ++i) a[i] = 0;
        point = new float[N*dim];
        for(int i=0; i<N; ++i){
           for(int j=0; j<dim; ++j) point[i*dim+j] = x[i*dim+j];
        }
        target = new float[N];
        for(int i=0; i<N; ++i) target[i] = t[i];
        E = new float[N];
        for(int i=0; i<N; ++i) E[i] = -target[i];
        float threshold = 0;
        int numChanged = 0;
        int examineAll = 1;
        while(numChanged>0 || examineAll){
            numChanged = 0;
            if(examineAll){
                for(int i=0; i<N; ++i) numChanged += examineExample(i);
            }
            else{
                for(int i=0; i<N; ++i){
                    if(a[i]!=0 && a[i]!=C) numChanged += examineExample(i);
                }
            }
            if(examineAll == 1){
                examineAll = 0;
            }
            else if(numChanged == 0){
                examineAll = 1;
            }
        }
    }
};
#endif
