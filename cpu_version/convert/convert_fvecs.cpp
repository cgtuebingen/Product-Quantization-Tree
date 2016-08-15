#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "../helper.hpp"
#include "../filehelper.hpp"

using namespace std;

typedef float out_t;

int main(int argc, char const *argv[])
{
    string path = "/graphics/projects/scratch/ANNSIFTDB/ANN_RAND100K/base";
    string fs = path+".fvecs";
    // ****************************** HEADER **********************

    uint n1 = 0;
    uint d1 = 0;
    readJegouHeader<float>(fs.c_str(), n1, d1);

    cout << "header dim "<< d1<<endl;
    cout << "header num "<< n1<<endl;
    cout << "filesize   "<< (sizeof(float)*n1*d1)/1000000.0 << " mb "<<endl;
	// ****************************** READ **********************
    

    uint n = 0;
    uint d = 0;

    float *data = readJegou<float>(fs.c_str(), n, d);

    cout << "load dim "<< d<<endl;
    cout << "load num "<< n<<endl;

    cout << "dump:"<<endl;
    for (int j = 0; j < 5; ++j)
    {
    	cout << j << ": ";
    	for (int i = 0; i < 10; ++i)
	    {
	    	cout << data[j*d+i]<< " ";
	    }
	    cout << " ..." << endl;
    }
    cout << endl;

    // ****************************** CONVERT **********************
    out_t *out = batchcast<float,out_t>(data,d*n);

    // ****************************** WRITE **********************
    write(path+".umem", n, d, out, n*d); 

    // ****************************** TEST **********************
    uint nn = 0;
    uint dd = 0;
    out_t *test_in = new out_t[n*d];
    read(path+".umem",nn,dd,test_in,n*d);

    cout << "test dump:"<<endl;
    cout << "load dim "<< dd<<endl;
    cout << "load num "<< nn<<endl;
    for (int j = 0; j < 5; ++j)
    {
    	cout << j << ": ";
    	for (int i = 0; i < 10; ++i)
	    {
	    	cout << (float)test_in[j*d+i]<< " ";
	    }
	    cout << " ..." << endl;
    }
    cout << endl;

    for (int j = 0; j < nn; ++j)
    {
        
        for (int i = 0; i < dd; ++i)
        {
            if(test_in[j*d+i] != out[j*d+i]){
                cout << "ERROR: MISSMATCH vector "<< j << " coordinate " << i<<endl;
                return 1;
            }
        }
    }

    cout << "INFO: identical files"<< endl;

    return 0;
}