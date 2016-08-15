#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "../helper.hpp"
#include "../filehelper.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    string path = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/idx_50M";
	string fs =  path+".ivecs";
	// ****************************** HEADER **********************

	uint n1 = 0;
	uint d1 = 0;
	readJegouHeader<int>(fs.c_str(), n1, d1);

	cout << "header dim "<< d1<<endl;
    cout << "header num "<< n1<<endl;
    cout << "filesize   "<< (sizeof(int)*n1*d1)/1000000.0 << " mb "<<endl;
	// ****************************** READ **********************
    

    uint n = 0;
    uint d = 0;

    int *data = readJegou<int>(fs.c_str(), n, d);

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
    //uint8_t *out = batchcast<float,uint8_t>(data,d*n);
    int *out = data;
    // ****************************** WRITE **********************
    //write<int>( path+".imem", n, d, out, n*d); 
    write<int>( "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT50M/groundtruth.imem", n, d, out, n*d); 

    // ****************************** TEST **********************
    uint nn = 0;
    uint dd = 0;
    int *test_in = new int[n*d];
    read( "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT50M/groundtruth.imem",nn,dd,test_in,n*d);

    cout << "test dump:"<<endl;
    cout << "load dim "<< dd<<endl;
    cout << "load num "<< nn<<endl;
    for (int j = 0; j < 5; ++j)
    {
    	cout << j << ": ";
    	for (int i = 0; i < 10; ++i)
	    {
	    	cout << (int)test_in[j*d+i]<< " ";
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