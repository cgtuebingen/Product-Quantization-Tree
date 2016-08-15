Matlab Wrapper
==============

This just uses the slow CPU implementation!

There are several steps to do before you can use PQT in Matlab. However, most of them are just need to be done once. I suggest to do all pre-computation (create tree, create bins) directly in C++ and just use MATLAB for querying.

The file `pqt` contains a functions with several commands specified by the first argument:

    pqt(cmd,...)

The wrapper is build using mex-lock to load all needed information like tree, bins, re-ranking information only once by the commands `init, load_tree, load_bins`. Do not forget to destroy the datastructre in the end by the command `destroy`.

prepare information
-------------------

Adjust the lines in `pqt.cxx`:

		const uint D  = 128;   // dimension of vector
		const uint P  = 2;     // number of parts
		const uint C1 = 16;    // number of clusters in 1st level per part
		const uint C2 = 8;     // number of refinements
		const uint W  = 4;     // neighbors to visit
		const uint LP = 32;    // parts for reranking

Note, the number of bins is `(C1*C2)^P` which is limited by `unsigned int` for limiting the maximal nuber of bins to $4.2*10^9$ by simple modulo-hashing. More bins, are more descriptive and need more time to build the datastructure once. 

Then navigate to the `pqt/matlab` folder in MATLAB. Verify that the path to `Eigen` is set correctly and run the script `make` to compile the wrapper. I assume that your `mex` compiler works. Otherwise follow the instructions of `mex -setup`.

An example is given in the script `test`.

usage
------

A short usage description is:

1. load prepared PQT by `hnd    = pqt('init','test.tree','test.bins');`
2. get a list of sorted candidates by `[ids,dst] = pqt('query',hnd,q);`

commands
----------

To let the datastructure live outside the `mex` calls, you need to use a handle `hnd`.

    hnd = pqt('init','tree.tree','bins.bins')   // load a PQT from files
    hnd = pqt('read_tree',  'tree.tree')        // step of 'init'
    pqt('read_bins', hnd,  'tree.tree')         // step of 'init'
    pqt('save_tree', hnd, 'tree.tree')          // inverse of 'read_tree'
    pqt('save_bins',hnd, 'tree.bins')           // inverse of 'read_tree'
    pqt('destroy',hnd)                          // unlock the mex file
    pqt('query',hnd,q)                          // query a vector
    pqt('build', data)                          // data-format d x n
    pqt('insert',hnd,vec)                       // data-format 1 x d
    hnd = pqt('notify', hnd, 1000)              // request memory for db