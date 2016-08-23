mkdir workdir
# get SIFT-1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -P workdir
cd workdir 
# extract
mv sift.tar.gz ANN_SIFT1M.tar.gz
tar -zxvf ANN_SIFT1M.tar.gz
cd ../
# convert to our data format
./build/convert_fvecs --fvecs workdir/sift/sift_base.fvecs --umem workdir/sift/base.umem
./build/convert_fvecs --fvecs workdir/sift/sift_learn.fvecs --umem workdir/sift/learn.umem
./build/convert_fvecs --fvecs workdir/sift/sift_query.fvecs --umem workdir/sift/query.umem
./build/convert_ivecs --ivecs workdir/sift/sift_groundtruth.ivecs --imem workdir/sift/groundtruth.imem