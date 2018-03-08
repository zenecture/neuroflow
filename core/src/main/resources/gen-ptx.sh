PATH="$PATH:/Developer/NVIDIA/CUDA-8.0/bin/"

if [ `uname` == Darwin ]; then
   NVCC_OPTS="-gencode arch=compute_30,code=sm_30 -ccbin /usr/bin/g++ $NVCC_OPTS"
fi

nvcc $NVCC_OPTS -D TYPE=float  --ptx cuda/matrix_kernels.cu -o cuda/matrix_kernels_float.ptx
nvcc $NVCC_OPTS -D TYPE=double --ptx cuda/matrix_kernels.cu -o cuda/matrix_kernels_double.ptx

nvcc $NVCC_OPTS -D TYPE=float  --ptx cuda/vector_kernels.cu -o cuda/vector_kernels_float.ptx

nvcc $NVCC_OPTS -D TYPE=float  --ptx cuda/matrix_convops.cu -o cuda/matrix_convops_float.ptx
nvcc $NVCC_OPTS -D TYPE=double --ptx cuda/matrix_convops.cu -o cuda/matrix_convops_double.ptx

nvcc $NVCC_OPTS -D TYPE=float  --ptx cuda/matrix_misc.cu -o cuda/matrix_misc_float.ptx
nvcc $NVCC_OPTS -D TYPE=double --ptx cuda/matrix_misc.cu -o cuda/matrix_misc_double.ptx

