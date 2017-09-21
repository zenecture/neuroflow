extern "C"
__global__ void sparse2dense_float(float *densevec, float *data, int *indices, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    densevec[indices[i]] = data[i];
}

extern "C"
__global__ void sparse2dense_double(double *densevec, double *data, int *indices, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    densevec[indices[i]] = data[i];
}