// zeros out the part of a block above the diagonal
// sets ones on the diagonal (kernel by V.Volkov)
extern "C" {

__global__ void enforceLU( float *matrix, int lda )
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if( i <= j )
        matrix[i + j*lda] = (i == j) ? 1 : 0;
}

}


// zeros out the whole part of matrix above the diagonal (not just a block)
extern "C" {

__global__ void zerosU(int m, int n, float *matrix, int lda, int incl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= m || j >= n) return;

    if (i < j)
        matrix[i + j*lda] = 0;
    else if (i == j && incl)
        matrix[i + j*lda] = 0;
}

}

// zeros out the whole part of matrix below the diagonal
extern "C" {

__global__ void zerosL(int m, int n, float *matrix, int lda, int incl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= m || j >= n) return;

    if( i > j )
        matrix[i + j*lda] = 0;
    else if (i == j && incl)
        matrix[i + j*lda] = 0;
}

}