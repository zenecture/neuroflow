#define SWAPS_PER_RUN 64
#define VL  64

extern "C"
__global__ void batch_sswap(int nswaps, int n, float *A, int lda, int *ipiv)
{
	unsigned int tid = threadIdx.x + VL * blockIdx.x;
	if( tid >= n ) return;

	float *d_A = A + tid;
	for (int i = 0; i < nswaps; i++)
	{
		int j = ipiv[i];
		float temp = d_A[i*lda];
		d_A[i*lda] = d_A[j*lda];
		d_A[j*lda] = temp;
	}
}

extern "C"
__global__ void batch_dswap(int nswaps, int n, double *A, int lda, int *ipiv)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if( tid >= n ) return;

	double *d_A = A + tid;
	for (int i = 0; i < nswaps; i++)
	{
		int j = ipiv[i];
		double temp = d_A[i*lda];
		d_A[i*lda] = d_A[j*lda];
		d_A[j*lda] = temp;
	}
}