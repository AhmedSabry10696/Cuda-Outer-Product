#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>
#include <curand.h>
#include <math.h>

using namespace std;

# define N 10
# define threadsPerBlock 100
# define num_blocks 1

void Outer_Product_Cpu(int A[N], int B[N], int C[N][N], int n)
{
	long long cpu_start = clock();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			C[i][j] = A[i] * B[j];
		}
	}
	cout << "CPU output:\n===============\n\n";
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << C[i][j] << "\t";
		}
		cout << "\n";
	}
	clock_t cpu_stop = clock();
	float CPU_TIME = float(cpu_stop - cpu_start);
	cout << "CPU Time = " << CPU_TIME << " ms\n" << endl;
	cout << "==============================================\n\n";
}

__global__ void Outer_Product_GPU(int *d_A, int *d_B, int **d_C, int n)
{
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	d_C[row][col] = d_A[row] * d_B[col];
}

void main(void)
{

	cout << "\t\t\t*** CUDA TASK ***\n\t\t\t==================\n\n";
	float GPU_TIME;
	// Host Data 
	int A[N] = { 2,4,6,8,10,12,14,16,18,20 };
	int B[N] = { 3,6,9,12,15,18,21,24,27,30 };
	int **C = (int **)malloc(N*sizeof(int*));

	for (int j = 0; j < N; j++)
	{
		C[j] = (int *)malloc(N * sizeof(int));
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i][j] = 2;
		}
	}

	//device data
	int *d_A;
	int *d_B;
	int **d_C;

	long long size = N * sizeof(int);

	// allocate device data
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, N*N*sizeof(int*));

	// copy data from host to device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;   // define 2 events     

	// create 2 events 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);   // begin START event

	// call kernal
	Outer_Product_GPU <<< 1,dim3(10,10) >>> (d_A, d_B, d_C, N);

	// copy data from device to host
	cudaMemcpy(C, d_C, N*size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);    // begin STOP event
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_TIME, start, stop); // calculate execution time

	// destroy 2 events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "GPU output:\n===============\n\n";
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << C[i][j] << "\t";
		}
		cout << "\n";
	}
	cout << "GPU Time = " << GPU_TIME << " ms\n" << endl;
	cout << "==============================================\n\n";
	Outer_Product_Cpu(A,B,C,N);
}