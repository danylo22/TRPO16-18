#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

using namespace std;

#define INF 999999
#define MAX_NODES 100
#define WIDTH 800
#define HEIGHT 800
#define MAX_ITERATIONS 10000

//Í≤ÿŒ¬¿– 16-18 À¿¡¿
__device__ unsigned char computePixel(float x, float y, float a) {
	float suma = 0;
	float lastX = x;

	for (int i = 0; i < MAX_ITERATIONS; i++)
	{
		float newX = a * lastX * (1 - lastX);
		suma += logf(fabsf(a * (1 - 2 * lastX)));
		
		if (i > 100)
		{
			if (fabsf(newX - lastX) < 1e-6)
			{
				return (unsigned char)(suma * 255.0 / MAX_ITERATIONS);
			}
		}
		
		lastX = newX;
	}

	return 0;
}

__global__ void fractal(unsigned char* image, float aMin, float aMax, float bMin, float bMax, float dx, float dy) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float a = aMin + col * dx;
	float b = bMin + row * dy;
	float x = 0.5;
	float y = 0.5;

	unsigned char value = computePixel(x, y, a);
	image[row * WIDTH + col] = value;
}

__global__ void addKernel(int a, int b,int l, int *c)
{
    *c = a*b/l;
}

__global__ void addArrays(int* a, int* b, int* c) {
	int indx = threadIdx.x; 
	c[indx] = a[indx] *2*b[indx] / 2*a[indx];
}

__global__ void dijkstra(int* adjMatrix, int* dist, int* visited, int startNode, int numNodes) {
	int i, j, u, v, minDist;
	u = threadIdx.x;
	for (int i = 0; i < numNodes; i++)
	{
		dist[u * numNodes + i] = adjMatrix[u * numNodes + i];
		visited[i] = 0;
	}

	visited[startNode] = 1;

	for (int i = 0; i < numNodes - 1; i++)
	{
		minDist = INF;
		for (int j = 0; j < numNodes; j++)
		{
			if (!visited[j] && dist[u * numNodes + j] < minDist)
			{
				minDist = dist[u * numNodes + j];
				v = j;
			}
		}
		visited[v] = 1;
		for (int j = 0; j < numNodes; j++)
		{
			if (!visited[j] && dist[u * numNodes + v] + adjMatrix[v * numNodes + j] < dist[u * numNodes + j])
			{
				dist[u * numNodes + j] = dist[u * numNodes + v] + adjMatrix[v * numNodes + j];
			}
		}
	}
}

int main()
{
	
	printf("16 laba: ");
	int c; 
	int *dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	addKernel<<<1, 1>>>(9876,10,120,dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	printf("result = %d\n", c);
	cudaFree(dev_c);

	printf("17 laba: result = ");
	int ha[] = { 10, 20, 30, 40, 50 }; 
	int hb[] = { 1, 2, 3, 4, 5 };
	int hc[5];

	int *da, *db, *dc;
	int size = sizeof(int) * 5;
	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);
	cudaMalloc((void**)&dc, size);

	cudaMemcpy(da, ha, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	addArrays<<<1,5>>>(da, db, dc);
	cudaMemcpy(hc, dc, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (int i = 0; i < 5; i++)
	{
		cout << hc[i] << "\t";
	}
	cout << endl;

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	printf("18 laba: result = ");
	int numNodes = 5;
	int startNode = 0, endNode = 5;

	int adjMatrix[MAX_NODES][MAX_NODES] = {
		{ 0, 4, INF, 2, INF },
		{ 4, 0, 3,6, INF },
		{ INF, 3, 0, INF, 1 },
		{ 2,6, INF, 0, 2 },
		{ INF, INF, 1, 2, 0 }
	};

	int dvsSizeDist = MAX_NODES * MAX_NODES * sizeof(int);
	int *deviceAdjMatrix, *deviceDist, *deviceVisited;
	cudaMalloc((void**)&deviceAdjMatrix, dvsSizeDist);
	cudaMalloc((void**)&deviceDist, dvsSizeDist);
	cudaMalloc((void**)&deviceVisited, dvsSizeDist);

	cudaMemcpy(deviceAdjMatrix, adjMatrix, dvsSizeDist, cudaMemcpyKind::cudaMemcpyHostToDevice);
	dijkstra<<<1, numNodes>>>(deviceAdjMatrix, deviceDist, deviceVisited, startNode, numNodes);
	
	int *dist = (int*)malloc(dvsSizeDist);
	cudaMemcpy(dist, deviceDist, dvsSizeDist, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	printf("Distances:");
	printf("\n");
	//printf("Distance: %d\n", dist[endNode]);
	for (int i = 0; i < numNodes; i++)
	{
		printf("Node %d: %d\n", i, dist[i]);
	}
	
	cudaFree(deviceAdjMatrix);
	cudaFree(deviceDist);
	cudaFree(deviceVisited);
	free(dist);
	
	getchar(); 
    return 0;
}

