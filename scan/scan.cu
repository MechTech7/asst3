#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
CUDA_upsweep_kernel(int* device_result, int N, int two_dplus, int two_d){
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx = thread_idx * two_dplus;
    if (idx < N){
        device_result[idx+two_dplus-1] += device_result[idx+two_d-1];
    }
}

__global__ void
CUDA_downsweep_kernel(int* device_result, int N, int two_dplus, int two_d){
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx = thread_idx * two_dplus;
    if (idx < N){
        int t = device_result[idx+two_d-1];
        device_result[idx+two_d-1] = device_result[idx+two_dplus-1];
        device_result[idx+two_dplus-1] += t;
    }
}
// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

void thread_block_choice(int count, int* numBlocks, int* threadsPer){
    if (count > *threadsPer){
        *numBlocks = (count + *threadsPer - 1) / *threadsPer;
    }
    else{
        *numBlocks = 1;
        *threadsPer = count;
    }
}

void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    //TODO: Set these values for NVIDIA K80
    int threadsPerBlock = 64;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;


    for (int two_d = 1; two_d <= N/2; two_d*=2){
        int two_dplus = 2 * two_d;

        int count = N / two_dplus;

        threadsPerBlock = 64;
        thread_block_choice(count, &numBlocks, &threadsPerBlock);
        
        CUDA_upsweep_kernel<<<numBlocks, threadsPerBlock>>>(result, N, two_dplus, two_d);
    }
    int cap = 0;
    cudaMemcpy(result+(N-1), &cap, sizeof(int), cudaMemcpyHostToDevice);
    //TODO: Find a faster way to do this (ask about it in OH)

    for (int two_d = N/2; two_d >= 1; two_d /= 2){
        int two_dplus = 2 * two_d;

        int count = N / two_dplus;
        
        threadsPerBlock = 64;
        thread_block_choice(count, &numBlocks, &threadsPerBlock);

        CUDA_downsweep_kernel<<<numBlocks, threadsPerBlock>>>(result, N, two_dplus, two_d);
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void map_bool(int *device_input, int *device_result, int N){
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    //TODO: Change this to improve utilization
    if (thread_idx < (N-1))
        device_result[thread_idx] = (device_input[thread_idx] == device_input[thread_idx + 1]);
} 

__global__ void map_to_result(int *device_input, int *device_bool_arr, int *device_result, int N){
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

    //TODO: Fix this divergence
    if (thread_idx < N && device_bool_arr[thread_idx] == 1){
        int result_idx = device_input[thread_idx];
        device_result[result_idx] = thread_idx;
    }
        
}
/*    int host_mask[length]; //= (int*)malloc(length*sizeof(int));

    cudaMemcpy(host_mask, prefix_sum, length*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < length; i++){
        printf("%d, ", host_mask[i]);
    }
    printf("\n");*/

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int* bool_mask = nullptr;
    int* prefix_sum = nullptr;

    cudaMalloc(&bool_mask, sizeof(int)*length);
    cudaMalloc(&prefix_sum, sizeof(int)*length);

    cudaMemset(bool_mask, 0, sizeof(int)*length);
    int threadsPerBlock = 32;
    int numBlocks = 1;

    if(length > threadsPerBlock){
        numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    }

    map_bool<<<numBlocks, threadsPerBlock>>>(device_input, bool_mask, length);
    cudaMemcpy(prefix_sum, bool_mask, sizeof(int)*length, cudaMemcpyDeviceToDevice);

    exclusive_scan(bool_mask, length, prefix_sum);
    map_to_result<<<numBlocks, threadsPerBlock>>>(prefix_sum, bool_mask, device_output, length);

    int last_bin;
    int last_count;
    cudaMemcpy(&last_bin, bool_mask + (length - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_count, prefix_sum + (length - 1), sizeof(int), cudaMemcpyDeviceToHost);

    if (last_bin)
        last_count += 1;
    return last_count;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
