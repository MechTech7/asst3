#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"
//##############RENDERING CONSTANTS##############
const int REGION_WIDTH = 128;
const int REGION_HEIGHT = 128;
const int SPACE_WIDTH = 4;//16;
const int SPACE_HEIGHT = 4;//16;

const int CIRCLE_CHUNK_SIZE = 10;


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

__global__ void kernelOwnRegion(int* ownershipSpace, int circle_chunk_index, int circleCount, float rel_top_y, float rel_top_x, int CIRCLE_CHUNK_SIZE, int REGION_WIDTH, int REGION_HEIGHT, int SPACE_WIDTH, int SPACE_HEIGHT){
    //TODO: Debug me!
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int regionX = index % SPACE_WIDTH;
    int regionY = index / SPACE_WIDTH;

    float flHeight = static_cast<float>(cuConstRendererParams.imageHeight);
    float flWidth = static_cast<float>(cuConstRendererParams.imageWidth);

    float region_L = static_cast<float>(regionX * REGION_WIDTH)/flWidth + rel_top_x;
    float y_sub = static_cast<float>(REGION_HEIGHT * regionY) / flHeight;
    float region_T = rel_top_y - y_sub;
    float region_B = region_T - static_cast<float>(REGION_HEIGHT) / flHeight;
    float region_R = region_L + static_cast<float>(REGION_WIDTH) / flWidth;

    //TODO: Move this to CPU Calculation
    int ownershipIdx = index * CIRCLE_CHUNK_SIZE; //Index to output to in the ownership space
    int circleSpaceIDX = CIRCLE_CHUNK_SIZE * circle_chunk_index; //Starting index in the circle space

    //TODO: Make sure we don't read past the end of circlesArr if > numCircles, stop
    
    for (int i = 0; i < circleCount; i++){
        int index3 = 3 * (i + circleSpaceIDX);
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[i];

        int circIn = circleInBoxConservative(p.x, p.y, rad, region_L, region_R, region_T, region_B);
        ownershipSpace[ownershipIdx + i] = circIn;
        //printf("!!!!%d, ", circIn);
    }
}


__global__ void compactRegionOwnership(int* ownershipSpace, int* compactOwnershipSpace){
//Use Exclusive_sum to compute an Ownership array with indices instead of binary values (parallel for each index in the SPACE)

}

__global__ void renderRegion(int* ownershipSpace, int circleCount, int circle_chunk_idx, int SPACE_WIDTH, int SPACE_HEIGHT, int REGION_WIDTH, int REGION_HEIGHT, int CIRCLE_CHUNK_SIZE, int top_x, int top_y){
//Pixel-parallel method (parallel to pixels in the REGION) to calculate
//TODO: optimize this method and remove divergence
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int space_size = SPACE_WIDTH * REGION_WIDTH;
    int space_x = (index % space_size);
    int space_y = (index / space_size);

    int pixel_x = space_x + top_x;
    int pixel_y = space_y + top_y;
    pixel_y = (cuConstRendererParams.imageHeight - 1) - pixel_y;

    float invWidth = 1.0f / static_cast<float>(cuConstRendererParams.imageWidth);
    float invHeight = 1.0f / static_cast<float>(cuConstRendererParams.imageHeight);

    float fpix_x = static_cast<float>(pixel_x);
    float fpix_y = static_cast<float>(pixel_y);
    
    
    int pixelInImage = (pixel_x < cuConstRendererParams.imageWidth && pixel_y < cuConstRendererParams.imageHeight);
    if (!pixelInImage) return;

    //printf("-----------------Here!---------------------");
    int ownershipStart = CIRCLE_CHUNK_SIZE * ((space_y / REGION_HEIGHT)*SPACE_WIDTH + (space_x / REGION_WIDTH)); //Go backwards and calculate your region index (inside space)
    
    int positionStart = CIRCLE_CHUNK_SIZE * circle_chunk_idx;

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixel_y * cuConstRendererParams.imageWidth + pixel_x)]);
    
    int f_val = (space_y / REGION_HEIGHT) % 2;
    int s_val = (space_x / REGION_WIDTH) % 2;

    /*if (f_val == s_val){
        //printf("[%d] (%f, %f) [%d, %d]\n", index, fpix_y*invHeight, fpix_x*invWidth, pixel_y, pixel_x);
        float4 newColor;
        newColor.x = 1.0f;
        newColor.y = 0.0f;
        newColor.z = 0.0f;
        newColor.w = 1.0f;
        *imgPtr = newColor;
    }*/


    for (int i = 0; i < circleCount; i++){
        float3 p = *(float3*)(&cuConstRendererParams.position[3 * (i + positionStart)]);


        if (ownershipSpace[ownershipStart + i] == 1){
            //printf("!Rendering! \n");
            float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixel_y * cuConstRendererParams.imageWidth + pixel_x)]);
            float2 pixelCenterNorm = make_float2((fpix_x + 0.5f)*invWidth,
                                                 (fpix_y - 0.5f)*invHeight);

            //if (index == SPACE_WIDTH * SPACE_HEIGHT * REGION_HEIGHT * REGION_WIDTH - 1)
            //    printf("----------------ENDEN: [%f, %f]\n---------------", fpix_y, fpix_x);
            shadePixel(i + positionStart, pixelCenterNorm, p, imgPtr);
        }

    }
}
////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    cudaMalloc(&cudaOwnershipSpace, sizeof(int) * SPACE_HEIGHT * SPACE_WIDTH * CIRCLE_CHUNK_SIZE);
    cudaMalloc(&cudaCompactOwnershipSpace, sizeof(int) * SPACE_HEIGHT * SPACE_WIDTH * CIRCLE_CHUNK_SIZE);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

//Round integer up to a given multiple
int roundUp(int number, int multiple){
    if (multiple == 0) return number;

    int remainder = number % multiple;
    
    if (remainder == 0) return number;

    return number + multiple - remainder;
}

void CudaRenderer::determineOwnership(int space_idx, int circ_chunk_idx){
    //TODO: Debug Me!
    //TODO: Move these calculations somewhere singlular
    
    //The dimension of the grid of region-squares needed to cover the image
    int rgrid_w = (image->width + REGION_WIDTH - 1) / REGION_WIDTH;
    int rgrid_h = (image->height + REGION_HEIGHT - 1) / REGION_HEIGHT;

    int downsc_w = (rgrid_w + SPACE_WIDTH - 1) / SPACE_WIDTH;
    int downsc_h = (rgrid_h + SPACE_HEIGHT - 1) / SPACE_HEIGHT;

    //Box index calculations
    //Indices for the top left corner (In the region space) +1 = one REGION over not pixels
    int top_y = (space_idx / downsc_w) * SPACE_HEIGHT;
    int top_x = (space_idx % downsc_w) * SPACE_WIDTH;

    //Convert indices to relative coordinates, ordered to pass to circleInBoxConservative
    float flHeight = static_cast<float>(image->height);
    float flWidth = static_cast<float>(image->width);

    float fly = static_cast<float>(top_y * REGION_HEIGHT);
    float flx = static_cast<float>(top_x * REGION_WIDTH);

    float rel_top_y = (flHeight - fly) / flHeight;
    float rel_top_x = (flx) / flWidth;

    //Indices for the bottom right
    int bot_x = top_x + (SPACE_WIDTH - 1);
    int bot_y = top_y + (SPACE_HEIGHT - 1);

    //printf("Top (y, x): (%f, %f)\n", rel_top_y, rel_top_x);
    //printf("Bottom (y, x): (%d, %d)\n", bot_y, bot_x);

    int circ_to_end = numCircles - (circ_chunk_idx * CIRCLE_CHUNK_SIZE);
    int circle_count = min(circ_to_end, CIRCLE_CHUNK_SIZE);

    
    //printf("Circle count: %d\n", circle_count);

    for (int i = 0; i < SPACE_WIDTH * SPACE_HEIGHT; i++){
        int regionX = i % SPACE_WIDTH;
        int regionY = i / SPACE_WIDTH;

        float region_L = static_cast<float>(regionX * REGION_WIDTH)/flWidth + rel_top_x;
        float y_sub = static_cast<float>(REGION_HEIGHT * regionY) / flHeight;
        float region_T = rel_top_y - y_sub;
        float region_B = region_T - static_cast<float>(REGION_HEIGHT) / flHeight;
        float region_R = region_L + static_cast<float>(REGION_WIDTH) / flWidth;

        
        //printf("Box vars: (L: %f, R: %f, T: %f, B: %f)\n", region_L, region_R, region_T, region_B);
    }

    kernelOwnRegion<<<1, SPACE_WIDTH * SPACE_HEIGHT>>>(cudaOwnershipSpace, circ_chunk_idx, circle_count, rel_top_y, rel_top_x, CIRCLE_CHUNK_SIZE, REGION_WIDTH, REGION_HEIGHT, SPACE_WIDTH, SPACE_HEIGHT);

    dim3 blockDim(256, 1);
    dim3 gridDim((SPACE_WIDTH * SPACE_HEIGHT * REGION_HEIGHT * REGION_WIDTH + blockDim.x - 1) / blockDim.x);

    int pix_top_x = top_x * REGION_WIDTH;
    int pix_top_y = top_y * REGION_HEIGHT;

    //printf ("Pix top (y, x): (%d, %d)\n", pix_top_x, pix_top_y);
    int space_size = SPACE_WIDTH * REGION_WIDTH;
    for (int index = 0; index < SPACE_WIDTH * SPACE_HEIGHT * REGION_HEIGHT * REGION_WIDTH; index++){    
        int space_x = (index % space_size);
        int space_y = (index / space_size);

        int pixel_x = space_x + pix_top_x;
        int pixel_y = space_y + pix_top_y;

        float rel_pix_x = static_cast<float>(pixel_x) / static_cast<float>(image->width);
        float rel_pix_y = static_cast<float>(image->height - pixel_y) / static_cast<float>(image->height);
        
        int pixelInImage = (pixel_x < image->width && pixel_y < image->height);
        int ownershipStart = /*CIRCLE_CHUNK_SIZE */ ((space_y / REGION_HEIGHT)*SPACE_WIDTH + (space_x / REGION_WIDTH)); //Go backwards and calculate your region index (inside space)

        
        
        if (!pixelInImage) continue;

        //printf("[%d] pixel (y, x): (%d, %d) [%f, %f]\n", index, pixel_y, pixel_x, rel_pix_y, rel_pix_x);
        //printf("ownership_idx: %d\n", ownershipStart);
    }
    //renderRegion(int* ownershipSpace, int circleCount, int circle_chunk_idx, int SPACE_WIDTH, int SPACE_HEIGHT, int REGION_WIDTH, int REGION_HEIGHT, int CIRCLE_CHUNK_SIZE, int top_x, int top_y)
    //printf("Circ chunk: %d\n", circ_chunk_idx);
    renderRegion<<<gridDim, blockDim>>>(cudaOwnershipSpace, circle_count, circ_chunk_idx, SPACE_WIDTH, SPACE_HEIGHT, REGION_WIDTH, REGION_HEIGHT, CIRCLE_CHUNK_SIZE, pix_top_x, pix_top_y);
    
}

void
CudaRenderer::render() {
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    //kernelRenderCircles<<<gridDim, blockDim>>>();

    //"Pads" the input so that the window can mathematically "fit" into the rendered image
    int pad_width = roundUp(image->width, SPACE_WIDTH * REGION_WIDTH);
    int pad_height = roundUp(image->height, SPACE_HEIGHT * REGION_HEIGHT);

    int pixels_left = pad_height * pad_width;//image->width * image->height;
    int space_area = REGION_WIDTH * REGION_HEIGHT * SPACE_WIDTH * SPACE_HEIGHT; //Number of pixels covered by one window pass
    int space_idx = 0;

    //printf("Space area: %d\n", space_area);


    while (pixels_left > 0){
        int circles_left = numCircles;
        int circ_chunk_idx = 0;

        while (circles_left > 0){
            determineOwnership(space_idx, circ_chunk_idx);
            circ_chunk_idx += 1;
            circles_left -= CIRCLE_CHUNK_SIZE;
        }

        //printf("Pixels left: %d\n", pixels_left);
        
        space_idx += 1;
        pixels_left -= space_area;
    }
    cudaDeviceSynchronize();
}
