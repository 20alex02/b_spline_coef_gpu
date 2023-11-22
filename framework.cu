#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define X 8192
#define Y 8192

//#define X 1024 // cols
//#define Y 1152 // rows

#include "kernel.cu"
#include "kernel_CPU.C"

int main(int argc, char **argv) {
    float *in, *out, *out_gpu;
    in = out = out_gpu = NULL;
    float *din, *dout;
    din = dout = NULL;

    // parse command line
    int device = 0;
    if (argc == 2)
        device = atoi(argv[1]);
    if (cudaSetDevice(device) != cudaSuccess) {
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);

    // create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate and set host memory
    in = (float *) malloc(X * Y * sizeof(in[0]));
    out = (float *) malloc(X * Y * sizeof(out[0]));
    out_gpu = (float *) malloc(X * Y * sizeof(out_gpu[0]));
    for (int i = 0; i < X * Y; i++) {
//        in[i] = i;
        in[i] = (float) rand() / float(RAND_MAX);
    }

    // allocate and set device memory
    if (cudaMalloc((void **) &din, X * Y * sizeof(in[0])) != cudaSuccess
        || cudaMalloc((void **) &dout, X * Y * sizeof(dout[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    cudaMemcpy(din, in, X * Y * sizeof(din[0]), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, X * Y * sizeof(din[0]));

    // solve on CPU
    printf("Solving on CPU...\n");
    cudaEventRecord(start, 0);
    solveCPU(in, out, X, Y);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU performance: %f megapixels/s\n", float(X) * float(Y) / time / 1e3f);

    // solve on GPU
    printf("Solving on GPU...\n");
    cudaEventRecord(start, 0);
    solveGPU(din, dout, X, Y);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU performance: %f megapixels/s\n", float(X) * float(Y) / time / 1e3f);

    // check GPU results
    cudaMemcpy(out_gpu, dout, X * Y * sizeof(dout[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < X * Y; i++) {
        if ((out_gpu[i] != out_gpu[i]) ||
            (fabsf(out[i] - out_gpu[i]) > 0.0001f)) { /* out_gpu[i] != out_gpu[i] catches NaN */
            fprintf(stderr, "Data mismatch at index %i: %f should be %f :-(\n", i, out_gpu[i], out[i]);
            goto cleanup;
        }
    }
    printf("Test OK.\n");

    cleanup:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (din) cudaFree(din);
    if (dout) cudaFree(dout);
    if (in) free(in);
    if (out) free(out);
    if (out_gpu) free(out_gpu);

    return 0;
}
