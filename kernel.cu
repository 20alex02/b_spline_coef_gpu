__global__ void computeOnGPU(float *in, float *out, int cols, int rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) {
        return;
    }
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;
    float *myLine = out + (tid * cols);

    // copy input data
    for (int i = 0; i < cols; i++) {
        myLine[i] = in[i + (tid * cols)] * gain;
    }

    // compute 'sum'
    float sum = (myLine[0] + powf(z, cols) * myLine[cols - 1]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * cols - 2);
    float iz = 1.f / z;
    for (int j = 1; j < (cols - 1); ++j) {
        sum += (z2 + z1) * myLine[j];
        z1 *= z;
        z2 *= iz;
    }

    // iterate back and forth
    myLine[0] = sum * z / (1.f - powf(z, 2 * cols));
    for (int j = 1; j < cols; ++j) {
        myLine[j] += z * myLine[j - 1];
    }
    myLine[cols - 1] *= z / (z - 1.f);
    for (int j = cols - 2; 0 <= j; --j) {
        myLine[j] = z * (myLine[j + 1] - myLine[j]);
    }

    __syncthreads();
    float *myCol = out + tid;

    // multiply by gain (input data are already copied)
    for (int i = 0; i < y; i++) {
        myCol[i * x] *= gain;
    }

    // compute 'sum'
    sum = (myCol[0 * x] + powf(z, y) * myCol[(y - 1) * x]) * (1.f + z) / z;
    z1 = z;
    z2 = powf(z, 2 * y - 2);
    iz = 1.f / z;
    for (int j = 1; j < (y - 1); ++j) {
        sum += (z2 + z1) * myCol[j * x];
        z1 *= z;
        z2 *= iz;
    }

    // iterate back and forth
    myCol[0 * x] = sum * z / (1.f - powf(z, 2 * y));
    for (int j = 1; j < y; ++j) {
        myCol[j * x] += z * myCol[(j - 1) * x];
    }
    myCol[(y - 1) * x] *= z / (z - 1.f);
    for (int j = y - 2; 0 <= j; --j) {
        myCol[j * x] = z * (myCol[(j + 1) * x] - myCol[j * x]);
    }

}

void solveGPU(float *in, float *out, int x, int y) {
    computeOnGPU<<<1024, 1024>>>(in, out, x, y);
}

