__global__ void computeRowsOnGPU(float *out) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;
    float *row = out + (rowIdx * X);

    // copy input data
    for (int col = 0; col < X; ++col) {
        row[col] *= gain;
    }

    // compute 'sum'
    float sum = (row[0] + powf(z, X) * row[X - 1]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * X - 2);
    float iz = 1.f / z;
    for (int j = 1; j < (X - 1); ++j) {
        sum += (z2 + z1) * row[j];
        z1 *= z;
        z2 *= iz;
    }

    // iterate back and forth
    row[0] = sum * z / (1.f - powf(z, 2 * X));
    for (int j = 1; j < X; ++j) {
        row[j] += z * row[j - 1];
    }
    row[X - 1] *= z / (z - 1.f);
    for (int j = X - 2; 0 <= j; --j) {
        row[j] = z * (row[j + 1] - row[j]);
    }
}

__global__ void computeColsOnGPU(float *out) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;
    float *col = out + colIdx;

    // multiply by gain (input data are already copied)
    for (int row = 0; row < Y; ++row) {
        col[row * X] *= gain;
    }

    // compute 'sum'
    float sum = (col[0 * X] + powf(z, Y) * col[(Y - 1) * X]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * Y - 2);
    float iz = 1.f / z;
    for (int row = 1; row < (Y - 1); ++row) {
        sum += (z2 + z1) * col[row * X];
        z1 *= z;
        z2 *= iz;
    }

    // iterate back and forth
    col[0 * X] = sum * z / (1.f - powf(z, 2 * Y));
    for (int row = 1; row < Y; ++row) {
        col[row * X] += z * col[(row - 1) * X];
    }
    col[(Y - 1) * X] *= z / (z - 1.f);
    for (int row = Y - 2; 0 <= row; --row) {
        col[row * X] = z * (col[(row + 1) * X] - col[row * X]);
    }
}

void solveGPU(float **in, float **out, int x, int y) {
    float *tmp = *out;
    *out = *in;
    *in = tmp;
    computeRowsOnGPU<<<y / 128, 128>>>(*out);
    cudaDeviceSynchronize();
    computeColsOnGPU<<<x / 128, 128>>>(*out);
    cudaDeviceSynchronize();
}

