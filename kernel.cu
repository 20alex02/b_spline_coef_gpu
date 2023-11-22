__global__ void computeRowsOnGPU(float *in, float *out, int cols, int rows) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;
    float *row = out + (rowIdx * cols);

    // copy input data
    for (int col = 0; col < cols; ++col) {
        row[col] = in[col + (rowIdx * cols)] * gain;
    }

    // compute 'sum'
    float sum = (row[0] + powf(z, cols) * row[cols - 1]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * cols - 2);
    float iz = 1.f / z;
    for (int j = 1; j < (cols - 1); ++j) {
        sum += (z2 + z1) * row[j];
        z1 *= z;
        z2 *= iz;
    }

    // iterate back and forth
    row[0] = sum * z / (1.f - powf(z, 2 * cols));
    for (int j = 1; j < cols; ++j) {
        row[j] += z * row[j - 1];
    }
    row[cols - 1] *= z / (z - 1.f);
    for (int j = cols - 2; 0 <= j; --j) {
        row[j] = z * (row[j + 1] - row[j]);
    }
}

__global__ void computeColsOnGPU(float *in, float *out, int cols, int rows) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;
    float *col = out + colIdx;

    // multiply by gain (input data are already copied)
    for (int row = 0; row < rows; ++row) {
        col[row * cols] *= gain;
    }

    // compute 'sum'
    float sum = (col[0 * cols] + powf(z, rows) * col[(rows - 1) * cols]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * rows - 2);
    float iz = 1.f / z;
    for (int row = 1; row < (rows - 1); ++row) {
        sum += (z2 + z1) * col[row * cols];
        z1 *= z;
        z2 *= iz;
    }

    // iterate back and forth
    col[0 * cols] = sum * z / (1.f - powf(z, 2 * rows));
    for (int row = 1; row < rows; ++row) {
        col[row * cols] += z * col[(row - 1) * cols];
    }
    col[(rows - 1) * cols] *= z / (z - 1.f);
    for (int row = rows - 2; 0 <= row; --row) {
        col[row * cols] = z * (col[(row + 1) * cols] - col[row * cols]);
    }
}

void solveGPU(float *in, float *out, int x, int y) {
    computeRowsOnGPU<<<y / 128, 128>>>(in, out, x, y);
    cudaDeviceSynchronize();
    computeColsOnGPU<<<x / 128, 128>>>(in, out, x, y);
    cudaDeviceSynchronize();
}

