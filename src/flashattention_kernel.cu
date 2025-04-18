#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>

__global__ void forward_kernel(const float *Q, const float *K, const float *V, const int N, const int d,
                               const int num_tiles_K, const int num_tiles_Q, const int block_size_K, const int block_size_Q,
                               const float softmax_scale, float *l, float *m, float *O)
{
    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    // int qkv_offset_2 = (batch_idx * gridDim.y * N * d) + (head_idx / gridDim.y * (block_size_K)*d) + (head_idx * d);
    // int qkv_offset_2 = (batch_idx * gridDim.y * N * d) + (block_size_K * thread_idx * gridDim.y * d) + (head_idx * d);

    // Offset Calculation
    // Flattened 1D Data Shape: (B, N, nh, d)
    // gridDim shape = (x, y) = (B, nh)
    // 1) Batch -> (batch_idx * gridDim.y * N * d)
    // 2) N -> (block_size_K * thread_idx * gridDim.y * d)
    // 3) nh -> head_idx * d

    int lm_offset = (batch_idx * gridDim.y * N) + (head_idx * N);

    extern __shared__ float shared_memory[];
    int tile_size = block_size_K * d;
    float *Qi = shared_memory;
    float *Kj = &shared_memory[tile_size];
    float *Vj = &shared_memory[tile_size * 2];
    float *S = &shared_memory[tile_size * 3];

    for (int tile_idx_K = 0; tile_idx_K < num_tiles_K; tile_idx_K++)
    {
        for (int x = 0; x < d; x++)
        {

            Kj[(thread_idx * d) + x] = K[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
            Vj[(thread_idx * d) + x] = V[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
            // printf("Head %d: Key[%d] = %f, Value[%d] = %f, qkv_offset = %d\n", head_idx, qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x, Kj[(thread_idx * d) + x],
            // qkv_offset + (tile_size * tile_idx_K) + (thread_idx * gridDim.y * d) + x, Vj[(thread_idx * d) + x], qkv_offset);
            // printf("ADDITIONAL ON TOP OF OFFSET = %d\n",   (thread_idx * gridDim.y * d));

            // printf("K and V access: %d\n", qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x);
        }
        __syncthreads();

        for (int tile_idx_Q = 0; tile_idx_Q < num_tiles_Q; tile_idx_Q++)
        {

            for (int x = 0; x < d; x++)
            {
                Qi[(thread_idx * d) + x] = Q[qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x];
                // printf("Head %d: Query[%d] = %f\n", head_idx, qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x,  Qi[(thread_idx * d) + x]);

                // printf("Q access: %d\n", qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x);
            }

            float prev_m = m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            float prev_l = l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            // printf("m and l access: %d\n", lm_offset + (block_size_Q * tile_idx_Q) + thread_idx);

            float row_max = -INFINITY;
            for (int y = 0; y < block_size_K; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(thread_idx * d) + x] * Kj[(y * d) + x];
                    // printf("For index: %d, Head %d: Query[%d] = %f, Kj[%d] = %f\n", (block_size_K * thread_idx) + y, head_idx, (thread_idx * d) + x,  Qi[(thread_idx * d) + x], (y * d) + x,  Kj[(y * d) + x]);
                }
                // printf("sum = %f\n", sum);
                sum *= softmax_scale;
                // printf("sum after scaling = %f\n", sum);
                S[(block_size_K * thread_idx) + y] = sum;
                // printf("Head %d: S[%d] = %f\n",  head_idx, (block_size_K * thread_idx) + y, S[(block_size_K * thread_idx) + y] );

                if (sum > row_max)
                    row_max = sum;
            }

            float row_sum = 0;
            for (int y = 0; y < block_size_K; y++)
            {
                S[(block_size_K * thread_idx) + y] = __expf(S[(block_size_K * thread_idx) + y] - row_max);
                row_sum += S[(block_size_K * thread_idx) + y];
            }

            float new_m = fmax(prev_m, row_max);
            float new_l = (__expf(prev_m - new_m) * prev_l) + (__expf(row_max - new_m) * row_sum);

            for (int x = 0; x < d; x++)
            {
                float weighted_sum = 0;
                for (int y = 0; y < block_size_K; y++)
                {
                    weighted_sum += S[(block_size_K * thread_idx) + y] * Vj[(y * d) + x];
                }
                __syncthreads();
                int index = qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x;
                O[index] = (1 / new_l) *
                           ((prev_l * __expf(prev_m - new_m) * O[index]) +
                            (__expf(row_max - new_m) * weighted_sum));
                // printf("O access after update: %d\n", index);
            }

            m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_m;
            l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_l;
        }
        __syncthreads();
    }
}

extern "C"
{
    void launch_flashattention_forward(float *Q, float *K, float *V, float *O, float *l, float *m, int batch_size, int num_heads, int N, int d)
    {

        int block_size_K, block_size_Q;

        // TODO: Optimise memory usage for lower d, low hanging fruit for those that can be 4x

        // Assume SRAM >= 32768
        if (d>2048) return;

        block_size_K = min(min(N, 2048/d), 64); block_size_Q = min(min(N, 2048/d), 64);


        const int num_tiles_K = ceil((float)N / block_size_K);
        const int num_tiles_Q = ceil((float)N / block_size_Q);

        const float softmax_scale = 1.0 / sqrt(d);

        float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
        const int Q_size = batch_size * num_heads * N * d;
        const int l_size = batch_size * num_heads * N;

        cudaMalloc((void **)&d_Q, Q_size * sizeof(float));
        cudaMalloc((void **)&d_K, Q_size * sizeof(float));
        cudaMalloc((void **)&d_V, Q_size * sizeof(float));
        cudaMalloc((void **)&d_O, Q_size * sizeof(float));
        cudaMalloc((void **)&d_l, l_size * sizeof(float));
        cudaMalloc((void **)&d_m, l_size * sizeof(float));

        cudaMemcpy(d_Q, Q, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_O, O, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l, l, l_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, l_size * sizeof(float), cudaMemcpyHostToDevice);

        const int shared_mem_size = (2 * block_size_K * d * sizeof(float)) + (block_size_Q * d * sizeof(float)) + (block_size_K * block_size_Q * sizeof(float)) + (2 * block_size_K * sizeof(float));

        // Uncomment if you want to see SRAM requested
        // printf("d: %d | num_head: %d | N: %d | Block Size Q: %d | Block Size K: %d | Max shared memory: %d, requested shared memory: %d \n", d, num_heads, N, block_size_Q, block_size_K, max_sram_size, shared_mem_size);

        // // This should never run if your code checks for too large d
        // if (shared_mem_size >= max_sram_size) {
        //     // fprintf(stderr, "Too much SRAM requested: %f\n", shared_mem_size);
        //     // exit(EXIT_FAILURE);
        //     // printf("Too much SRAM requested: %d\n", shared_mem_size);
        //     return;
        // }

        dim3 grid_dim(batch_size, num_heads);
        dim3 block_dim(block_size_K);

        // Launch the kernel
        forward_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            d_Q, d_K, d_V, N, d, num_tiles_K, num_tiles_Q, block_size_K, block_size_Q, softmax_scale, d_l, d_m, d_O);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error in kernel launch: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(O, d_O, Q_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_l);
        cudaFree(d_m);
    }
}

__global__ void forward_kernel_causal(const float *Q, const float *K, const float *V, const float *mask, const int N, const int d,
                                      const int num_tiles_K, const int num_tiles_Q, const int block_size_K, const int block_size_Q,
                                      const float softmax_scale, float *l, float *m, float *O)
{

    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    int lm_offset = (batch_idx * gridDim.y * N) + (head_idx * N);

    extern __shared__ float shared_memory[];
    int tile_size = block_size_K * d;
    float *Qi = shared_memory;
    float *Kj = &shared_memory[tile_size];
    float *Vj = &shared_memory[tile_size * 2];
    float *S = &shared_memory[tile_size * 3];

    for (int tile_idx_K = 0; tile_idx_K < num_tiles_K; tile_idx_K++)
    {
        // Load Kj, Vj into SRAM
        for (int x = 0; x < d; x++)
        {
            Kj[(thread_idx * d) + x] = K[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
            Vj[(thread_idx * d) + x] = V[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
        }
        __syncthreads();

        for (int tile_idx_Q = 0; tile_idx_Q < num_tiles_Q; tile_idx_Q++)
        {
            // Load Qi into SRAM
            for (int x = 0; x < d; x++)
            {
                Qi[(thread_idx * d) + x] = Q[qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x];
            }

            // Load li, mi into SRAM
            float prev_m = m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            float prev_l = l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];

            // Compute Sij & rowmax
            float row_max = -INFINITY;
            for (int y = 0; y < block_size_K; y++)
            {

                if (mask[(tile_idx_Q * block_size_Q + thread_idx) * N + (tile_idx_K * block_size_K) + y] == 0) {
                    float sum = 0;
                    for (int x = 0; x < d; x++)
                    {
                        sum += Qi[(thread_idx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(block_size_K * thread_idx) + y] = sum;

                    if (sum > row_max)
                        row_max = sum;
                }
            }

            // Compute rowsum
            float row_sum = 0;
            for (int y = 0; y < block_size_K; y++)
            {
                if (mask[(tile_idx_Q * block_size_Q + thread_idx) * N + (tile_idx_K * block_size_K) + y] == 0) {
                    S[(block_size_K * thread_idx) + y] = __expf(S[(block_size_K * thread_idx) + y] - row_max);
                    row_sum += S[(block_size_K * thread_idx) + y];
                }
            }

            // Compute m_new, l_new
            float new_m = fmax(prev_m, row_max);
            float new_l = (__expf(prev_m - new_m) * prev_l) + (__expf(row_max - new_m) * row_sum);

            // Write O to HBM
            for (int x = 0; x < d; x++)
            {
                float weighted_sum = 0;
                for (int y = 0; y < block_size_K; y++)
                {
                    if (mask[(tile_idx_Q * block_size_Q + thread_idx) * N + (tile_idx_K * block_size_K) + y] == 0) {
                        weighted_sum += S[(block_size_K * thread_idx) + y] * Vj[(y * d) + x];
                    }
                }
                __syncthreads();
                int index = qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x;
                O[index] = (1 / new_l) *
                        ((prev_l * __expf(prev_m - new_m) * O[index]) +
                            (__expf(row_max - new_m) * weighted_sum));
            }

            // Write li, mi to HBM
            m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_m;
            l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_l;
        }
        __syncthreads();
    }
}

extern "C"
{
    void launch_flashattention_forward_causal(float *Q, float *K, float *V, float *O, float *l, float *m, float *mask, int batch_size, int num_heads, int N, int d)
    {

        int block_size_K, block_size_Q;

        if (d>2048) return;

        block_size_K = min(min(N, 2048/d), 64); block_size_Q = min(min(N, 2048/d), 64);

        const int num_tiles_K = ceil((float)N / block_size_K);
        const int num_tiles_Q = ceil((float)N / block_size_Q);

        const float softmax_scale = 1.0 / sqrt(d);

        float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m, *d_mask;
        const int Q_size = batch_size * num_heads * N * d;
        const int l_size = batch_size * num_heads * N;
        const int mask_size = N * N;

        cudaMalloc((void **)&d_Q, Q_size * sizeof(float));
        cudaMalloc((void **)&d_K, Q_size * sizeof(float));
        cudaMalloc((void **)&d_V, Q_size * sizeof(float));
        cudaMalloc((void **)&d_O, Q_size * sizeof(float));
        cudaMalloc((void **)&d_l, l_size * sizeof(float));
        cudaMalloc((void **)&d_m, l_size * sizeof(float));
        cudaMalloc((void **)&d_mask, mask_size * sizeof(float));


        cudaMemcpy(d_Q, Q, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l, l, l_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, l_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

        const int shared_mem_size = (2 * block_size_K * d * sizeof(float)) + (block_size_Q * d * sizeof(float)) + (block_size_K * block_size_Q * sizeof(float)) + (2 * block_size_K * sizeof(float));

        dim3 grid_dim(batch_size, num_heads);
        dim3 block_dim(block_size_K);

        // Launch the kernel
        forward_kernel_causal<<<grid_dim, block_dim, shared_mem_size>>>(
            d_Q, d_K, d_V, d_mask, N, d, num_tiles_K, num_tiles_Q, block_size_K, block_size_Q, softmax_scale, d_l, d_m, d_O);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error in kernel launch: %s\n", cudaGetErrorString(err));
            // exit(EXIT_FAILURE);
            return;
        }

        cudaMemcpy(O, d_O, Q_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_l);
        cudaFree(d_m);
    }
}