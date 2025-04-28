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


    // Offset Calculation
    // Flattened 1D Data (Q, K, V) Shape: (B, nh, N, d)
    // Flattened 1D Data (l, m) Shape: (B, nh, N)
    // gridDim shape = (x, y) = (B, nh)
    // 1) Batch -> (batch_idx * gridDim.y * N * d)
    // 2) nh -> (head_idx * N * d)
    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    int lm_offset = (batch_idx * gridDim.y * N) + (head_idx * N);

    // Allocate SRAM
    extern __shared__ float shared_memory[];
    int tile_size = block_size_K * d;
    float *Qi = shared_memory;
    float *Kj = &shared_memory[tile_size];
    float *Vj = &shared_memory[tile_size * 2];
    float *S = &shared_memory[tile_size * 3];


    // Loop through num_tiles_K
    for (int tile_idx_K = 0; tile_idx_K < num_tiles_K; tile_idx_K++)
    {
        // Load Kj and Vj into SRAM for all threads in the thread block to use
        for (int x = 0; x < d; x++)
        {

            Kj[(thread_idx * d) + x] = K[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
            Vj[(thread_idx * d) + x] = V[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
        }

        __syncthreads();

        // Loop through num_tiles_Q
        for (int tile_idx_Q = 0; tile_idx_Q < num_tiles_Q; tile_idx_Q++)
        {
            // Load Qi, li, mi into SRAM
            for (int x = 0; x < d; x++)
            {
                Qi[(thread_idx * d) + x] = Q[qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x];
                
            }

            float prev_m = m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            float prev_l = l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];

            // Compute Sij and mij values for the relevant Qi and Kj values in the block
            float row_max = -INFINITY;
            for (int y = 0; y < block_size_K; y++)
            {
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

            // Compute Pij and lij values for the relevant Qi and Kj values in the block
            float row_sum = 0;
            for (int y = 0; y < block_size_K; y++)
            {
                S[(block_size_K * thread_idx) + y] = __expf(S[(block_size_K * thread_idx) + y] - row_max);
                row_sum += S[(block_size_K * thread_idx) + y];
            }

            // Compute new mi and li values
            float new_m = fmax(prev_m, row_max);
            float new_l = (__expf(prev_m - new_m) * prev_l) + (__expf(row_max - new_m) * row_sum);

            // Compute Oi values and write to HBM
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
            }

            // Write mi and li to HBM
            m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_m;
            l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_l;
        }
        __syncthreads();
    }
}


__global__ void backward_kernel(const float *Q, const float *K, const float *V, const int N, const int d,
                               const int num_tiles_K, const int num_tiles_Q, const int block_size_K, const int block_size_Q,
                               const float softmax_scale, float *l, float *m, float *O,
                               float* dQ, float* dK, float* dV, float* dO
                               )
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
   
    float* dKj = &shared_memory[tile_size * 3];
    float* dVj = &shared_memory[tile_size * 4];
    float* Oi = &shared_memory[tile_size * 5];
    float* dOi = &shared_memory[tile_size * 6];

    float *S = &shared_memory[tile_size * 7];
    float *dS = &shared_memory[tile_size * 7 + block_size_K * block_size_Q];

    // Loop through num_tiles_K
    for (int tile_idx_K = 0; tile_idx_K < num_tiles_K; tile_idx_K++)
    {   
        // Load Kj and Vj into SRAM for all threads in the thread block to use
        // Initialize gradients of dK and dV to 0.0
        for (int x = 0; x < d; x++)
        {
            int hbm_idx = qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x;
            int sram_idx = (thread_idx * d) + x;

            Kj[sram_idx] = K[hbm_idx];
            Vj[sram_idx] = V[hbm_idx];
            dKj[sram_idx] = 0.0f;
            dVj[sram_idx] = 0.0f;
        }
        
        // Loop through num_tiles_Q
        for (int tile_idx_Q = 0; tile_idx_Q < num_tiles_Q; tile_idx_Q++)
        {   
            // Load Qi, Oi, dO into SRAM
            for (int x = 0; x < d; x++)
            {
                int hbm_idx = qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x;
                int sram_idx = (thread_idx * d) + x;

                Qi[sram_idx] = Q[hbm_idx];
                Oi[sram_idx] = O[hbm_idx];
                dOi[sram_idx] = dO[hbm_idx];

            }

            //Load l and m that was calculated into the SRAM
            float m_val = m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            float l_val = l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];

            // Compute Sij using Q and K and softmax scale
            for (int y = 0; y < block_size_K; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(thread_idx * d) + x] * Kj[(y * d) + x];
                }

                sum *= softmax_scale;
                S[(block_size_K * thread_idx) + y] = sum;
            }

            //Compute Pij and load into SRAM
            for (int y = 0; y < block_size_K; y++)
            {   
                S[(block_size_K * thread_idx) + y] = (1 / l_val) * __expf(S[(block_size_K * thread_idx) + y] - m_val);
            }
            __syncthreads();

            //Use Pij and dOi values to calculate dV and store it in SRAM
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < block_size_K; y++) {
                    sum += S[(block_size_K * y) + thread_idx] * dOi[(thread_idx * d) + x];
                }
                dVj[(thread_idx * d) + x] += sum;
            }

            // Use previously calculated dOi values with V to calculate dS
            for (int y = 0; y < block_size_K; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += dOi[(thread_idx * d) + x] * Vj[(y * d) + x];
                }
                dS[(block_size_K * thread_idx) + y] = sum;
            }

            // Accumulate value of Di using Oi and dOi for future calculations
            float Di = 0;
            for (int x = 0; x < d; x++) {
                Di += dOi[(thread_idx * d) + x] * Oi[(thread_idx * d) + x];
            }

            //Calculate dS using S, dS and Di
            for (int y = 0; y < block_size_K; ++y) {
                dS[(block_size_K * thread_idx) + y] = S[(block_size_K * thread_idx) + y] * (dS[(block_size_K * thread_idx) + y] - Di);
            }

            //Calculate dQ using calculated dS and K
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < block_size_K; y++) {
                    sum += dS[(block_size_K * thread_idx) + y] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                dQ[qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x] += sum;
            }

            //Calculate dK using calculated dS and Q
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < block_size_K; y++) {
                    sum += dS[(block_size_K * y) + thread_idx] * Qi[(y * d) + x];
                }
                sum *= softmax_scale;
                dKj[(thread_idx * d) + x] += sum;
            }
        }

        // Load dK and dV vvalues into HBM after calculation
        for (int x = 0; x < d; x++) {
            dK[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x] = dKj[(thread_idx * d) + x];
            dV[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x] = dVj[(thread_idx * d) + x];
        }
    
    }
}

extern "C"
{
    void launch_flashattention_forward(float *Q, float *K, float *V, float *O, float *l, float *m, int batch_size, int num_heads, int N, int d)
    {

        int block_size_K, block_size_Q;

        // Assume SRAM >= 32768, works for V100, A100 and H100
        // returns if block size < 1, unable to fit into SRAM
        if (d>2048) return;

        // Compute block size dynamically to maximize block size and maximise parallelism
        block_size_K = min(min(N, 2048/d), 64); block_size_Q = min(min(N, 2048/d), 64);
        while(N % block_size_K != 0){
            block_size_K/=2;
            block_size_Q/=2;
        }

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
        cudaMemcpy(m, d_m, l_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(l, d_l, l_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_l);
        cudaFree(d_m);
    }
}

int find_max_block_size(int d, int max_sram_size) {
    int B = 1;
    int best_B = 1;

    while (true) {
        size_t shared_mem_size = sizeof(float) * (
            (7 * B * d) +         // Qi, Kj, Vj, dKj, dVj, Oi, dOi
            (2 * B * B) +         // S, dS
            (2 * B)               // l, m
        );

        if (shared_mem_size > max_sram_size)
            break;

        best_B = B;
        B *= 2;
        if (B > 64) break;
    }

    return best_B;
}

extern "C"
{
    void launch_flashattention_backward(float *Q, float *K, float *V, float *O, float *dQ,  float *dK,  float *dV, float *dO,  float *l, float *m, int batch_size, int num_heads, int N, int d)
    {

        int block_size_K, block_size_Q;

        // Assume SRAM >= 32768, works for V100, A100 and H100
        // returns if block size < 1, unable to fit into SRAM
        if (d>2048) return;

        // Compute block size dynamically to maximize block size and maximise parallelism
        block_size_K = min(find_max_block_size(d, max_sram_size), N); block_size_Q = block_size_K;
        
        const int num_tiles_K = ceil((float)N / block_size_K);
        const int num_tiles_Q = ceil((float)N / block_size_Q);

        const float softmax_scale = 1.0 / sqrt(d);

        float *d_Q, *d_dQ, *d_K, *d_dK, *d_V, *d_dV, *d_O, *d_dO, *d_l, *d_m;
        const int Q_size = batch_size * num_heads * N * d;
        const int l_size = batch_size * num_heads * N;

        cudaMalloc((void **)&d_Q, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dQ, Q_size * sizeof(float));

        cudaMalloc((void **)&d_K, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dK, Q_size * sizeof(float));

        cudaMalloc((void **)&d_V, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dV, Q_size * sizeof(float));

        cudaMalloc((void **)&d_O, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dO, Q_size * sizeof(float));
        
        cudaMalloc((void **)&d_l, l_size * sizeof(float));
        cudaMalloc((void **)&d_m, l_size * sizeof(float));

        cudaMemcpy(d_Q, Q, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dQ, dQ, Q_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_K, K, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dK, dK, Q_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_V, V, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dV, dV, Q_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_O, O, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dO, dO, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_l, l, l_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, l_size * sizeof(float), cudaMemcpyHostToDevice);

        const int shared_mem_size = (4 * block_size_K * d * sizeof(float)) + (3 * block_size_Q * d * sizeof(float)) + (2 * block_size_K * block_size_Q * sizeof(float)) + (2 * block_size_K * sizeof(float));;

        dim3 grid_dim(batch_size, num_heads);
        dim3 block_dim(block_size_K);

        // Launch the kernel
        backward_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
            d_Q, d_K, d_V, N, d, num_tiles_K, num_tiles_Q, block_size_K, block_size_Q, softmax_scale, d_l, d_m, d_O, d_dQ, d_dK, d_dV, d_dO);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error in kernel launch: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(dO, d_dO, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dQ, d_dQ, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dK, d_dK, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dV, d_dV, Q_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_Q);
        cudaFree(d_dQ);
        cudaFree(d_K);
        cudaFree(d_dK);
        cudaFree(d_V);
        cudaFree(d_dV);
        cudaFree(d_O);
        cudaFree(d_dO);
        cudaFree(d_l);
        cudaFree(d_m);
    }
}

__global__ void forward_kernel_causal(const float *Q, const float *K, const float *V, const int N, const int d,
                                      const int num_tiles_K, const int num_tiles_Q, const int block_size_K, const int block_size_Q,
                                      const float softmax_scale, float *l, float *m, float *O)
{

    int thread_idx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    // Offset Calculation
    // Flattened 1D Data (Q, K, V) Shape: (B, nh, N, d)
    // Flattened 1D Data (l, m) Shape: (B, nh, N)
    // gridDim shape = (x, y) = (B, nh)
    // 1) Batch -> (batch_idx * gridDim.y * N * d)
    // 2) nh -> (head_idx * N * d)
    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    int lm_offset = (batch_idx * gridDim.y * N) + (head_idx * N);

    // Allocate SRAM
    extern __shared__ float shared_memory[];
    int tile_size = block_size_K * d;
    float *Qi = shared_memory;
    float *Kj = &shared_memory[tile_size];
    float *Vj = &shared_memory[tile_size * 2];
    float *S = &shared_memory[tile_size * 3];


    for (int tile_idx_K = 0; tile_idx_K < num_tiles_K; tile_idx_K++)
    {
        // Load Kj and Vj into SRAM for all threads in the thread block to use
        for (int x = 0; x < d; x++)
        {
            Kj[(thread_idx * d) + x] = K[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
            Vj[(thread_idx * d) + x] = V[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x];
        }
        __syncthreads();

        // Apply coarse-grained optimisation (Block-level)
        if (tile_idx_Q < tile_idx_K) continue;
        
        for (int tile_idx_Q = tile_idx_K; tile_idx_Q < num_tiles_Q; tile_idx_Q++)
        {
            
            // Load Qi, li, mi into SRAM
            for (int x = 0; x < d; x++)
            {
                Qi[(thread_idx * d) + x] = Q[qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x];
            }

            float prev_m = m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            float prev_l = l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];

            // Compute Sij and mij values for the relevant Qi and Kj values in the block
            float row_max = -INFINITY;
            for (int y = 0; y < block_size_K; y++)
            {
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

            // Compute Pij and lij values for the relevant Qi and Kj values in the block
            // Apply fine-grained optimisation (Element-level), skip unnecessary computations
            float row_sum = 0;
            for (int y = 0; y < block_size_K; y++)
            {
                if ((tile_idx_Q * block_size_Q + thread_idx) >= (tile_idx_K * block_size_K) + y) {
                    S[(block_size_K * thread_idx) + y] = __expf(S[(block_size_K * thread_idx) + y] - row_max);
                } else {
                    S[(block_size_K * thread_idx) + y] = 0.0;
                }

                row_sum += S[(block_size_K * thread_idx) + y];
            }

            // Compute new mi and li values
            float new_m = fmax(prev_m, row_max);
            float new_l = (__expf(prev_m - new_m) * prev_l) + (__expf(row_max - new_m) * row_sum);

            // Compute Oi values and write to HBM
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
            }

            // Write mi and li to HBM
            m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_m;
            l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx] = new_l;
        }
        __syncthreads();
    }
}

__global__ void backward_kernel_causal(const float *Q, const float *K, const float *V, const int N, const int d,
                               const int num_tiles_K, const int num_tiles_Q, const int block_size_K, const int block_size_Q,
                               const float softmax_scale, float *l, float *m, float *O,
                               float* dQ, float* dK, float* dV, float* dO
                               )
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
   
    float* dKj = &shared_memory[tile_size * 3];
    float* dVj = &shared_memory[tile_size * 4];
    float* Oi = &shared_memory[tile_size * 5];
    float* dOi = &shared_memory[tile_size * 6];

    float *S = &shared_memory[tile_size * 7];
    float *dS = &shared_memory[tile_size * 7 + block_size_K * block_size_Q];

    // Loop through num_tiles_K
    for (int tile_idx_K = 0; tile_idx_K < num_tiles_K; tile_idx_K++)
    {
        // Load Kj and Vj into SRAM for all threads in the thread block to use
        // Initialize gradients of dK and dV to 0.0
        for (int x = 0; x < d; x++)
        {
            int hbm_idx = qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x;
            int sram_idx = (thread_idx * d) + x;

            Kj[sram_idx] = K[hbm_idx];
            Vj[sram_idx] = V[hbm_idx];
            dKj[sram_idx] = 0.0f;
            dVj[sram_idx] = 0.0f;
        }

        // Loop through num_tiles_Q
        for (int tile_idx_Q = 0; tile_idx_Q < num_tiles_Q; tile_idx_Q++)
        {
            // Load Qi, Oi, dO into SRAM
            for (int x = 0; x < d; x++)
            {
                int hbm_idx = qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x;
                int sram_idx = (thread_idx * d) + x;

                Qi[sram_idx] = Q[hbm_idx];
                Oi[sram_idx] = O[hbm_idx];
                dOi[sram_idx] = dO[hbm_idx];
            }

            //Load l and m that was calculated into the SRAM
            float m_val = m[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            float l_val = l[lm_offset + (block_size_Q * tile_idx_Q) + thread_idx];
            
            // Compute Sij using Q and K and softmax scale
            for (int y = 0; y < block_size_K; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(thread_idx * d) + x] * Kj[(y * d) + x];
                }

                sum *= softmax_scale;
                S[(block_size_K * thread_idx) + y] = sum;
            }

            //Compute Pij and load into SRAM. Skip certain calculation and account for causal computation
            for (int y = 0; y < block_size_K; y++)
            {   
                if ((tile_idx_Q * block_size_Q + thread_idx) >= (tile_idx_K * block_size_K) + y) {
                    S[(block_size_K * thread_idx) + y] = (1 / l_val) * __expf(S[(block_size_K * thread_idx) + y] - m_val);
                } else {
                    S[(block_size_K * thread_idx) + y] = 0.0;
                }
            }
            __syncthreads();    
            
            //Use Pij and dOi values to calculate dV and store it in SRAM
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < block_size_K; y++) {
                    sum += S[(block_size_K * y) + thread_idx] * dOi[(thread_idx * d) + x];
                }
                dVj[(thread_idx * d) + x] += sum;
            }

            // Use previously calculated dOi values with V to calculate dS
            for (int y = 0; y < block_size_K; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += dOi[(thread_idx * d) + x] * Vj[(y * d) + x];
                }
                dS[(block_size_K * thread_idx) + y] = sum;
            }

            // Accumulate value of Di using Oi and dOi for future calculations
            float Di = 0;
            for (int x = 0; x < d; x++) {
                Di += dOi[(thread_idx * d) + x] * Oi[(thread_idx * d) + x];
            }

            //Calculate dS using S, dS and Di
            for (int y = 0; y < block_size_K; ++y) {
                dS[(block_size_K * thread_idx) + y] = S[(block_size_K * thread_idx) + y] * (dS[(block_size_K * thread_idx) + y] - Di);
            }

            //Calculate dQ using calculated dS and K
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < block_size_K; y++) {
                    sum += dS[(block_size_K * thread_idx) + y] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                dQ[qkv_offset + (tile_size * tile_idx_Q) + (thread_idx * d) + x] += sum;
            }

            //Calculate dK using calculated dS and Q
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < block_size_K; y++) {
                    sum += dS[(block_size_K * y) + thread_idx] * Qi[(y * d) + x];
                }
                sum *= softmax_scale;
                dKj[(thread_idx * d) + x] += sum;
            }
        }

        // Load dK and dV vvalues into HBM after calculation
        for (int x = 0; x < d; x++) {
            dK[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x] = dKj[(thread_idx * d) + x];
            dV[qkv_offset + (tile_size * tile_idx_K) + (thread_idx * d) + x] = dVj[(thread_idx * d) + x];
        }
    
    }
}

extern "C"
{
    void launch_flashattention_forward_causal(float *Q, float *K, float *V, float *O, float *l, float *m, int batch_size, int num_heads, int N, int d)
    {

        int block_size_K, block_size_Q;

        // Assume SRAM >= 32768, works for V100, A100 and H100
        // returns if block size < 1, unable to fit into SRAM
        if (d>2048) return;

        // Compute block size dynamically to maximize block size and maximise parallelism
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
        cudaMemcpy(d_l, l, l_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, l_size * sizeof(float), cudaMemcpyHostToDevice);

        const int shared_mem_size = (2 * block_size_K * d * sizeof(float)) + (block_size_Q * d * sizeof(float)) + (block_size_K * block_size_Q * sizeof(float)) + (2 * block_size_K * sizeof(float));

        dim3 grid_dim(batch_size, num_heads);
        dim3 block_dim(block_size_K);

        // Launch the kernel
        forward_kernel_causal<<<grid_dim, block_dim, shared_mem_size>>>(
            d_Q, d_K, d_V, N, d, num_tiles_K, num_tiles_Q, block_size_K, block_size_Q, softmax_scale, d_l, d_m, d_O);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error in kernel launch: %s\n", cudaGetErrorString(err));
            return;
        }

        cudaMemcpy(O, d_O, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(m, d_m, l_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(l, d_l, l_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_l);
        cudaFree(d_m);
    }
}

extern "C"
{
    void launch_flashattention_backward_causal(float *Q, float *K, float *V, float *O, float *dQ,  float *dK,  float *dV, float *dO,  float *l, float *m, int batch_size, int num_heads, int N, int d)
    {

        int block_size_K, block_size_Q;

        // Assume SRAM >= 32768, works for V100, A100 and H100
        // returns if block size < 1, unable to fit into SRAM
        if (d>2048) return;

        // Compute block size dynamically to maximize block size and maximise parallelism
        block_size_K = min(find_max_block_size(d, max_sram_size), N); block_size_Q = block_size_K;

        const int num_tiles_K = ceil((float)N / block_size_K);
        const int num_tiles_Q = ceil((float)N / block_size_Q);

        const float softmax_scale = 1.0 / sqrt(d);

        float *d_Q, *d_dQ, *d_K, *d_dK, *d_V, *d_dV, *d_O, *d_dO, *d_l, *d_m;
        const int Q_size = batch_size * num_heads * N * d;
        const int l_size = batch_size * num_heads * N;

        cudaMalloc((void **)&d_Q, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dQ, Q_size * sizeof(float));

        cudaMalloc((void **)&d_K, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dK, Q_size * sizeof(float));

        cudaMalloc((void **)&d_V, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dV, Q_size * sizeof(float));

        cudaMalloc((void **)&d_O, Q_size * sizeof(float));
        cudaMalloc((void **)&d_dO, Q_size * sizeof(float));
        
        cudaMalloc((void **)&d_l, l_size * sizeof(float));
        cudaMalloc((void **)&d_m, l_size * sizeof(float));

        cudaMemcpy(d_Q, Q, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dQ, dQ, Q_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_K, K, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dK, dK, Q_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_V, V, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dV, dV, Q_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_O, O, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dO, dO, Q_size * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_l, l, l_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, l_size * sizeof(float), cudaMemcpyHostToDevice);

        const int shared_mem_size = (4 * block_size_K * d * sizeof(float)) + (3 * block_size_Q * d * sizeof(float)) + (2 * block_size_K * block_size_Q * sizeof(float)) + (2 * block_size_K * sizeof(float));;

        dim3 grid_dim(batch_size, num_heads);
        dim3 block_dim(block_size_K);

        // Launch the kernel
        backward_kernel_causal<<<grid_dim, block_dim, shared_mem_size>>>(
            d_Q, d_K, d_V, N, d, num_tiles_K, num_tiles_Q, block_size_K, block_size_Q, softmax_scale, d_l, d_m, d_O, d_dQ, d_dK, d_dV, d_dO);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error in kernel launch: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(dO, d_dO, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dQ, d_dQ, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dK, d_dK, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dV, d_dV, Q_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_Q);
        cudaFree(d_dQ);
        cudaFree(d_K);
        cudaFree(d_dK);
        cudaFree(d_V);
        cudaFree(d_dV);
        cudaFree(d_O);
        cudaFree(d_dO);
        cudaFree(d_l);
        cudaFree(d_m);
    }
}