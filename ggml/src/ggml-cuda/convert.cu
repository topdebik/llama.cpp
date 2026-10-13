#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d * (q[l] & 0xF) + dm);
        y[l+16] = ggml_cuda_cast<dst_t>(d * (q[l] >>  4) + dm);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = ggml_cuda_cast<dst_t>(d.x * (q[l] >>  4) + d.y);
    }
}

#if !defined(GGML_USE_HIP)
template <typename dst_t>
static __device__
__forceinline__ void dequantize_ptq1_0_qs4(uint32_t packed, float d, dst_t * __restrict__ y, int base, int stride) {
    uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
    uint32_t v_hi = __byte_perm(packed, 0, 0x4342);

#    pragma unroll
    for (int t = 0; t < 5; ++t) {
        const uint32_t w_lo = v_lo * 3;
        const uint32_t w_hi = v_hi * 3;
        v_lo                = w_lo & 0x00FF00FF;
        v_hi                = w_hi & 0x00FF00FF;

        const uint32_t q = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
#    pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int trit           = (int8_t) (q >> (8 * b));
            y[base + t * stride + b] = ggml_cuda_cast<dst_t>(d * trit);
        }
    }
}

template <typename dst_t>
static __global__ void dequantize_block_ptq1_0(const block_ptq1_0 * __restrict__ x,
                                               dst_t * __restrict__ y,
                                               int64_t nb) {
    constexpr int    threads_per_quant_block     = 8;
    constexpr int    quant_blocks_per_cuda_block = CUDA_DEQUANTIZE_BLOCK_SIZE / threads_per_quant_block;
    // The shared transpose keeps packed-byte ownership while making the CTA stores contiguous.
    __shared__ dst_t dequantized[quant_blocks_per_cuda_block * QK_PTQ1_0];

    const int64_t ib0   = int64_t(blockIdx.x) * quant_blocks_per_cuda_block;
    const int64_t ib    = ib0 + threadIdx.x / threads_per_quant_block;
    const int     lane  = threadIdx.x % threads_per_quant_block;
    const bool    valid = ib < nb;

    const block_ptq1_0 * bq = valid ? x + ib : x;
    float                d  = valid && lane == 7 ? (float) bq->d : 0.0f;
    d                       = __shfl_sync(__activemask(), d, 7, threads_per_quant_block);

    dst_t * out = dequantized + (threadIdx.x / threads_per_quant_block) * QK_PTQ1_0;
    if (valid && lane < 4) {
        const uint32_t packed = ((const uint32_t *) bq->qs)[lane];
        dequantize_ptq1_0_qs4(packed, d, out, 4 * lane, 16);
    } else if (valid && lane < 6) {
        const int      g      = lane - 4;
        const uint32_t packed = ((const uint32_t *) (bq->qs + 16))[g];
        dequantize_ptq1_0_qs4(packed, d, out, 80 + 4 * g, 8);
    } else if (valid && lane == 6) {
        uint32_t v = (uint32_t) bq->qh[0] | ((uint32_t) bq->qh[1] << 16);
#    pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint32_t w     = v * 3;
            v                    = w & 0x00FF00FF;
            out[120 + 2 * t + 0] = ggml_cuda_cast<dst_t>(d * ((int) ((w >> 8) & 0xFF) - 1));
            out[120 + 2 * t + 1] = ggml_cuda_cast<dst_t>(d * ((int) ((w >> 24) & 0xFF) - 1));
        }
    }

    __syncthreads();
    const int nblocks   = nb - ib0 < quant_blocks_per_cuda_block ? nb - ib0 : quant_blocks_per_cuda_block;
    const int nelements = nblocks * QK_PTQ1_0;
    for (int i = threadIdx.x; i < nelements; i += CUDA_DEQUANTIZE_BLOCK_SIZE) {
        y[ib0 * QK_PTQ1_0 + i] = dequantized[i];
    }
}

template <typename dst_t>
static void dequantize_row_ptq1_0_cuda(const void * __restrict__ vx,
                                       dst_t * __restrict__ y,
                                       const int64_t k,
                                       cudaStream_t  stream) {
    GGML_ASSERT(k % QK_PTQ1_0 == 0);
    constexpr int quant_blocks_per_cuda_block = CUDA_DEQUANTIZE_BLOCK_SIZE / 8;
    const int64_t nb                          = k / QK_PTQ1_0;
    const int     num_blocks                  = (nb + quant_blocks_per_cuda_block - 1) / quant_blocks_per_cuda_block;
    dequantize_block_ptq1_0<<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>((const block_ptq1_0 *) vx, y, nb);
}
#endif

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q2_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q3_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q5_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q6_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_m(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_mxfp4(vx, i, yy + i*QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

// ============================================================
// TBQ _4 → spatial f16: fused dequant + cooperative IWHT
// 1 CUDA block = 1 TBQ block. 256 threads cooperate on IWHT butterfly.
// ============================================================
template <typename block_type, int n_bits>
static __device__ __forceinline__ void tbq_dequant_sub(const block_type * blk, float * buf, int tid, int sub) {
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr float c4[16] = { -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,
                                       0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f };
    const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
    const uint8_t * qs = sub == 0 ? blk->qs1 : blk->qs2;
    if constexpr (n_bits == 3) {
        const int bp = tid*3, by = bp/8, bo = bp%8;
        uint32_t r = (uint32_t)qs[by]>>bo; if (bo>5) r|=(uint32_t)qs[by+1]<<(8-bo);
        buf[tid] = c3[r&7] * norm;
    } else {
        buf[tid] = c4[(qs[tid/2]>>((tid%2)*4))&0xF] * norm;
    }
}

// Optimized IWHT: warp shuffle for stages 0-4, shared memory for stages 5-7.
// 256 threads, each handles 2 elements (tid and tid+128). 8→3 __syncthreads.
static __device__ __forceinline__ void tbq_iwht_256(float * buf, int tid) {
    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    __syncthreads();

    // 256 threads, 256 elements. First 128 threads handle 2 elements each.
    // Threads 128-255 idle during butterfly but help with shared memory sync.
    if (tid < 128) {
        float f0 = buf[tid], f1 = buf[tid + 128];

        // Stage 7 (len=128): butterfly between f0 and f1 (same thread)
        { float u = f0, v = f1; f0 = u + v; f1 = u - v; }

        // Stages 0-4 (len=1,2,4,8,16): warp shuffle, no __syncthreads
        #pragma unroll
        for (int s = 0; s < 5; s++) {
            float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, 32);
            float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, 32);
            if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
            else                { f0 = f0 + o0; f1 = f1 + o1; }
        }

        // Stage 5 (len=32): cross-warp via shared memory
        buf[tid] = f0; buf[tid + 128] = f1;
    }
    __syncthreads();
    if (tid < 128) {
        float f0, f1;
        { float u0 = buf[tid], v0 = buf[tid ^ 32];
          float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 32];
          f0 = (tid & 32) ? (v0 - u0) : (u0 + v0);
          f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }

        // Stage 6 (len=64): cross-warp via shared memory
        buf[tid] = f0; buf[tid + 128] = f1;
    }
    __syncthreads();
    if (tid < 128) {
        float f0, f1;
        { float u0 = buf[tid], v0 = buf[tid ^ 64];
          float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 64];
          f0 = (tid & 64) ? (v0 - u0) : (u0 + v0);
          f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }

        // Sign unscramble + /256
        float s0 = ((signs[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
        float s1 = ((signs[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
        buf[tid]       = f0 * s0 / 256.0f;
        buf[tid + 128] = f1 * s1 / 256.0f;
    }
    __syncthreads();
}

template <typename block_type, int n_bits>
static __global__ void dequantize_tbq_4_spatial_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half * out = y + bi * TBQ_K576;
    int tid = threadIdx.x;
    __shared__ float buf[256];
    // Sub-block 1: dequant → IWHT → f16
    tbq_dequant_sub<block_type, n_bits>(blk, buf, tid, 0);
    tbq_iwht_256(buf, tid);
    out[tid] = __float2half(buf[tid]);
    // Sub-block 2: dequant → IWHT → f16
    tbq_dequant_sub<block_type, n_bits>(blk, buf, tid, 1);
    tbq_iwht_256(buf, tid);
    out[256+tid] = __float2half(buf[tid]);
    // Sub-block 3 (rope): f16 passthrough
    if (tid < 64) out[512+tid] = blk->rope[tid];
}

// TBQP → WHT-domain f16 with QJL correction (NO IWHT).
// QJL is valid in WHT domain: value = centroid*norm + qjl_sign*d_qjl.
// MMA computes WHT(Q)·WHT(K)^T = D×Q·K^T (correct by WHT orthogonality).
// Q WHT and output IWHT are applied externally in launch_fattn.
// Simple per-element kernel — 288 threads (576/2 half2), no shared memory needed.
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_wht_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half2 * out = (half2 *)(y + bi * TBQ_K576);
    const int tid = threadIdx.x;
    const int elem = tid * 2;

    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };

    float v0, v1;
    if (elem < 512) {
        const int sub = elem < 256 ? 0 : 1;
        const int e = elem - sub * 256;
        const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
        const float dq   = __half2float(sub == 0 ? blk->d1_qjl : blk->d2_qjl);
        const uint8_t * qs  = sub == 0 ? blk->qs1  : blk->qs2;
        const uint8_t * qjl = sub == 0 ? blk->qjl1 : blk->qjl2;
        const int s0 = ((qjl[e/8]>>(e%8))&1) ? 1 : -1;
        const int s1 = ((qjl[(e+1)/8]>>((e+1)%8))&1) ? 1 : -1;
        if constexpr (n_bits == 3) {
            // TBQP3: 2-bit MSE + 1-bit QJL
            v0 = c2[(qs[e/4]>>((e%4)*2))&0x3]*norm + s0*dq;
            v1 = c2[(qs[(e+1)/4]>>(((e+1)%4)*2))&0x3]*norm + s1*dq;
        } else {
            // TBQP4: 3-bit MSE + 1-bit QJL
            int bp0=e*3, by0=bp0/8, bo0=bp0%8;
            int bp1=(e+1)*3, by1=bp1/8, bo1=bp1%8;
            uint32_t r0=(uint32_t)qs[by0]>>bo0; if(bo0>5) r0|=(uint32_t)qs[by0+1]<<(8-bo0);
            uint32_t r1=(uint32_t)qs[by1]>>bo1; if(bo1>5) r1|=(uint32_t)qs[by1+1]<<(8-bo1);
            v0 = c3[r0&7]*norm + s0*dq;
            v1 = c3[r1&7]*norm + s1*dq;
        }
    } else {
        const int e = elem - 512;
        v0 = __half2float(blk->rope[e]);
        v1 = __half2float(blk->rope[e+1]);
    }
    out[tid] = make_half2(__float2half(v0), __float2half(v1));
}

// Q WHT: transform Q from spatial to WHT domain (for TBQP MMA pre-processing).
// 1 CUDA block = 1 sub-block (256 elements). 256 threads cooperate on butterfly.
static __global__ void tbq_q_wht_kernel(float * __restrict__ Q, int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;  // 0 or 1
    const int tid = threadIdx.x;

    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };

    __shared__ float buf[256];
    float * q = Q + row * row_stride + sub * 256;

    int s = ((signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = q[tid] * s;
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    q[tid] = buf[tid] / 256.0f;
}

// Fused Q_wht1 + Q_wht2: signs1 WHT → store Q_wht1, then signs2 WHT → store Q_wht2.
// Single kernel, single block read, shared memory reused between the two WHTs.
// Output: Q_wht1 at q1[sub*256+tid], Q_wht2 at q2[sub*256+tid]
static __global__ void tbq_q_wht12_kernel(const float * __restrict__ Q_src,
                                           float * __restrict__ Q_wht1, float * __restrict__ Q_wht2,
                                           int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;
    const int tid = threadIdx.x;

    static constexpr uint8_t signs1[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr uint8_t signs2[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };

    __shared__ float buf[256];
    const float * q_src = Q_src + row * row_stride + sub * 256;
    float * q1 = Q_wht1 + row * row_stride + sub * 256;
    float * q2 = Q_wht2 + row * row_stride + sub * 256;

    // === WHT 1: signs1 → butterfly → /256 ===
    int s1 = ((signs1[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = q_src[tid] * s1;  // Read from Q_src directly
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    q1[tid] = buf[tid] / 256.0f;  // Store Q_wht1
    // buf still has WHT(signs1*Q) (un-scaled) — use for Q_wht2

    // === WHT 2: signs2 applied to WHT result, then butterfly, then qjl_factor/(256*256) ===
    int s2 = ((signs2[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = buf[tid] * s2;  // signs2 * WHT(signs1*Q)
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    constexpr float qjl_factor = 1.2533f;
    q2[tid] = buf[tid] * qjl_factor / (256.0f * 256.0f);  // Store Q_wht2
}

// Output IWHT: transform attention output from WHT to spatial domain (for TBQP MMA post-processing).
static __global__ void tbq_output_iwht_kernel(float * __restrict__ dst, int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;
    const int tid = threadIdx.x;

    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };

    __shared__ float buf[256];
    float * out = dst + row * row_stride + sub * 256;

    buf[tid] = out[tid];
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    int s = ((signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
    out[tid] = buf[tid] * s / 256.0f;
}

// Host wrappers
void tbq_q_wht12_cuda(const float * Q_src, float * Q_wht1, float * Q_wht2, int64_t DKQ, int64_t n_rows, int64_t row_stride, cudaStream_t stream) {
    tbq_q_wht12_kernel<<<dim3(n_rows, 2), 256, 0, stream>>>(Q_src, Q_wht1, Q_wht2, n_rows, row_stride);
}

static void dequantize_row_tbq3_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbq_4_spatial_f16_kernel<block_tbq3_4, 3><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}
static void dequantize_row_tbq4_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbq_4_spatial_f16_kernel<block_tbq4_4, 4><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}
static void dequantize_row_tbqp3_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    // WHT-domain dequant with QJL. Q WHT + output IWHT applied externally.
    dequantize_tbqp_4_wht_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}
static void dequantize_row_tbqp4_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_wht_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}

// TBQP V spatial dequant: MSE only (no QJL) + cooperative IWHT → spatial f16.
// For TBQP MMA: K uses WHT+QJL, V needs MSE-only spatial f16 (QJL breaks IWHT reconstruction).
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_v_spatial_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half * out = y + bi * TBQ_K576;
    int tid = threadIdx.x;
    __shared__ float buf[256];
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    for (int sub = 0; sub < 2; ++sub) {
        const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
        const uint8_t * qs = sub == 0 ? blk->qs1 : blk->qs2;
        if constexpr (n_bits == 3) {
            buf[tid] = c2[(qs[tid/4]>>((tid%4)*2))&0x3] * norm;
        } else {
            int bp = tid*3, by = bp/8, bo = bp%8;
            uint32_t r = (uint32_t)qs[by]>>bo; if (bo>5) r|=(uint32_t)qs[by+1]<<(8-bo);
            buf[tid] = c3[r&7] * norm;
        }
        tbq_iwht_256(buf, tid);
        out[sub*256+tid] = __float2half(buf[tid]);
    }
    if (tid < 64) out[512+tid] = blk->rope[tid];
}

void tbqp3_v_spatial_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_v_spatial_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}
void tbqp4_v_spatial_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_v_spatial_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}

// (Q WHT and output IWHT kernels defined above — no duplicates)

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_PQ2_0:
            return dequantize_block_cont_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;
        case GGML_TYPE_PTQ1_0:
#if !defined(GGML_USE_HIP)
            return dequantize_row_ptq1_0_cuda;
#else
            return dequantize_block_cont_cuda<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>;
#endif
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_PQ2_0:
            return dequantize_block_cont_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;
        case GGML_TYPE_PTQ1_0:
#if !defined(GGML_USE_HIP)
            return dequantize_row_ptq1_0_cuda;
#else
            return dequantize_block_cont_cuda<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>;
#endif
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        case GGML_TYPE_TBQ3_4:
            return dequantize_row_tbq3_4_to_f16_cuda;
        case GGML_TYPE_TBQ4_4:
            return dequantize_row_tbq4_4_to_f16_cuda;
        case GGML_TYPE_TBQP3_4:
            return (to_fp16_cuda_t)tbqp3_v_spatial_f16_cuda;  // Spatial (MSE + IWHT). V=K view=spatial, no output IWHT.
        case GGML_TYPE_TBQP4_4:
            return (to_fp16_cuda_t)tbqp4_v_spatial_f16_cuda;
        default:
            return nullptr;
    }
}

// ============================================================
// TBQ _4 → spatial f16: fused dequant + cooperative IWHT
// 1 CUDA block = 1 TBQ block (576 elements). 256 threads cooperate on IWHT butterfly.
// (duplicate removed — definitions above)

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_PQ2_0:
            return dequantize_block_cont_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;
        case GGML_TYPE_PTQ1_0:
#if !defined(GGML_USE_HIP)
            return dequantize_row_ptq1_0_cuda;
#else
            return dequantize_block_cont_cuda<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>;
#endif
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_PQ2_0:
            return dequantize_block_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;
        case GGML_TYPE_PTQ1_0:
            return dequantize_block_cuda<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_PQ2_0:
            return dequantize_block_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;
        case GGML_TYPE_PTQ1_0:
            return dequantize_block_cuda<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_PQ2_0:
            return dequantize_block_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;
        case GGML_TYPE_PTQ1_0:
            return dequantize_block_cuda<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
