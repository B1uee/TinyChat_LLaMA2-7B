#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

namespace matmul {
void MatmulOperator::mat_mul_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;  
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) { 
        for (int col = 0; col < n; col += 4) { // 一次算4个C的元素
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) { // 注意，ch是按照block_size*2(x86)来更新的，因此后面的指针指的都是数组起始位置
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch]; // 长度为64B的数组，存有64个A_Element
                // pointer of the int4 weights    
                // col对B来说是行，相当于取了B的列的4个值 与A(row,ch)的值计算得到C(row,col~col+3)的四个值的1/k
                // 循环完一遍后即可直接得到C的四个值，实现循环展开 
                // 为了最大化SIMD指令集的效率，读满了w，而A为了对齐就不得不多读一倍。
                uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];    // 长度为32B的数组，存有64个B_Element
                uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];  // 用uint8_存int4，索引/2
                uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];  //每个block_size共享一个缩放系数s
                // scale of weight
                float s_w0 = params->scales[(col * k + ch) / block_size];
                float s_w1 = params->scales[((col + 1) * k + ch) / block_size];
                float s_w2 = params->scales[((col + 2) * k + ch) / block_size];
                float s_w3 = params->scales[((col + 3) * k + ch) / block_size];
#ifdef QM_ARM
                // order of weights with QM_ARM:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
                // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         128 bit                         127
                // process 16 bytes of weigths (128 bit) = 1 block for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)

                    // TODO: int8 multiply and accumulate operation
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                ch += block_size;
#endif
#ifdef QM_x86   // SIMD指令集允许同时处理多个数据元素，其宽度与处理器的字长无关。
                // arm: 常用的 NEON 指令集支持 128 位寄存器
                // x86: 通常使用 AVX 指令集（如 AVX2），支持 256 位寄存器，即32Byte，由于INT4 QUANT，
                // 会有来自两个块的数据交替存放，因此要提前获取下一个块的缩放系数，同时qj的循环次数改为32    
                // scales of the second block  获取下一个block的缩放系数
                float s_w0_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_w1_2nd = params->scales[((col + 1) * k + ch) / block_size + 1];
                float s_w2_2nd = params->scales[((col + 2) * k + ch) / block_size + 1];
                float s_w3_2nd = params->scales[((col + 3) * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_X86 order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                int intermediate_sum0_2nd = 0, intermediate_sum1_2nd = 0, intermediate_sum2_2nd = 0,
                    intermediate_sum3_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {   // qj <= 32 Byte
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)  
                    uint8_t packed_w_col_0 = w0_int4[qj];   // 虽然做了&操作并转为int，但符号位会默认在第一位，这里用的4bit，还是需要手动-8
                    signed char w_low_0 = (packed_w_col_0 & 0x0F) - 8;  
                    signed char w_high_0 = (packed_w_col_0 >> 4) - 8;  // 右移左边自动赋全0

                    uint8_t packed_w_col_1 = w1_int4[qj];  
                    signed char w_low_1 = (packed_w_col_1 & 0x0F) - 8;  
                    signed char w_high_1 = (packed_w_col_1 >> 4) - 8; 

                    uint8_t packed_w_col_2 = w2_int4[qj];  
                    signed char w_low_2 = (packed_w_col_2 & 0x0F) - 8;  
                    signed char w_high_2 = (packed_w_col_2 >> 4) - 8; 

                    uint8_t packed_w_col_3 = w3_int4[qj];  
                    signed char w_low_3 = (packed_w_col_3 & 0x0F) - 8;  
                    signed char w_high_3 = (packed_w_col_3 >> 4) - 8; 

                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum0 += w_low_0 *  a_int8[qj];
                    intermediate_sum0_2nd += w_high_0 *  a_int8[qj + 32];

                    intermediate_sum1 += w_low_1 *  a_int8[qj];
                    intermediate_sum1_2nd += w_high_1 *  a_int8[qj + 32];

                    intermediate_sum2 += w_low_2 *  a_int8[qj];
                    intermediate_sum2_2nd += w_high_2 *  a_int8[qj + 32];

                    intermediate_sum3 += w_low_3 *  a_int8[qj];
                    intermediate_sum3_2nd += w_high_3 *  a_int8[qj + 32];
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc0 += (float)intermediate_sum0_2nd * s_a_2nd * s_w0_2nd;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc1 += (float)intermediate_sum1_2nd * s_a_2nd * s_w1_2nd;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc2 += (float)intermediate_sum2_2nd * s_a_2nd * s_w2_2nd;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                acc3 += (float)intermediate_sum3_2nd * s_a_2nd * s_w3_2nd;
                // process two blocks
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
};
}  // namespace matmul
