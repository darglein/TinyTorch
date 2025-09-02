#include "half.h"


#ifdef _MSC_VER
#pragma warning( disable : 4146 ) // warning C4146: unary minus operator applied to unsigned type, result still unsigned
#endif


#if 0
inline uint32_t uint32_nor(uint32_t a, uint32_t b)
{
    return ~(a | b);
}

inline uint32_t uint32_andc(uint32_t a, uint32_t b)
{
    return (a & (~b));
}

inline uint32_t uint32_and(uint32_t a, uint32_t b)
{
    return (a & b);
}

inline uint32_t uint32_or(uint32_t a, uint32_t b)
{
    return (a | b);
}

inline uint32_t uint32_or3(uint32_t a, uint32_t b, uint32_t c)
{
    return (a | b | c);
}

inline uint32_t uint32_mux(uint32_t mask, uint32_t a, uint32_t b)
{
    return ((mask & (uint32_t)a) | ((~mask) & (uint32_t)b));
}

inline uint32_t uint32_lt(uint32_t a, uint32_t b)
{
    // NOTE: Result is invalid when a=INT32_MAX, b=INT32_MIN
    // For the purposes used in half.c the result is always valid
    return (uint32_t)((int32_t)(a - b) >> 31);
}

inline uint32_t uint32_gte(uint32_t a, uint32_t b)
{
    return ~uint32_lt(a, b);
}

inline uint32_t uint32_gt(uint32_t a, uint32_t b)
{
    // NOTE: Result is invalid when b=INT32_MIN, a=INT32_MAX
    // For the purposes used in half.c the result is always valid
    return (uint32_t)((int32_t)(b - a) >> 31);
}

inline uint32_t uint32_nez(uint32_t a)
{
    return (uint32_t)((int32_t)(a | -a) >> 31);
}

inline uint32_t uint32_eqz(uint32_t a)
{
    return ~uint32_nez(a);
}

inline uint32_t uint32_nez_p(uint32_t a)
{
    return (uint32_t)((uint32_t)(a | -a) >> 31);
}

inline uint32_t uint32_eq(uint32_t a, uint32_t b)
{
    return (~uint32_nez(a - b));
}

inline uint32_t uint32_srl(uint32_t a, uint32_t sa)
{
    return (a >> sa);
}

inline uint32_t uint32_sll(uint32_t a, uint32_t sa)
{
    return (a << sa);
}

inline uint32_t uint32_cp(uint32_t a)
{
    return (a);
}

inline uint32_t uint32_add(uint32_t a, uint32_t b)
{
    return (a + b);
}

inline uint32_t uint32_sub(uint32_t a, uint32_t b)
{
    return (a - b);
}

inline uint16_t uint16_mux(uint16_t mask, uint16_t a, uint16_t b)
{
    return ((mask & (uint16_t)a) | ((~mask) & (uint16_t)b));
}

inline uint16_t uint16_lt(uint16_t a, uint16_t b)
{
    // NOTE: Result is invalid when a=INT16_MAX, b=INT16_MIN
    // For the purposes used in half.c the result is always valid
    return (uint16_t)((int16_t)(a - b) >> 15);
}

inline uint16_t uint16_gte(uint16_t a, uint16_t b)
{
    return ~uint16_lt(a, b);
}

inline uint16_t uint16_gt(uint16_t a, uint16_t b)
{
    // NOTE: Result is invalid when b=INT32_MIN, a=INT32_MAX
    // For the purposes used in half.c the result is always valid
    return (uint16_t)((int16_t)(b - a) >> 15);
}

inline uint16_t uint16_nez(uint16_t a)
{
    return (uint16_t)((int16_t)(a | -a) >> 15);
}

inline uint16_t uint16_eqz(uint16_t a)
{
    return ~uint16_nez(a);
}

inline uint16_t uint16_nez_p(uint16_t a)
{
    return (uint16_t)((uint16_t)(a | -a) >> 15);
}

inline uint16_t uint16_eqz_p(uint16_t a)
{
    return ~uint16_nez_p(a);
}

inline uint16_t uint16_eq(uint16_t a, uint16_t b)
{
    return (~uint16_nez(a - b));
}

inline uint16_t uint16_andc(uint16_t a, uint16_t b)
{
    return (a & (~b));
}

inline uint16_t uint16_and(uint16_t a, uint16_t b)
{
    return (a & b);
}

inline uint16_t uint16_andsrl(uint16_t a, uint16_t b, uint16_t sa)
{
    return ((a & b) >> sa);
}

inline uint16_t uint16_or(uint16_t a, uint16_t b)
{
    return (a | b);
}

inline uint16_t uint16_or3(uint16_t a, uint16_t b, uint16_t c)
{
    return (a | b | c);
}

inline uint16_t uint16_add(uint16_t a, uint16_t b)
{
    return (a + b);
}

inline uint16_t uint16_addm(uint16_t a, uint16_t b, uint16_t mask)
{
    return ((a + b) & mask);
}

inline uint16_t uint16_sub(uint16_t a, uint16_t b)
{
    return (a - b);
}

inline uint16_t uint16_xor(uint16_t a, uint16_t b)
{
    return (a ^ b);
}

inline uint16_t uint16_srl(uint16_t a, uint16_t sa)
{
    return (a >> sa);
}

inline uint16_t uint16_srlm(uint16_t a, uint16_t sa, uint16_t mask)
{
    return ((a >> sa) & mask);
}

inline uint16_t uint16_sll(uint16_t a, uint16_t sa)
{
    return (a << sa);
}

inline uint16_t uint16_not(uint16_t a)
{
    return (~a);
}

inline uint16_t uint16_cp(uint16_t a)
{
    return (a);
}

inline uint16_t uint16_cntlz(uint16_t x)
{
    const uint16_t x0  = uint16_srl(x, 1);
    const uint16_t x1  = uint16_or(x, x0);
    const uint16_t x2  = uint16_srl(x1, 2);
    const uint16_t x3  = uint16_or(x1, x2);
    const uint16_t x4  = uint16_srl(x3, 4);
    const uint16_t x5  = uint16_or(x3, x4);
    const uint16_t x6  = uint16_srl(x5, 8);
    const uint16_t x7  = uint16_or(x5, x6);
    const uint16_t x8  = uint16_not(x7);
    const uint16_t x9  = uint16_srlm(x8, 1, 0x5555);
    const uint16_t xA  = uint16_sub(x8, x9);
    const uint16_t xB  = uint16_and(xA, 0x3333);
    const uint16_t xC  = uint16_srlm(xA, 2, 0x3333);
    const uint16_t xD  = uint16_add(xB, xC);
    const uint16_t xE  = uint16_srl(xD, 4);
    const uint16_t xF  = uint16_addm(xD, xE, 0x0f0f);
    const uint16_t x10 = uint16_srl(xF, 8);
    const uint16_t x11 = uint16_addm(xF, x10, 0x001f);

    return (x11);
}

inline uint16_t half_from_float2(uint32_t f)
{
    const uint16_t one                        = 0x0001;
    const uint32_t f_s_mask                   = 0x80000000;
    const uint32_t f_e_mask                   = 0x7f800000;
    const uint32_t f_m_mask                   = 0x007fffff;
    const uint32_t f_m_hidden_bit             = 0x00800000;
    const uint32_t f_m_round_bit              = 0x00001000;
    const uint32_t f_snan_mask                = 0x7fc00000;
    const uint16_t f_e_bias                   = 0x007f;
    const uint16_t h_e_bias                   = 0x000f;
    const uint16_t f_s_pos                    = 0x001f;
    const uint16_t h_s_pos                    = 0x000f;
    const uint16_t f_e_pos                    = 0x0017;
    const uint16_t h_e_pos                    = 0x000a;
    const uint16_t h_e_mask                   = 0x7c00;
    const uint16_t h_snan_mask                = 0x7e00;
    const uint16_t f_e_flagged_value          = 0x00ff;
    const uint16_t h_e_mask_value             = uint16_srl(h_e_mask, h_e_pos);
    const uint16_t f_h_s_pos_offset           = uint16_sub(f_s_pos, h_s_pos);
    const uint16_t f_h_bias_offset            = uint16_sub(f_e_bias, h_e_bias);
    const uint16_t f_h_m_pos_offset           = uint16_sub(f_e_pos, h_e_pos);
    const uint16_t h_nan_min                  = uint16_or(h_e_mask, one);
    const uint32_t f_s_masked                 = uint32_and(f, f_s_mask);
    const uint32_t f_e_masked                 = uint32_and(f, f_e_mask);
    const uint16_t h_s                        = uint32_srl(f_s_masked, f_h_s_pos_offset);
    const uint16_t f_e                        = uint32_srl(f_e_masked, f_e_pos);
    const uint32_t f_m                        = uint32_and(f, f_m_mask);
    const uint16_t f_e_half_bias              = uint16_sub(f_e, f_h_bias_offset);
    const uint32_t f_m_round_mask             = uint32_and(f_m, f_m_round_bit);
    const uint32_t f_m_round_offset           = uint32_sll(f_m_round_mask, one);
    const uint32_t f_m_rounded                = uint32_add(f_m, f_m_round_offset);
    const uint32_t f_m_denorm_sa              = uint32_sub(one, f_e_half_bias);
    const uint32_t f_m_with_hidden            = uint32_or(f_m_rounded, f_m_hidden_bit);
    const uint32_t f_m_denorm                 = uint32_srl(f_m_with_hidden, f_m_denorm_sa);
    const uint16_t h_m_denorm                 = uint32_srl(f_m_denorm, f_h_m_pos_offset);
    const uint16_t h_denorm                   = uint16_or(h_s, h_m_denorm);
    const uint16_t h_inf                      = uint16_or(h_s, h_e_mask);
    const uint16_t m_nan                      = uint32_srl(f_m, f_h_m_pos_offset);
    const uint16_t h_nan                      = uint16_or3(h_s, h_e_mask, m_nan);
    const uint16_t h_nan_notinf               = uint16_or(h_s, h_nan_min);
    const uint16_t h_e_norm_overflow_offset   = uint16_add(f_e_half_bias, one);
    const uint16_t h_e_norm_overflow          = uint16_sll(h_e_norm_overflow_offset, h_e_pos);
    const uint16_t h_norm_overflow            = uint16_or(h_s, h_e_norm_overflow);
    const uint16_t h_e_norm                   = uint16_sll(f_e_half_bias, h_e_pos);
    const uint16_t h_m_norm                   = uint32_srl(f_m_rounded, f_h_m_pos_offset);
    const uint16_t h_norm                     = uint16_or3(h_s, h_e_norm, h_m_norm);
    const uint16_t is_h_denorm                = uint16_gte(f_h_bias_offset, f_e);
    const uint16_t f_h_e_biased_flag          = uint16_sub(f_e_flagged_value, f_h_bias_offset);
    const uint16_t is_f_e_flagged             = uint16_eq(f_e_half_bias, f_h_e_biased_flag);
    const uint16_t is_f_m_zero                = uint32_eqz(f_m);
    const uint16_t is_h_nan_zero              = uint16_eqz(m_nan);
    const uint16_t is_f_inf                   = uint16_and(is_f_e_flagged, is_f_m_zero);
    const uint16_t is_f_nan_underflow         = uint16_and(is_f_e_flagged, is_h_nan_zero);
    const uint16_t is_f_nan                   = uint16_cp(is_f_e_flagged);
    const uint16_t is_e_overflow              = uint16_gt(f_e_half_bias, h_e_mask_value);
    const uint32_t f_m_rounded_overflow       = uint32_and(f_m_rounded, f_m_hidden_bit);
    const uint32_t is_m_norm_overflow         = uint32_nez(f_m_rounded_overflow);
    const uint16_t is_h_inf                   = uint16_or(is_e_overflow, is_f_inf);
    const uint32_t f_snan                     = uint32_and(f, f_snan_mask);
    const uint32_t is_f_snan                  = uint32_eq(f_snan, f_snan_mask);
    const uint16_t h_snan                     = uint16_or(h_s, h_snan_mask);
    const uint16_t check_overflow_result      = uint16_mux(is_m_norm_overflow, h_norm_overflow, h_norm);
    const uint16_t check_nan_result           = uint16_mux(is_f_nan, h_nan, check_overflow_result);
    const uint16_t check_nan_underflow_result = uint16_mux(is_f_nan_underflow, h_nan_notinf, check_nan_result);
    const uint16_t check_inf_result           = uint16_mux(is_h_inf, h_inf, check_nan_underflow_result);
    const uint16_t check_denorm_result        = uint16_mux(is_h_denorm, h_denorm, check_inf_result);
    const uint16_t check_snan_result          = uint16_mux(is_f_snan, h_snan, check_denorm_result);
    const uint16_t result                     = uint16_cp(check_snan_result);

    return (result);
}
#endif