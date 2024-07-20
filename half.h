#ifndef HALF_H
#define HALF_H

#include <cstdio>
#include <cmath>
#include <climits>
#include <cfloat>
#include <cstdint>

/**
 * Experimental implementation of half-precision floating point conversion functions.
 */
struct half {
    uint16_t data;
};

half halfFromFloat(float f)
{
    union {
        float f;
        uint32_t u;
    } floatUnion = { f };

    uint32_t float32 = floatUnion.u;

    // Constants for bit masks, shifts, and biases
    const uint16_t ONE                     = 0x0001;
    const uint32_t FLOAT_SIGN_MASK         = 0x80000000;
    const uint32_t FLOAT_EXP_MASK          = 0x7f800000;
    const uint32_t FLOAT_MANTISSA_MASK     = 0x007fffff;
    const uint32_t FLOAT_HIDDEN_BIT        = 0x00800000;
    const uint32_t FLOAT_ROUND_BIT         = 0x00001000;
    const uint16_t FLOAT_EXP_BIAS          = 0x007f;
    const uint16_t HALF_EXP_BIAS           = 0x000f;
    const uint16_t FLOAT_SIGN_POS          = 0x001f;
    const uint16_t HALF_SIGN_POS           = 0x000f;
    const uint16_t FLOAT_EXP_POS           = 0x0017;
    const uint16_t HALF_EXP_POS            = 0x000a;
    const uint16_t HALF_EXP_MASK           = 0x7c00;
    const uint16_t FLOAT_EXP_FLAGGED_VALUE = 0x00ff;
    const uint16_t HALF_EXP_MASK_VALUE     = HALF_EXP_MASK >> HALF_EXP_POS;
    const uint16_t HALF_EXP_MAX_VALUE      = HALF_EXP_MASK_VALUE - ONE;
    const uint16_t FLOAT_HALF_SIGN_POS_OFFSET = FLOAT_SIGN_POS - HALF_SIGN_POS;
    const uint16_t FLOAT_HALF_BIAS_OFFSET  = FLOAT_EXP_BIAS - HALF_EXP_BIAS;
    const uint16_t FLOAT_HALF_MANTISSA_POS_OFFSET = FLOAT_EXP_POS - HALF_EXP_POS;
    const uint16_t HALF_NAN_MIN            = HALF_EXP_MASK | ONE;

    // Extracting the sign, exponent, and mantissa from the 32-bit float
    uint32_t floatSignMasked = float32 & FLOAT_SIGN_MASK;
    uint32_t floatExpMasked  = float32 & FLOAT_EXP_MASK;
    uint16_t halfSign        = static_cast<uint16_t>(floatSignMasked >> FLOAT_HALF_SIGN_POS_OFFSET);
    uint16_t floatExp        = static_cast<uint16_t>(floatExpMasked >> FLOAT_EXP_POS);
    uint32_t floatMantissa   = float32 & FLOAT_MANTISSA_MASK;

    // Check for NaN
    if ((floatExpMasked == FLOAT_EXP_MASK) && (floatMantissa != 0)) {
        half result;
        result.data = HALF_EXP_MASK | (floatMantissa >> FLOAT_HALF_MANTISSA_POS_OFFSET);
        return result;
    }

    // Adjusting the exponent and rounding the mantissa
    uint16_t floatExpHalfBias = floatExp - FLOAT_HALF_BIAS_OFFSET;
    uint32_t floatMantissaRoundMask = floatMantissa & FLOAT_ROUND_BIT;
    uint32_t floatMantissaRoundOffset = floatMantissaRoundMask << ONE;
    uint32_t floatMantissaRounded = floatMantissa + floatMantissaRoundOffset;

    // Handling denormalized numbers
    uint32_t floatMantissaDenormShiftAmount = ONE - floatExpHalfBias;
    uint32_t floatMantissaWithHidden = floatMantissaRounded | FLOAT_HIDDEN_BIT;
    uint32_t floatMantissaDenorm = floatMantissaWithHidden >> floatMantissaDenormShiftAmount;
    uint16_t halfMantissaDenorm = static_cast<uint16_t>(floatMantissaDenorm >> FLOAT_HALF_MANTISSA_POS_OFFSET);
    uint16_t halfDenorm = halfSign | halfMantissaDenorm;

    // Handling special cases: infinity and NaN
    uint16_t halfInf = halfSign | HALF_EXP_MASK;
    uint16_t mantissaNan = static_cast<uint16_t>(floatMantissa >> FLOAT_HALF_MANTISSA_POS_OFFSET);
    uint16_t halfNan = halfSign | HALF_EXP_MASK | mantissaNan;
    uint16_t halfNanNotInf = halfSign | HALF_NAN_MIN;

    // Handling overflow
    uint16_t halfExpNormOverflowOffset = floatExpHalfBias + ONE;
    uint16_t halfExpNormOverflow = halfExpNormOverflowOffset << HALF_EXP_POS;
    uint16_t halfNormOverflow = halfSign | halfExpNormOverflow;

    // Handling normalized numbers
    uint16_t halfExpNorm = floatExpHalfBias << HALF_EXP_POS;
    uint16_t halfMantissaNorm = static_cast<uint16_t>(floatMantissaRounded >> FLOAT_HALF_MANTISSA_POS_OFFSET);
    uint16_t halfNorm = halfSign | halfExpNorm | halfMantissaNorm;

    // Checks and conditions
    uint16_t halfIsDenorm = FLOAT_HALF_BIAS_OFFSET >= floatExp;
    uint16_t floatHalfExpBiasedFlag = FLOAT_EXP_FLAGGED_VALUE - FLOAT_HALF_BIAS_OFFSET;
    uint16_t floatExpIsFlagged = floatExpHalfBias == floatHalfExpBiasedFlag;
    uint16_t isFloatMantissaZero = floatMantissa == 0;
    uint16_t isHalfNanZero = mantissaNan == 0;
    uint16_t floatIsInf = floatExpIsFlagged && isFloatMantissaZero;
    uint16_t floatIsNanUnderflow = floatExpIsFlagged && isHalfNanZero;
    uint16_t floatIsNan = floatExpIsFlagged;
    uint16_t expIsOverflow = floatExpHalfBias > HALF_EXP_MAX_VALUE;
    uint32_t floatMantissaRoundedOverflow = floatMantissaRounded & FLOAT_HIDDEN_BIT;
    uint32_t mantissaNormIsOverflow = floatMantissaRoundedOverflow != 0;
    uint16_t halfIsInf = expIsOverflow || floatIsInf;

    // Selecting final result based on conditions
    uint16_t checkOverflowResult = mantissaNormIsOverflow ? halfNormOverflow : halfNorm;
    uint16_t checkNanResult = floatIsNan ? halfNan : checkOverflowResult;
    uint16_t checkNanUnderflowResult = floatIsNanUnderflow ? halfNanNotInf : checkNanResult;
    uint16_t checkInfResult = halfIsInf ? halfInf : checkNanUnderflowResult;
    uint16_t checkDenormResult = halfIsDenorm ? halfDenorm : checkInfResult;

    // Final result after all checks
    half result;
    result.data = checkDenormResult;

    return result;
}

float halfToFloat(half h)
{
    // Constants for bit masks, shifts, and biases
    const uint16_t ONE                   = 0x0001;
    const uint16_t TWO                   = 0x0002;
    const uint32_t FLOAT_EXP_MASK        = 0x7f800000;
    const uint32_t FLOAT_MANTISSA_MASK   = 0x007fffff;
    const uint16_t FLOAT_EXP_BIAS        = 0x007f;
    const uint16_t HALF_EXP_BIAS         = 0x000f;
    const uint16_t HALF_SIGN_MASK        = 0x8000;
    const uint16_t HALF_EXP_MASK         = 0x7c00;
    const uint16_t HALF_MANTISSA_MASK    = 0x03ff;
    const uint16_t HALF_EXP_POS          = 0x000a;
    const uint16_t FLOAT_EXP_POS         = 0x0017;
    const uint16_t FLOAT_SIGN_POS        = 0x001f;
    const uint16_t HALF_SIGN_POS         = 0x000f;
    const uint16_t HALF_FLOAT_DENORM_SA_OFFSET = 0x000a;
    const uint32_t HALF_FLOAT_BIAS_OFFSET = HALF_EXP_BIAS - FLOAT_EXP_BIAS;
    const uint16_t HALF_FLOAT_SIGN_POS_OFFSET = FLOAT_SIGN_POS - HALF_SIGN_POS;
    const uint16_t HALF_FLOAT_MANTISSA_POS_OFFSET = FLOAT_EXP_POS - HALF_EXP_POS;

    // Extracting the sign, exponent, and mantissa from the 16-bit float
    uint32_t halfSignMasked = h.data & HALF_SIGN_MASK;
    uint32_t halfExpMasked  = h.data & HALF_EXP_MASK;
    uint16_t halfMantissa   = h.data & HALF_MANTISSA_MASK;

    // Shifting the sign bit to the correct position for the 32-bit float
    uint32_t floatSign = halfSignMasked << HALF_FLOAT_SIGN_POS_OFFSET;

    // Adjusting the exponent
    uint16_t halfExpHalfBias = halfExpMasked >> HALF_EXP_POS;
    uint32_t floatExp = halfExpHalfBias - HALF_FLOAT_BIAS_OFFSET;

    // Shifting the mantissa to the correct position for the 32-bit float
    uint32_t floatMantissa = halfMantissa << HALF_FLOAT_MANTISSA_POS_OFFSET;

    // Checking conditions for zero, denormalized, infinity, and NaN
    uint32_t isExpNonZero = halfExpMasked != 0;
    uint32_t isMantissaNonZero = halfMantissa != 0;
    uint32_t isZero = !(isExpNonZero || isMantissaNonZero);
    uint32_t isDenorm = !isZero && !isExpNonZero;
    uint32_t isExpFlagged = halfExpMasked == HALF_EXP_MASK;
    uint32_t isInf = isExpFlagged && !isMantissaNonZero;
    uint32_t isNan = isExpFlagged && isMantissaNonZero;

    // Handling denormalized numbers
    uint16_t halfMantissaLeadingZeros = __builtin_clz(halfMantissa) - 16;
    uint16_t halfDenormShiftAmount = halfMantissaLeadingZeros + HALF_FLOAT_DENORM_SA_OFFSET;
    uint32_t halfFloatDenormMantissaShiftAmount = halfDenormShiftAmount - TWO;
    uint32_t halfFloatDenormMantissa = halfMantissa << halfFloatDenormMantissaShiftAmount;
    uint32_t floatDenormMantissa = halfFloatDenormMantissa & FLOAT_MANTISSA_MASK;
    uint32_t halfFloatDenormShiftAmount = ONE - halfDenormShiftAmount;
    uint32_t floatDenormExp = halfFloatDenormShiftAmount + FLOAT_EXP_BIAS;
    uint32_t floatDenormExpPacked = floatDenormExp << FLOAT_EXP_POS;
    uint32_t floatDenorm = floatSign | floatDenormExpPacked | floatDenormMantissa;

    // Handling special cases: infinity and NaN
    uint32_t floatInf = floatSign | FLOAT_EXP_MASK;
    uint32_t floatNan = floatSign | FLOAT_EXP_MASK | floatMantissa;

    // Handling zero
    uint32_t floatZero = floatSign;

    // Handling normalized numbers
    uint32_t floatExpPacked = floatExp << FLOAT_EXP_POS;
    uint32_t packed = floatSign | floatExpPacked | floatMantissa;

    // Selecting final result based on conditions
    uint32_t checkZeroResult = isZero ? floatZero : packed;
    uint32_t checkDenormResult = isDenorm ? floatDenorm : checkZeroResult;
    uint32_t checkInfResult = isInf ? floatInf : checkDenormResult;
    uint32_t checkNanResult = isNan ? floatNan : checkInfResult;

    // Final result after all checks
    uint32_t result = checkNanResult;

    // Reinterpret the uint32_t result as a float using a union
    union {
        uint32_t u;
        float f;
    } floatUnion;
    floatUnion.u = result;

    return floatUnion.f;
}

#endif // HALF_H
