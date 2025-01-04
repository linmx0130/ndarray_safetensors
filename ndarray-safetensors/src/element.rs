/// Element type traits for data types supported by both ndarray and safetensors
pub trait CommonSupportedElement: Clone {
    /// Extend the buffer vector with the little endian bytes of this value.
    fn extend_byte_vec(&self, v: &mut Vec<u8>);
    /// Safetensor dtype for the type.
    fn safetensors_dtype() -> safetensors::Dtype;
    /// Create the element value from bytes slice. Caller should ensure that it has enough
    /// bytes to consume.
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl CommonSupportedElement for f32 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    } 
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::F32
    }   

    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        f32::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for f64 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::F64
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed: [u8; 8] = [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]];
        f64::from_le_bytes(bytes_fixed)
    }
    
}

impl CommonSupportedElement for i8 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::I8
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0]];
        i8::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for u8 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.push(*self);
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::U8
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

impl CommonSupportedElement for i16 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::I16
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0], bytes[1]];
        i16::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for u16 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::U16
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0], bytes[1]];
        u16::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for i32 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::I32
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0], bytes[1], bytes[2], bytes[3]];
        i32::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for u32 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::U32
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0], bytes[1], bytes[2], bytes[3]];
        u32::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for i64 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::I64
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]];
        i64::from_le_bytes(bytes_fixed)
    }
}

impl CommonSupportedElement for u64 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::U64
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        let bytes_fixed = [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]];
        u64::from_le_bytes(bytes_fixed)
    }
}

/// Element type traits that can be used to load/save IEEE 754 binary 16 floating point numbers
/// 
/// This trait is internal for this crate even though we will need to expose it as a part of the API.
/// It could be changed in the future (for example, if fp16 is natively supported by Rust).
/// 
/// All APIs are experimental.
pub trait Float16ConversionSupportedElement where Self: CommonSupportedElement + ndarray::NdFloat {
    /// Create an instance from fp16 little-endian bytes
    fn from_fp16_bytes(bytes: &[u8]) -> Self;

    /// Conver the value as fp16 and save it to the buffer.
    fn extend_byte_vec_fp16(&self, v: &mut Vec<u8>);
}

impl Float16ConversionSupportedElement for f32 {
    fn from_fp16_bytes(bytes: &[u8]) -> Self {
        let sign = ((bytes[1] & 0x80) as u32) << 24; // 0b10000000   single bit of sign
        let exponent = (bytes[1] & 0x7C) >> 2;  //  5 bits of exponent
        let fraction: u32 = (((bytes[1] & 0x3) as u32) << 8) | (bytes[0] as u32);   // 10 bits of fraction
        
        if exponent == 0 {
            if fraction == 0 {
                // zero
                f32::from_bits(sign)
            } else {
                // rebias subnormal numbers to normalized numbers
                let e = fraction.leading_zeros() - 22;
                let exp = (127 - 15 - e) << 23;
                let new_frac = (fraction << (14 + e)) & 0x7FFFFF;
                f32::from_bits((sign << 24)| exp | new_frac)
            }
        } else if exponent == 0x1F {
            let bits = sign | 0x7F800000; // full 1 for exponents
            f32::from_bits(bits | (fraction << 13))   // keep fraction and fill zeros for remaining bits.
        } else {
            let exponent = (exponent as u32) + 127 - 15;    // adjust exponent to have 8 bits
            f32::from_bits(sign | (exponent << 23) | (fraction << 13))
        }
    }

    fn extend_byte_vec_fp16(&self, v: &mut Vec<u8>) {
        let bits = self.to_bits();
        let sign = ((bits >> 24) & 0x80) as u8;
        let exponent = (bits >> 23) & 0xFF;
        let fraction = bits & 0x007FFFFF;

        let frac16 = (fraction >> 13) as u16;
        
        if exponent == 0 {
            // zero or subnormal, return fp16 as 0
            if fraction == 0 {
                // zero
                v.extend_from_slice(&[0x0, sign]);
            } else {
                v.extend_from_slice(&[(frac16 & 0xFF) as u8, sign | ((frac16 >> 8) as u8)]);
            }
        } else if exponent == 0xFF {
            // inf or nan
            if fraction == 0 {
                // inf
                v.extend_from_slice(&[0x0, sign | 0x7C]);
            } else {
                v.extend_from_slice(&[(frac16 & 0xFF) as u8, sign | 0x7C | ((frac16 >> 8) as u8)]);
            }
        } else { 
            let rounded = fraction & 0x1FFF;
            let round = if rounded > 0x1000 {
                1
            } else if rounded < 0x1000 {
                0
            } else {
                frac16 & 1
            };
            let mut frac16 = frac16 + round;
            let mut exp16 = (exponent - (127 - 15)) as u8;
            // fraction part is larger than 10 bits => shift right and adjust exponent
            if frac16 > 0x3FF {
                frac16 >>= 1;
                exp16 += 1;
            }
            
            if exp16 > 0x1F {
                // the number is too large to be represented in fp16. Represented it as inf
                v.extend_from_slice(&[0x0, sign | 0x7C]);
            } else {
                let b1 = sign | (exp16 << 2) | (((frac16 >> 8) & 0x3) as u8);
                v.extend_from_slice(&[(frac16 & 0xFF) as u8, b1]);
            }
        }
    }
}


/// Element type traits that can be used to load/save 
/// [Brain float point 16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) numbers.
/// 
/// All APIs are experimental
pub trait BFloat16ConversionSupportedElement where Self: CommonSupportedElement + ndarray::NdFloat {
    /// Create an instance from fp16 little-endian bytes
    fn from_bf16_bytes(bytes: &[u8]) -> Self;

    /// Conver the value as fp16 and save it to the buffer.
    fn extend_byte_vec_bf16(&self, v: &mut Vec<u8>);
}

impl BFloat16ConversionSupportedElement for f32 {
    fn from_bf16_bytes(bytes: &[u8]) -> Self {
        // padding zeros in the end to get fp32
        f32::from_le_bytes([0x0, 0x0, bytes[0], bytes[1]])
    }

    fn extend_byte_vec_bf16(&self, v: &mut Vec<u8>) {
        let bits = self.to_bits();
        let sign = ((bits & 0x8000_0000) >> 24) as u8;
        let mut exponent = ((bits & 0x7f80_0000) >> 23) as u16;
        let fraction_cut_off = bits & 0xFFFF;
        let fraction_keep = (bits & 0x7F0000) >> 16;
        let round = if fraction_cut_off > 0x8000 {
            1
        } else if fraction_cut_off < 0x8000 {
            0
        } else {
            fraction_keep & 1
        };

        let mut new_frac = fraction_keep + round;
        if new_frac > 0x7F {
            new_frac >>= 1;
            exponent += 1;
            // exceeding the upper bound due to rounding, return inf
            if exponent >= 0x100 {
                v.extend_from_slice(&[0x80, sign | 0x7F]);
                return;
            }
        }
        let exponent = (exponent & 0xFF) as u8; 
        v.extend_from_slice(&[(new_frac as u8) | ((exponent & 1) << 7), (exponent >> 1) | sign]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    const F16_EPS: f32 = 6e-5;

    #[test]
    pub fn test_load_fp16_to_f32(){
        let test_cases = [
            (1.0f32, [0x0u8, 0x3Cu8]),
            (-1.0f32, [0x0u8, 0xBCu8]),
            (0.3333, [0x55, 0x35]),
            (0.9995, [0xff, 0x3b]),
            (65504.0, [0xff, 0x7b]),
            (-0.0, [0x0, 0x80]),
            (-2.0, [0x0, 0xC0]),
            (6.1e-5, [0xff, 0x3]),
            (6.1e-5, [0x00, 0x4]),
            (0.0, [0x1, 0]),
            (3.140625, [0x48, 0x42])
        ];

        for (expected, bytes) in test_cases {
            assert_approx_eq!(f32::from_fp16_bytes(&bytes), expected, F16_EPS);
        }
        // test inf
        assert_eq!(f32::from_fp16_bytes(&[0x0, 0x7C]), f32::INFINITY);
        assert_eq!(f32::from_fp16_bytes(&[0x0, 0xFC]), f32::NEG_INFINITY);
    }

    #[test]
    pub fn test_load_bf6_to_f32(){
        let test_cases:[(f32, [u8;2]); 4] = [
            (1.0f32, [0x80, 0x3F]),
            (-2.0f32, [0x0, 0xC0]),
            (3.140625, [0x49, 0x40]),
            (0.334, [0xAB, 0x3E]),
        ];

        for (expected, bytes) in test_cases {
            assert_approx_eq!(f32::from_bf16_bytes(&bytes), expected, F16_EPS);
        }
        // test inf
        assert_eq!(f32::from_bf16_bytes(&[0x80, 0x7F]), f32::INFINITY);
        assert_eq!(f32::from_bf16_bytes(&[0x80, 0xFF]), f32::NEG_INFINITY);
        assert!(f32::is_nan(f32::from_bf16_bytes(&[0xC1, 0xFF])));
    }


}