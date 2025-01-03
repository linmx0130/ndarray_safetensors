use ndarray;
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
pub trait Float16ConversionSupportedElement where Self: CommonSupportedElement + ndarray::NdFloat {
    /// Create an instance from fp16 little-endian bytes
    fn from_fp16_bytes(bytes: &[u8]) -> Self;
}

impl Float16ConversionSupportedElement for f32 {
    fn from_fp16_bytes(bytes: &[u8]) -> Self {
        let sign = ((bytes[1] & 0x80) as u32) << 24; // 0b10000000   single bit of sign
        let exponent = (bytes[1] & 0x7C) >> 2;  //  5 bits of exponent
        let fraction: u32 = (((bytes[1] & 0x3) as u32) << 8) | (bytes[0] as u32);   // 10 bits of fraction
        
        if exponent == 0 {
            return if fraction == 0 {
                // zero
                f32::from_bits(sign)
            } else {
                // rebias subnormal numbers to normalized numbers
                let e = fraction.leading_zeros() - 22;
                let exp = (127 - 15 - e) << 23;
                let new_frac = (fraction << (14 + e)) & 0x7FFFFF;
                f32::from_bits(((sign as u32) << 24)| exp | new_frac)
            }
        } else if exponent == 0x1F {
            let bits = sign | 0x7F800000; // full 1 for exponents
            f32::from_bits(bits | (fraction << 13))   // keep fraction and fill zeros for remaining bits.
        } else {
            let exponent = (exponent as u32) + 127 - 15;    // adjust exponent to have 8 bits
            f32::from_bits(sign | (exponent << 23) | (fraction << 13))
        }
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
}