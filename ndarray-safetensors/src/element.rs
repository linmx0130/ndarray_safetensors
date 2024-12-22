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