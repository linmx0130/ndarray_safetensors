//! Serialize / deserialize [Rust ndarray](https://docs.rs/ndarray/latest/ndarray/) with [Safetensors](https://huggingface.co/docs/safetensors/en/index).
//! 
//! ## Main APIs
//! - **[`TensorViewWithDataBuffer`]**: `TensorView` implementation accepted by Safetensors crate, but it owns the data and can be easily created from ndarray.
//! - **[`parse_tensor_view_data`]**: Parse a TensorView from Safetensors into ndarray.
//! 
//! ## Demo
//! 
//! ```
//! # use ndarray::array;
//! # use ndarray_safetensors::{TensorViewWithDataBuffer, parse_tensors};
//! // Serailize ndarrays
//! let arr = array![[1.0, -1.0], [2.0, -2.0]];
//! let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
//! let serialized_data = safetensors::serialize(data, &None).unwrap();
//! 
//! // Deserialize ndarrays
//! let tensors = safetensors::SafeTensors::deserialize(&serialized_data).unwrap();
//! let arrays = parse_tensors::<f64>(&tensors).unwrap();
//! let (name, array) = &arrays[0];
//! assert_eq!(name, "arr");
//! assert_eq!(array[[1,1]], -2.0);
//! 
//! // Deserialize with a wrong type hint
//! let parse_with_wrong_type = parse_tensors::<f32>(&tensors);
//! assert!(parse_with_wrong_type.is_err())
//! ```
//! 
//! ## License
//! Copyright (c) 2024, Mengxiao Lin. The crate is published under MIT License.

use safetensors;
use ndarray;
use std::{borrow::Cow, mem::size_of};
use std::error::Error;

/// A data structure like `TensorView` in safetensors, but it owns the data
pub struct TensorViewWithDataBuffer{
    buf: Vec<u8>,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
}

impl TensorViewWithDataBuffer {
    /// Create a standard TensorView from this buffer. No copy of data occurs.
    pub fn to_tensor_view<'data>(&'data self) -> safetensors::tensor::TensorView<'data> {
        safetensors::tensor::TensorView::new(
            self.dtype,
            self.shape.clone(), 
            self.buf.as_ref()
        ).unwrap()
    }

    /// Create a new TensorViewWithDataBuffer object from a ndarray.
    /// 
    /// Notes: Copy of the data occurs once. The original array won't be consumed.
    pub fn new<A, S, D>(array: &ndarray::ArrayBase<S, D>) -> TensorViewWithDataBuffer
        where 
            A: CommonSupportedElement, 
            S: ndarray::Data<Elem = A>,
            D:ndarray::Dimension 
    {
        let shape = Vec::from(array.shape());
        // convert the tensor to one dim array with row-major (C style)
        let one_dim_array = array.to_shape(
            ((array.len(),), ndarray::Order::RowMajor)
        ).unwrap();
        let v = one_dim_array.to_vec();
        let mut buf: Vec<u8> = Vec::with_capacity(size_of::<A>() * v.len());
        for value in v {
            value.extend_byte_vec(&mut buf);
        }

        TensorViewWithDataBuffer {
            dtype: A::safetensors_dtype(),
            shape,
            buf
        }
    }
}

impl<'data> safetensors::View for TensorViewWithDataBuffer {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self)-> Cow<'_, [u8]> {
        Cow::Borrowed(self.buf.as_ref())
    }
    fn data_len(&self) -> usize {
        self.buf.len()
    }
}

/// Element type traits for f32 and f64, which is supported by both ndarray and safetensors
pub trait CommonSupportedElement: Clone + ndarray::NdFloat {
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

/// Error to emit if the type doesn't match for parsing a tensor view
#[derive(Debug, Clone, Copy)]
pub struct TypeMismatchedError {
    expected_type: safetensors::Dtype,
    actual_type: safetensors::Dtype
}
impl Error for TypeMismatchedError {}
impl std::fmt::Display for TypeMismatchedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Expected data type to be {:?}, but found {:?}", self.expected_type, self.actual_type)
    }
}

/// Deserialized a Safetensors View as a ndarray
/// 
/// The return ndarray will own the data. If the data type of the tensor doesn't match with `A`, 
/// a [`TypeMismatchedError`] will be returned.
pub fn parse_tensor_view_data<A>(view: &safetensors::tensor::TensorView) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<ndarray::IxDynImpl>>, TypeMismatchedError>  
    where 
        A: CommonSupportedElement,
{
    if A::safetensors_dtype() != view.dtype() {
        return Err(TypeMismatchedError{
            expected_type: A::safetensors_dtype(),
            actual_type: view.dtype()
        });
    }
    let dtype_size = size_of::<A>();
    let data = view.data();
    let shape = Vec::from(view.shape());
    let mut values: Vec<A> = Vec::with_capacity(data.len() / dtype_size);
    for idx in (0..data.len()).step_by(dtype_size) {
        values.push(A::from_bytes(&data[idx..(idx+dtype_size)]))
    }
    let array = ndarray::Array::from_vec(values);
    Ok(array.into_shape_with_order(shape).unwrap())
}

/// Deserialize safetensors to ndarrays of the same element type and dimension type.
pub fn parse_tensors<A>(tensors: &safetensors::SafeTensors) -> Result<Vec<(String, ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<ndarray::IxDynImpl>>)>, TypeMismatchedError>
    where
        A: CommonSupportedElement,
{
    let mut arrays = Vec::with_capacity(tensors.len());

    for (name, tensor) in tensors.iter() {
        let array = parse_tensor_view_data::<A>(&tensor)?;
        arrays.push((String::from(name), array)); 
    }

    Ok(arrays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    pub fn test_serialize_and_deserialize_f32(){
        let x = ndarray::array![[1.0f32, -1.0f32], [2.0f32, -2.0f32]];
        let y = ndarray::array![3.14f32, 2.718f32];
        let data = vec![("x", TensorViewWithDataBuffer::new(&x)), ("y", TensorViewWithDataBuffer::new(&y))];
        let serialized_data = safetensors::serialize(data, &None).unwrap();
        let deserialized_data = parse_tensors::<f32>(&safetensors::SafeTensors::deserialize(&serialized_data).unwrap()).unwrap();

        let mut data_map = HashMap::new();
        deserialized_data.iter().for_each(|(name, array)| { data_map.insert(name.clone(), array); });
        let d_x = data_map.get("x").unwrap().to_shape([2, 2]).unwrap().to_owned();
        let d_y = data_map.get("y").unwrap().to_shape([2]).unwrap().to_owned();
        
        assert_eq!(x, d_x);
        assert_eq!(y, d_y);
    }
}