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
mod element;

pub use crate::element::{CommonSupportedElement, Float16ConversionSupportedElement, BFloat16ConversionSupportedElement};
use ndarray::{self, ShapeBuilder};
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
    pub fn to_tensor_view(&self) -> safetensors::tensor::TensorView<'_> {
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
            D: ndarray::Dimension 
    {
        let shape = Vec::from(array.shape());
        // convert the tensor to one dim array with row-major (C style)
        let one_dim_array = array.to_shape(
            ((array.len(),), ndarray::Order::RowMajor)
        ).unwrap();

        let buf = if cfg!(all(target_endian = "little", feature = "unsafe_copy")) {
            // Directly copy the raw data on little endian machines
            let v = one_dim_array.to_vec();
            let raw_ptr = v.as_ptr();
            unsafe {
                let u8_ptr = raw_ptr as *mut u8;
                let length = one_dim_array.len() * size_of::<A>();
                let layout = std::alloc::Layout::array::<u8>(length).unwrap();
                let buf_ptr = std::alloc::alloc(layout);
                if buf_ptr.is_null() {
                    panic!("Error in allocating memory for new tensor views");
                }
                std::ptr::copy_nonoverlapping(u8_ptr, buf_ptr, length);
                Vec::from_raw_parts(buf_ptr, length, length)
            }
        } else {
            let v = one_dim_array.to_vec();
            let mut buf: Vec<u8> = Vec::with_capacity(size_of::<A>() * v.len());
            for value in v {
                value.extend_byte_vec(&mut buf);
            }
            buf
        };

        TensorViewWithDataBuffer {
            dtype: A::safetensors_dtype(),
            shape,
            buf
        }
    }

    /// Create a new TensorViewWithDataBuffer object in FP16 from a ndarray.
    pub fn new_fp16<A, S, D>(array: &ndarray::ArrayBase<S, D>) -> TensorViewWithDataBuffer
        where 
            A: Float16ConversionSupportedElement, 
            S: ndarray::Data<Elem = A>,
            D: ndarray::Dimension 
    {
        let shape = Vec::from(array.shape());
        // convert the tensor to one dim array with row-major (C style)
        let one_dim_array: ndarray::ArrayBase<ndarray::CowRepr<'_, A>, ndarray::Dim<[usize; 1]>> = array.to_shape(
            ((array.len(),), ndarray::Order::RowMajor)
        ).unwrap();
        let v = one_dim_array.to_vec();
        let mut buf: Vec<u8> = Vec::with_capacity(v.len() * 2);
        for value in v {
            value.extend_byte_vec_fp16(&mut buf);
        }

        TensorViewWithDataBuffer {
            dtype: safetensors::Dtype::F16,
            shape,
            buf
        }
    }

    /// Create a new TensorViewWithDataBuffer object in BF16 from a ndarray.
    pub fn new_bf16<A, S, D>(array: &ndarray::ArrayBase<S, D>) -> TensorViewWithDataBuffer
        where 
            A: BFloat16ConversionSupportedElement, 
            S: ndarray::Data<Elem = A>,
            D: ndarray::Dimension 
    {
        let shape = Vec::from(array.shape());
        // convert the tensor to one dim array with row-major (C style)
        let one_dim_array = array.to_shape(
            ((array.len(),), ndarray::Order::RowMajor)
        ).unwrap();
        let v = one_dim_array.to_vec();
        let mut buf: Vec<u8> = Vec::with_capacity(v.len() * 2);
        for value in v {
            value.extend_byte_vec_bf16(&mut buf);
        }

        TensorViewWithDataBuffer {
            dtype: safetensors::Dtype::BF16,
            shape,
            buf
        }
    }


}

impl safetensors::View for TensorViewWithDataBuffer {
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


/// Error to emit if the type doesn't match for parsing a tensor view
#[derive(Debug, Clone)]
pub enum DeserializationError {
    TypeMismatchedError { expected_type: safetensors::Dtype, actual_type: safetensors::Dtype},
    ShapeMismatchedError { expected_shape: Vec<usize>, actual_shape: Vec<usize> }
}
impl Error for DeserializationError {}
impl std::fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatchedError { expected_type, actual_type } => {
                write!(f, "Expected data type to be {:?}, but found {:?}", expected_type, actual_type)
            }
            Self::ShapeMismatchedError { expected_shape, actual_shape } => {
                write!(f, "Expected data shape to be {:?}, but found {:?}", expected_shape, actual_shape)
            }
        }
    }
}

/// Deserialized a Safetensors View as a ndarray
/// 
/// The return ndarray will own the data. If the data type of the tensor doesn't match with `A`, 
/// a [`DeserializationError::TypeMismatchedError`] will be returned.
pub fn parse_tensor_view_data<A>(view: &safetensors::tensor::TensorView) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<ndarray::IxDynImpl>>, DeserializationError>  
    where 
        A: CommonSupportedElement,
{
    if A::safetensors_dtype() != view.dtype() {
        return Err(DeserializationError::TypeMismatchedError{
            expected_type: A::safetensors_dtype(),
            actual_type: view.dtype()
        });
    }
    let data = view.data();
    let shape = Vec::from(view.shape());

    let values = parse_data_from_u8_slice(data);
    let array = ndarray::ArrayBase::from_shape_vec(shape, values).unwrap();
    Ok(array)
}

/// Clone the data from a u8 slice into a vec of type A.
/// The u8 slice should be little-endian representation of an array of type A elements.
#[inline]
fn parse_data_from_u8_slice<A>(data: &[u8]) -> Vec<A> 
where A: CommonSupportedElement
{
   let dtype_size = size_of::<A>();
   if cfg!(all(target_endian="little", feature = "unsafe_copy")) {
        let length = data.len() / dtype_size;
        let layout = std::alloc::Layout::array::<A>(length).unwrap();
        unsafe{
            let data_ptr = data.as_ptr() as *const A;
            let buf_ptr = std::alloc::alloc(layout) as *mut A;
            if buf_ptr.is_null() {
                panic!("Error in allocating memory for new ndarray");
            }
            std::ptr::copy_nonoverlapping(data_ptr, buf_ptr, length);
            Vec::from_raw_parts(buf_ptr, length, length)
        }
    } else {
        let mut values: Vec<A> = Vec::with_capacity(data.len() / dtype_size);
        for idx in (0..data.len()).step_by(dtype_size) {
            values.push(A::from_bytes(&data[idx..(idx+dtype_size)]))
        }
        values
    }
}

/// Deserialized a Safetensors View as a ndarray with a known dimension.
/// 
/// The return ndarray will own the data. The shape type will be inferred based on the provided dimension,
/// instead of using the dynamic shape.
/// 
/// If the data type of the tensor doesn't match with `A`, a [`DeserializationError::TypeMismatchedError`] will be returned.
/// If the shape of the tnesor doesn't match the provided dimension, a [`DeserializationError::ShapeMismatchedError`] will be returned.
/// 
/// Example:
/// ```
/// # use ndarray::array;
/// # use ndarray::Array2;
/// # use ndarray_safetensors::{TensorViewWithDataBuffer, parse_tensor_view_data_with_dimension};
/// // Serailize ndarrays
/// let arr = array![[1.0, -1.0], [2.0, -2.0]];
/// let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
/// let serialized_data = safetensors::serialize(data, &None).unwrap();
/// 
/// // Deserialize ndarrays with type and dimension hint.
/// let tensors = safetensors::SafeTensors::deserialize(&serialized_data).unwrap();
/// let array :Array2<f64>  = parse_tensor_view_data_with_dimension(&tensors.tensor("arr").unwrap(), (2, 2)).unwrap();
/// assert_eq!(array[[0,0]], 1.0);
/// assert_eq!(array[[0,1]], -1.0);
/// assert_eq!(array[[1,0]], 2.0);
/// assert_eq!(array[[1,1]], -2.0);
/// 
/// // Wrong dimension.
/// assert!(parse_tensor_view_data_with_dimension::<f64,_,_>(&tensors.tensor("arr").unwrap(), (1, 4)).is_err())
/// ```
pub fn parse_tensor_view_data_with_dimension<A, D, ID>(view: &safetensors::tensor::TensorView, dim: ID) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<A>, D>, DeserializationError>
    where
        A: CommonSupportedElement,
        D: ndarray::Dimension,
        ID: ndarray::IntoDimension<Dim = D>,
{
    if A::safetensors_dtype() != view.dtype() {
        return Err(DeserializationError::TypeMismatchedError{
            expected_type: A::safetensors_dtype(),
            actual_type: view.dtype()
        });
    }
    let dim = dim.into_dimension();
    let hinted_shape= Vec::from(dim.as_array_view().as_slice().unwrap());
    let actual_shape = Vec::from(view.shape());
    if hinted_shape != actual_shape {
        return Err(DeserializationError::ShapeMismatchedError {
            expected_shape: hinted_shape,
            actual_shape 
        })
    }
    
    let data = view.data();
    let values: Vec<A> = parse_data_from_u8_slice(data);
    let decode_shape = ndarray::Shape::from(dim).set_f(false);
    let array = ndarray::ArrayBase::from_shape_vec(decode_shape, values).unwrap();
    Ok(array)
}

/// Deserialize safetensors to ndarrays of the same element type and dimension type.
pub fn parse_tensors<A>(tensors: &safetensors::SafeTensors) -> Result<Vec<(String, ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<ndarray::IxDynImpl>>)>, DeserializationError>
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

/// Deserialized a IEEE 754 FP16 Safetensors View as a ndarray.
/// 
/// The return ndarray will own the data. If the data type of the tensor is not FP16, 
/// a [`DeserializationError::TypeMismatchedError`] will be returned.
/// 
/// This API is experimental.
pub fn parse_fp16_tensor_view_data<A>(view: &safetensors::tensor::TensorView) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<ndarray::IxDynImpl>>, DeserializationError>  
    where 
        A: Float16ConversionSupportedElement,
{
    if view.dtype() != safetensors::Dtype::F16 {
        return Err(DeserializationError::TypeMismatchedError{
            expected_type: safetensors::Dtype::F16,
            actual_type: view.dtype()
        });
    }
    let dtype_size = size_of::<A>();
    let data = view.data();
    let shape = Vec::from(view.shape());
    let mut values: Vec<A> = Vec::with_capacity(data.len() / dtype_size);
    for idx in (0..data.len()).step_by(2) {
        values.push(A::from_fp16_bytes(&data[idx..(idx+2)]))
    }
    let array = ndarray::ArrayBase::from_shape_vec(shape, values).unwrap();
    Ok(array)
}

/// Deserialized a [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) Safetensors View as a ndarray
/// 
/// The return ndarray will own the data. If the data type of the tensor is not BF16, 
/// a [`DeserializationError::TypeMismatchedError`] will be returned.
/// 
/// This API is experimental.
pub fn parse_bf16_tensor_view_data<A>(view: &safetensors::tensor::TensorView) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<ndarray::IxDynImpl>>, DeserializationError>  
    where 
        A: BFloat16ConversionSupportedElement,
{
    if view.dtype() != safetensors::Dtype::BF16 {
        return Err(DeserializationError::TypeMismatchedError{
            expected_type: safetensors::Dtype::BF16,
            actual_type: view.dtype()
        });
    }
    let dtype_size = size_of::<A>();
    let data = view.data();
    let shape = Vec::from(view.shape());
    let mut values: Vec<A> = Vec::with_capacity(data.len() / dtype_size);
    for idx in (0..data.len()).step_by(2) {
        values.push(A::from_bf16_bytes(&data[idx..(idx+2)]))
    }
    let array = ndarray::ArrayBase::from_shape_vec(shape, values).unwrap();
    Ok(array)
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

    #[test]
    pub fn test_deserialize_f16_data() {
        let buf: Vec<u8> = vec![0x48, 0x42, 0x0, 0x3C, 0x0, 0xC0];
        let tensor_view = TensorViewWithDataBuffer {
            dtype: safetensors::Dtype::F16,
            buf: buf,
            shape: vec![3]
        };
        let arr = parse_fp16_tensor_view_data::<f32>(&tensor_view.to_tensor_view()).unwrap();
        assert_eq!(arr[0], 3.140625);
        assert_eq!(arr[1], 1.0);
        assert_eq!(arr[2], -2.0);
    }

    #[test]
    pub fn test_serialize_deserialize_bf16_data() {
        let data = ndarray::array![[0.1, 0.2, 1.0], [3.0, 4.0, -2.0],  [0.0, -1.0, 3.14]];
        let tensor_view = TensorViewWithDataBuffer::new_bf16(&data);
        let arr = parse_bf16_tensor_view_data::<f32>(&tensor_view.to_tensor_view()).unwrap();
        for i in 0..3usize {
            for j in 0..3usize {
                assert_approx_eq::assert_approx_eq!(data[[i,j]], arr[[i,j]], 1e-3);
            }
        }

        let special = ndarray::array![f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let tensor_view = TensorViewWithDataBuffer::new_bf16(&special);
        let sarr = parse_bf16_tensor_view_data::<f32>(&tensor_view.to_tensor_view()).unwrap();
        assert_eq!(sarr[0], f32::INFINITY);
        assert_eq!(sarr[1], f32::NEG_INFINITY);
        assert!(sarr[2].is_nan()); 
    }
     
    #[test]
    pub fn serialize_noncontiguous_data() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let slice = data.slice(ndarray::s![.., 1..]);
        let tensor_view = TensorViewWithDataBuffer::new(&slice);
        let deserialized:ndarray::Array2<f64> = parse_tensor_view_data_with_dimension(&(tensor_view.to_tensor_view()), (2, 1)).unwrap();
        assert_eq!(slice, deserialized);
    }
}