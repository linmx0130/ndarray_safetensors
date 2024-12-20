use safetensors;
use ndarray;
use std::{borrow::Cow, mem::size_of};

pub struct TensorViewWithDataBuffer{
    buf: Vec<u8>,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
}

impl TensorViewWithDataBuffer {
    pub fn to_tensor_view<'data>(&'data self) -> safetensors::tensor::TensorView<'data> {
        safetensors::tensor::TensorView::new(
            self.dtype,
            self.shape.clone(), 
            self.buf.as_ref()
        ).unwrap()
    }

    pub fn new<A, S, D>(array: &ndarray::ArrayBase<S, D>) -> TensorViewWithDataBuffer
        where 
            A: Clone + ndarray::NdFloat + CommonSupportedElement, 
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

pub trait CommonSupportedElement {
    fn extend_byte_vec(&self, v: &mut Vec<u8>);
    fn safetensors_dtype() -> safetensors::Dtype;
}

impl CommonSupportedElement for f32 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    } 
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::F32
    }   
}

impl CommonSupportedElement for f64 {
    fn extend_byte_vec(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.to_le_bytes());
    }
    fn safetensors_dtype() -> safetensors::Dtype {
        safetensors::Dtype::F64
    }    
}
