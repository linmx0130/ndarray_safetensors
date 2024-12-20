use ndarray_safetensors::TensorViewWithDataBuffer;
use ndarray::{Array2, Array3, array, s};
use safetensors;
use std::path::Path;

fn main() {
    let arr64: Array2<f64> = array![[1., 2.], [3., 4.]];
    let arr32: Array3<f32> = array![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
    
    let tvb32 = TensorViewWithDataBuffer::new(&arr32);
    let tvb64 = TensorViewWithDataBuffer::new(&arr64);
    
    let slice = arr32.slice(s![1.., .., -1]);
    let tvb_slice = TensorViewWithDataBuffer::new(&slice);
    
    let data = vec![
        ("arr32", tvb32),
        ("arr64", tvb64),
        ("slice", tvb_slice)
    ];
    let path = Path::new("data/test.safetensors");
    safetensors::serialize_to_file(data, &None, &path).unwrap();
}
