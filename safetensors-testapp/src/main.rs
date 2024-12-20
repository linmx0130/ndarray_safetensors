use ndarray_safetensors::TensorViewWithDataBuffer;
use ndarray::{Array2, Array3, array, s};
use safetensors;
use std::path::Path;
use std::fs::File;
use memmap2::MmapOptions;

fn serailize_demo () {
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
    let path = Path::new("data/from_rust.safetensors");
    safetensors::serialize_to_file(data, &None, &path).unwrap();
    println!("Serialized as data/from_rust.safetensors");
}

fn deserialize_demo() {
    println!("Read data/rand.safetensors");
    let path_example = Path::new("data/rand.safetensors");
    let file = File::open(path_example).unwrap();
    let buffer = unsafe {MmapOptions::new().map(&file).unwrap()};
    let tensors = safetensors::SafeTensors::deserialize(&buffer).unwrap();
    for (name, tensor_view) in tensors.tensors() {
        println!("Tensor: {}", name);
        let arr = ndarray_safetensors::parse_tensor_view_data::<f32>(&tensor_view).unwrap();
        println!("{}", arr);
    }
}
fn main() {
    serailize_demo();
    println!("===");
    deserialize_demo();
}
