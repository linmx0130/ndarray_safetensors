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
    let buf: Vec<u8> = vec![0x48, 0x42, 0x0, 0x3C, 0x0, 0xC0, 0x48, 0x42, 0x0, 0x3C, 0x0, 0xC0];
    let tensor_view = TensorViewWithDataBuffer {
        dtype: safetensors::Dtype::F16,
        buf: buf,
        shape: vec![6]
    };
    let arr = parse_fp16_tensor_view_data::<f32>(&tensor_view.to_tensor_view()).unwrap();
    assert_eq!(arr[0], 3.140625);
    assert_eq!(arr[1], 1.0);
    assert_eq!(arr[2], -2.0);
    assert_eq!(arr[3], 3.140625);
    assert_eq!(arr[4], 1.0);
    assert_eq!(arr[5], -2.0);
}

#[test]
pub fn test_serialize_deserialize_f16_data() {
    let x = ndarray::array![3.140625f32, 1.0f32, -2.0f32];
    let tensor_view = TensorViewWithDataBuffer::new_fp16(&x);
    let arr = parse_fp16_tensor_view_data::<f32>(&tensor_view.to_tensor_view()).unwrap();

    assert_eq!(arr[0], 3.140625);
    assert_eq!(arr[1], 1.0);
    assert_eq!(arr[2], -2.0);
}

#[test]
pub fn test_serialize_deserialize_f16_data2() {
    let x = ndarray::array![1.0f32, 2.0f32, 4.0f32, 0.5f32];
    let tensor_view = TensorViewWithDataBuffer::new_fp16(&x);
    let arr = parse_fp16_tensor_view_data::<f32>(&tensor_view.to_tensor_view()).unwrap();

    assert_eq!(arr[0], 1.0f32);
    assert_eq!(arr[1], 2.0f32);
    assert_eq!(arr[2], 4.0f32);
    assert_eq!(arr[3], 0.5f32);
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
