ndarray-safetensors
=====
Serialize / deserialize [Rust ndarray](https://docs.rs/ndarray/latest/ndarray/) with [Safetensors](https://huggingface.co/docs/safetensors/en/index).

## Demo
See `safetensors-testapp` for more details. Here is an end-to-end demo to print all tensors from a safetensors file.

```rust
use ndarray_safetensors::parse_tensor_view_data;

let path_example = Path::new("data/rand.safetensors");
let file = File::open(path_example).unwrap();
let buffer = unsafe {MmapOptions::new().map(&file).unwrap()};
let tensors = safetensors::SafeTensors::deserialize(&buffer).unwrap();
for (name, tensor_view) in tensors.tensors() {
    println!("Tensor: {}", name);
    if tensor_view.dtype() == safetensors::Dtype::F32 {
        let arr= parse_tensor_view_data::<f32>(&tensor_view).unwrap();
        println!("{}", arr);
    }
    if tensor_view.dtype() == safetensors::Dtype::F64 {
        let arr= parse_tensor_view_data::<f64>(&tensor_view).unwrap();
        println!("{}", arr);
    }
}
```

## Features
* `unsafe_copy`: Allowing directly copying the bytes instead of parsing floats in serialization/deserialization for little-endian machines.

## Copyright & License
Copyright (c) 2024-2025 Mengxiao Lin. Check LICENSE file.
