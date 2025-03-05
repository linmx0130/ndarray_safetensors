#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128, __m128i, _mm_cvtph_ps, _mm_set_epi8,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128, __m128i, _mm_cvtph_ps, _mm_set_epi8,
};

/// Create f32 vec from fp16 bytes with SSE instructions.
///
/// The conversion is done by `_mm_cvtph_ps` instruction, which can decode 
/// 4 numbers within a single instruction.
pub(crate) fn create_f32_vec_from_fp16_bytes(bytes: &[u8]) -> Vec<f32> {
    let mut values: Vec<f32> = Vec::with_capacity(bytes.len() / 2);
    let n = bytes.len();
    let mod_8 = n % 8;

    for i in (0..(n - mod_8)).step_by(8) {
        unsafe {
            let pack : __m128i = _mm_set_epi8(
                0, 0, 0, 0,
                0, 0, 0, 0,
                bytes[i + 7] as i8, bytes[i + 6] as i8, bytes[i + 5] as i8, bytes[i + 4] as i8,
                bytes[i + 3] as i8, bytes[i + 2] as i8, bytes[i + 1] as i8, bytes[i] as i8
            );
            let f32_pack: __m128 = _mm_cvtph_ps(pack);
            let f32values_arr = std::mem::transmute::<__m128, [f32; 4]>(f32_pack);            
            values.push(f32values_arr[0]);
            values.push(f32values_arr[1]);
            values.push(f32values_arr[2]);
            values.push(f32values_arr[3]);
        }
    }
    for idx in ((n - mod_8).. n).step_by(2) {
        unsafe {
            let value :__m128i = _mm_set_epi8(
                0, 0, 0, 0, 
                0, 0, 0, 0, 
                0, 0, 0, 0,
                0, 0, bytes[idx+1] as i8, bytes[idx] as i8
            );
            let f32values: __m128 = _mm_cvtph_ps(value);
            let f32values_arr = std::mem::transmute::<__m128, [f32; 4]>(f32values);            
            values.push(f32values_arr[0]);
        }
    }
    values
}