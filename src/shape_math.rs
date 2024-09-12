use candle_nn::Conv2d;
use crate::{ConvShape, MaxPoolConfig, BATCH_SIZE};


pub fn compute_conv2d_output_shape(
    h_in: usize,
    w_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_channels: usize,
) -> ConvShape {
    let shape_out = |shape_in| (shape_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    let h_out = shape_out(h_in);
    let w_out = shape_out(w_in);

    ConvShape {
        n: BATCH_SIZE,
        c: out_channels,
        h: h_out,
        w: w_out,
    }
}


pub fn compute_max_pool_output_shape(
    h_in: usize,
    w_in: usize,
    max_pool_config: &MaxPoolConfig,
) -> (usize, usize) {
    let shape_out = |shape_in| (shape_in + 2 * max_pool_config.padding - (max_pool_config.pool_size - 1) - 1) / max_pool_config.stride + 1;
    let h_out = shape_out(h_in);
    let w_out = shape_out(w_in);

    (h_out, w_out)
}
