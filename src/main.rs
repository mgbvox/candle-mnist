mod shape_math;

use std::fmt::Formatter;
use candle_core::{Device, Tensor, Result, DType, IndexOp, ModuleT};
use candle_nn::{Conv2d, Dropout, Linear, Module, VarBuilder, Sequential, VarMap, Conv2dConfig};
use anyhow;
use candle_core::op::Op;

const MODEL_DTYPE: DType = DType::F32;
const BATCH_SIZE: usize = 64;

fn default_device() -> Device {
    Device::new_metal(0).expect("Unable to create metal device.")
}


#[derive(Debug)]
pub struct CNNLayer {
    conv: Conv2d,
    n_max_pool: usize,
    dropout: Dropout,
    train: bool,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ConvShape {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

impl ConvShape {
    fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self {
            n: n,
            c: c,
            h: h,
            w: w,
        }
    }
}

#[derive(Debug)]
struct MaxPoolConfig {
    pool_size: usize,
    padding: usize,
    stride: usize,
}

impl MaxPoolConfig {
    /// accept pool_size and use smart defaults for the rest
    fn new(pool_size: usize) -> Self {
        Self {
            pool_size: pool_size,
            // typically 0 for max pool
            padding: 0,
            // typically stride == pool size
            stride: pool_size,
        }
    }
}

#[derive(Debug)]
pub struct Conv2dPooled {
    conv: Conv2d,
    in_shape: ConvShape,
    pre_pool_shape: ConvShape,
    out_shape: ConvShape,
}


impl Conv2dPooled {
    fn new<M: Into<Option<MaxPoolConfig>>, C: Into<Option<Conv2dConfig>>>(
        in_shape: ConvShape,
        out_channels: usize,
        kernel_size: usize,
        max_pool_cfg: M,
        cfg: C,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = cfg.into().unwrap_or_else(|| Conv2dConfig::default());


        let in_channels = in_shape.c;

        let conv = candle_nn::conv2d(
            in_channels,
            out_channels,
            kernel_size,
            cfg,
            vb.push_prefix("Conv2D[k:{}]"),
        )?;

        // get updated config post-create
        let cfg = conv.config();

        let pre_pool_shape = shape_math::compute_conv2d_output_shape(
            in_shape.h,
            in_shape.w,
            // non-square not supported yet, so just broadcast
            kernel_size,
            cfg.stride,
            cfg.padding,
            cfg.dilation,
            out_channels,
        );

        // Apply max pooling if requested

        let final_shape = if let Some(max_pool_cfg) = max_pool_cfg.into() {
            let pooled_shape = shape_math::compute_max_pool_output_shape(
                pre_pool_shape.h,
                pre_pool_shape.w,
                &max_pool_cfg,
            );
            ConvShape {
                n: pre_pool_shape.n,
                c: pre_pool_shape.c,
                h: pooled_shape.0,
                w: pooled_shape.1,
            }
        } else {
            pre_pool_shape.clone()
        };

        Ok(
            Self {
            conv: conv,
            in_shape: in_shape,
            pre_pool_shape: pre_pool_shape,
            out_shape: final_shape,
        })
    }
}

fn log_shape(t: &Tensor) {
    println!("{:?}", t)
}


impl Module for Conv2dPooled {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // get batch size and image dimension
        todo!()
        // log_shape(&xs);
        // let xs = xs
        //     .apply(&self.conv)?
        //     .max_pool2d(&self.n_max_pool)?;
        // log_shape(&xs);
        // // let xs = xs.apply(&self.linear)?;
        // // log_shape(&xs);
        // // let xs = xs
        // //     .apply(&self.linear)?
        // //     .relu()?;
        // //
        // // log_shape(&xs);
        // //
        // // let xs = self.dropout.forward(&xs, self.train)?;
        // Ok(xs)
    }
}


pub fn build_net<const S: usize, const C: usize>(
    input_shape: (usize, usize, usize),
    in_channels: [usize; S],
    // last element of kernels will be discarded
    kernels: [usize; S],
    linear: [usize; C],
    // last element of dropout will be discarded

    n_classes: usize,
    vs: VarBuilder,
) -> Result<Sequential> {
    let mut seq = candle_nn::sequential::seq();

    // todo: implement dropout where desired.

    let (in_c, in_h, in_w) = input_shape;
    for idx in 0..S - 1 {
        let conv = candle_nn::conv2d(
            in_channels[idx],
            in_channels[idx + 1],
            kernels[idx],
            Default::default(),
            vs.push_prefix(format!("{}-conv2D", idx)),
        )?;

        // update c, h, w
        // (in_c, in_h, in_w) = shape_math::compute_conv2d_output_shape(
        //     &conv,
        //     in_h,
        //     in_w,
        // );

        seq = seq.add(conv);
    }

    // we'll flatten along c, h, w - that's the first linear input
    let mut linear_input = in_c * in_h * in_w;

    for idx in 0..C - 1 {
        seq = seq.add(
            candle_nn::linear(
                linear_input,
                linear[idx],
                vs.push_prefix(format!("{}-lin", idx)),
            )?
        );

        // each linear after the first can have any shape the user wants
        linear_input = linear[idx];
    }
    //
    // // classifier head
    // let classifier_head = candle_nn::linear(
    //     linear[S - 1],
    //     n_classes,
    //     vs.clone(),
    // )?;
    //
    // seq = seq.add(classifier_head);

    Ok(seq)
}


fn main() -> anyhow::Result<()> {
    let device = default_device();
    let varmap = VarMap::new();
    let vars = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);

    let net = build_net(
        // input shape
        (1, 64, 64),
        // layerwise filter count
        [1, 32, 64, 128],
        // layerwise kernel size
        [5, 5, 5, 5],
        // layerwise cnn linear output dims
        [128, 128, 128, 128],
        // n_classes
        10,
        // varbuilder
        vars,
    )?;

    // Create a random tensor of shape (N, C, H, W)
    let ztensor = Tensor::zeros((1, 1, 64, 64,), MODEL_DTYPE, &device)?;
    // make it rAndOM
    let random_tensor = ztensor.rand_like(0.0, 10.0)?;

    let out = net.forward(&random_tensor)?;
    println!("out: {:?}", out);

    // todo: MNIST via candle_datasets, figure out dim mismatch error, training loop, weight save/load

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sequential() -> anyhow::Result<()> {
        todo!()
    }

    #[test]
    fn test_output_shape() -> anyhow::Result<()> {
        let device = default_device();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);

        // test with parameters from this third-party calc
        // https://dingyan89.medium.com/calculating-parameters-of-convolutional-and-fully-connected-layers-with-keras-186590df36c6


        let conv = Conv2dPooled::new(
            ConvShape::new(BATCH_SIZE, 3, 32, 32),
            8,
            5,
            MaxPoolConfig::new(2),
            Conv2dConfig { padding: 0, stride: 1, ..Default::default() },
            vb,
        )?;

        let pre_pool_shape = ConvShape::new(BATCH_SIZE, 8, 28, 28);
        let out_shape = ConvShape::new(BATCH_SIZE, 8, 14, 14);

        assert_eq!(conv.out_shape, out_shape);
        assert_eq!(conv.pre_pool_shape, pre_pool_shape);

        Ok(())
    }

    #[test]
    fn test_with_no_pooling() -> anyhow::Result<()> {
        let device = default_device();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);

        // test with parameters from this third-party calc
        // https://dingyan89.medium.com/calculating-parameters-of-convolutional-and-fully-connected-layers-with-keras-186590df36c6

        let cfg = Conv2dConfig { padding: 0, stride: 1, ..Default::default() };
        let conv = Conv2dPooled::new(
            ConvShape::new(BATCH_SIZE, 3, 32, 32),
            8,
            5,
            None,
            cfg,
            vb,
        )?;

        let pre_pool_shape = ConvShape::new(BATCH_SIZE, 8, 28, 28);

        assert_eq!(conv.out_shape, pre_pool_shape);
        assert_eq!(conv.pre_pool_shape, conv.out_shape);

        Ok(())
    }
}
