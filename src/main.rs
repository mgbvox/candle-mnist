use candle_core::{Device, Tensor, Result, DType, IndexOp, ModuleT};
use candle_nn::{Conv2d, Dropout, Linear, Module, VarBuilder, Sequential, VarMap};
use anyhow;

const MODEL_DTYPE: DType = DType::F32;

fn default_device() -> Device {
    Device::new_metal(0).expect("Unable to create metal device.")
}


#[derive(Debug)]
pub struct CNNLayer {
    conv: Conv2d,
    linear: Linear,
    dropout: Dropout,
    train: bool,
}


impl CNNLayer {
    fn new(vs: VarBuilder, layer_name: String, cnn_in: usize, cnn_out: usize, kernel_size: usize, linear_in: usize, linear_out: usize, dropout: f32) -> Result<Self> {
        let conv = candle_nn::conv2d(
            cnn_in,
            cnn_out,
            kernel_size,
            Default::default(),
            vs.push_prefix(format!("{}-conv2D", layer_name)),
        )?;

        let linear = candle_nn::linear(
            linear_in,
            linear_out,
            vs.push_prefix(format!("{}-linear", layer_name)),
        )?;

        let dropout = candle_nn::Dropout::new(dropout);

        Ok(Self {
            conv,
            linear,
            dropout,
            train: false,
        })
    }
}

impl Module for CNNLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // get batch size and image dimension
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv)?
            .flatten_from(1)?
            .apply(&self.linear)?
            .relu()?;

        let xs = self.dropout.forward(&xs, self.train)?;
        Ok(xs)
    }
}


pub fn build_net<const S: usize>(
    filters: [usize; S],
    // last element of kernels will be discarded
    kernels: [usize; S],
    linear: [usize; S],
    // last element of dropout will be discarded
    dropout: [f32; S],
    n_classes: usize,
    vs: VarBuilder,
) -> Result<Sequential> {
    // if S <= 1 {
    //     return Errs(candle_core::Error::msg("S must be greater than 1".into()));
    // }

    let mut seq = candle_nn::sequential::seq();

    for idx in 0..S - 1 {
        let cnn = CNNLayer::new(
            vs.clone(),
            format!("CNN{}", idx),
            filters[idx],
            filters[idx + 1],
            kernels[idx],
            linear[idx],
            linear[idx + 1],
            dropout[idx],
        )?;

        seq = seq.add(cnn);
    }

    // classifier head
    let classifier_head = candle_nn::linear(
        linear[S - 1],
        n_classes,
        vs.clone(),
    )?;

    seq = seq.add(classifier_head);

    Ok(seq)
}


fn main() -> anyhow::Result<()> {
    let device = default_device();
    let varmap = VarMap::new();
    let vars = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);

    let net = build_net(
        // layerwise filter count
        [1, 32, 64, 128],
        // layerwise kernel size
        [5, 5, 5, 5],
        // layerwise cnn linear dims
        [128, 128, 128, 128],
        // layerwise dropout rate
        [0.1, 0.1, 0.1, 0.1],
        // n_classes
        10,
        // varbuilder
        vars,
    )?;

    // Create a random tensor of shape (1, 64, 64)
    let ztensor = Tensor::zeros((64, 64), MODEL_DTYPE, &device)?;
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
        let device = default_device();
        let varmap = VarMap::new();
        let vars = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);

        let net = build_net(
            // layerwise filter count
            [1, 32, 64, 128],
            // layerwise kernel size
            [5, 5, 5, 5],
            // layerwise cnn linear dims
            [128, 128, 128, 128],
            // layerwise dropout rate
            [0.1, 0.1, 0.1, 0.1],
            // n_classes
            10,
            // varbuilder
            vars,
        )?;

        Ok(())
    }
}
