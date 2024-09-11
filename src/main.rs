use std::process::id;
use candle_core::{Device, Shape, Tensor, Result, DType};
use candle_nn::{Conv2d, Dropout, Linear, Module, VarBuilder, Sequential};
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
        })
    }
}

impl Module for CNNLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}


pub fn build_net<const S: usize>(
    filters: [usize; S],
    kernels: [usize; S],
    linear: [usize; S],
    dropout: [f32; S],
    vs: VarBuilder,
) -> Result<Sequential> {
    let seq = candle_nn::sequential::seq();

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

        seq.add(cnn);
    }

    Ok(seq)
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_nn::{Sequential, VarMap};
    use super::*;


    fn build_cnn_layer() -> Result<CNNLayer> {
        let device = default_device();
        let varmap = VarMap::new();
        let vars = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);


        let layer = CNNLayer::new(vars.clone(), "foo".into(), 5, 5, 5, 5, 5, 0.5)?;

        Ok(layer)
    }

    #[test]
    fn test_build_cnn_layer() -> anyhow::Result<()> {
        let layer = build_cnn_layer()?;

        println!("layer: {:?}", layer);
        Ok(())
    }

    #[test]
    fn test_cnn_forward() -> anyhow::Result<()> {
        let layer = build_cnn_layer()?;
        // let input
        Ok(())
    }


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
            // varbuilder
            vars,
        );
        // let cnn = build_cnn_layer()?;
        // net.add(cnn);
        // println!("{:?}", net.);

        Ok(())
    }
}


fn main() -> anyhow::Result<()> {
    if let device = Device::new_metal(0)? {
        println!("Device created.");
        let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
        let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

        let c = a.matmul(&b)?;


        println!("{:?}", c);
    } else {
        println!("Could not create metal device.");
    }

    Ok(())
}