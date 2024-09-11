use candle_core::{Device, Shape, Tensor, Result, DType};
use candle_nn::{Conv2d, Dropout, Linear, Module, VarBuilder};
use anyhow;

const MODEL_DTYPE: DType = DType::F32;

fn default_device() -> Device {
    Device::new_metal(0).expect("Unable to create metal device.")
}


enum ModelConfig {
    CNNLayerConfig,
}

#[derive(Debug)]
pub struct LinearConfig {
    in_shape: usize,
    out_shape: usize,
}


#[derive(Debug)]
pub struct ConvConfig {
    in_shape: usize,
    out_shape: usize,
    kernel_size: usize,
}

#[derive(Debug)]
pub struct CNNLayerConfig {
    layer_name: String,
    conv_layer: ConvConfig,
    linear_config: LinearConfig,
    dropout: f32,
}


#[derive(Debug)]
pub struct CNNLayer {
    config: CNNLayerConfig,
    conv: Conv2d,
    linear: Linear,
    dropout: Dropout,
}


impl CNNLayer {
    fn new(vs: VarBuilder, config: CNNLayerConfig) -> Result<Self> {
        let conv = candle_nn::conv2d(
            config.conv_layer.in_shape,
            config.conv_layer.out_shape,
            config.conv_layer.kernel_size,
            Default::default(),
            vs.push_prefix(format!("{}-conv2D", config.layer_name)),
        )?;

        let linear = candle_nn::linear(
            config.linear_config.in_shape,
            config.linear_config.out_shape,
            vs.push_prefix(format!("{}-linear", config.layer_name)),
        )?;

        let dropout = candle_nn::Dropout::new(config.dropout);

        Ok(Self {
            config,
            conv,
            linear,
            dropout,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_nn::VarMap;
    use super::*;


    fn build_cnn_layer() -> Result<CNNLayer> {
        let device = default_device();
        let varmap = VarMap::new();
        let vars = VarBuilder::from_varmap(&varmap.clone(), MODEL_DTYPE, &device);

        let config = CNNLayerConfig {
            layer_name: "foo".to_string(),
            conv_layer: ConvConfig {
                in_shape: 5,
                out_shape: 5,
                kernel_size: 5,
            },
            linear_config: LinearConfig {
                in_shape: 5,
                out_shape: 5,
            },
            dropout: 0.5,
        };


        let layer = CNNLayer::new(vars.clone(), config)?;

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