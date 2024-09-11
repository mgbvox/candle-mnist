use candle_core::{Device, Shape, Tensor, Result};
use candle_core::op::Op;

fn default_device() -> Device {
    Device::new_metal(0).expect("Unable to create metal device.")
}


struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new<S: Into<Shape>>(shape: S,
            device: Option<Device>,
           weights_initializer: Option<dyn Fn()->Result<Tensor>>,
           bias_initializer: Option<dyn Fn()->Result<Tensor>>
    ) -> Self {

        

        let weights = match weights_initializer {
            None => Tensor::randn(0f32, 1.0, shape, &device.unwrap_or(default_device())),
            Some(callback) => callback()
        };

        let bias = match bias_initializer {
            None => Tensor::randn(0f32, 1.0, shape, &device.unwrap_or(default_device())),
            Some(init) => init()
        };

        Linear {
            weights:
        }

    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if let device = Device::new_metal(0)? {
        println!("Device created.");
        let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
        let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

        let c = a.matmul(&b)?;


        println!("{:?}", c);
    } else {
        println!("Device not make uhoh.");
    }


    Ok(())
}
