use burn::tensor::backend::Backend;

#[cfg(not(target_arch = "wasm32"))]
pub type TrainBackend = burn::backend::NdArray;
#[cfg(not(target_arch = "wasm32"))]
pub type TrainDevice = burn::backend::ndarray::NdArrayDevice;

#[cfg(target_arch = "wasm32")]
pub type TrainBackend = burn::backend::NdArray;
#[cfg(target_arch = "wasm32")]
pub type TrainDevice = burn::backend::ndarray::NdArrayDevice;

pub type AutodiffBackend = burn::backend::Autodiff<TrainBackend>;
pub type AutodiffDevice = <AutodiffBackend as Backend>::Device;

pub fn default_train_device() -> TrainDevice {
    Default::default()
}
