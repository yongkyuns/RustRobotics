use burn::{
    module::Module,
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor, TensorData},
};
use rust_robotics_core::{LinearSnapshot, PolicySnapshot, ValueSnapshot};

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub input: Linear<B>,
    pub hidden: Linear<B>,
    pub output: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(device: &B::Device, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden_dim).init(device),
            hidden: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            output: LinearConfig::new(hidden_dim, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        self.output.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct PolicyNetwork<B: Backend> {
    pub mlp: Mlp<B>,
    #[module]
    pub action_limit: f32,
}

impl<B: Backend> PolicyNetwork<B> {
    pub fn new(device: &B::Device, input_dim: usize, hidden_dim: usize, action_limit: f32) -> Self {
        Self {
            mlp: Mlp::new(device, input_dim, hidden_dim, 1),
            action_limit,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::tanh(self.mlp.forward(input)).mul_scalar(self.action_limit)
    }

    pub fn snapshot(&self, action_std: f32) -> PolicySnapshot {
        PolicySnapshot {
            input: linear_snapshot_from_linear(&self.mlp.input),
            hidden: linear_snapshot_from_linear(&self.mlp.hidden),
            output: linear_snapshot_from_linear(&self.mlp.output),
            action_limit: self.action_limit,
            action_std,
        }
    }
}

#[derive(Module, Debug)]
pub struct ValueNetwork<B: Backend> {
    pub mlp: Mlp<B>,
}

impl<B: Backend> ValueNetwork<B> {
    pub fn new(device: &B::Device, input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            mlp: Mlp::new(device, input_dim, hidden_dim, 1),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mlp.forward(input)
    }

    pub fn snapshot(&self) -> ValueSnapshot {
        ValueSnapshot {
            input: linear_snapshot_from_linear(&self.mlp.input),
            hidden: linear_snapshot_from_linear(&self.mlp.hidden),
            output: linear_snapshot_from_linear(&self.mlp.output),
        }
    }
}

fn linear_snapshot_from_linear<B: Backend>(linear: &Linear<B>) -> LinearSnapshot {
    let weight = linear.weight.val().to_data().to_vec::<f32>().unwrap();
    let bias = linear
        .bias
        .as_ref()
        .map(|bias| bias.val().to_data().to_vec::<f32>().unwrap())
        .unwrap_or_default();
    let [in_dim, out_dim] = linear.weight.shape().dims();

    LinearSnapshot {
        in_dim,
        out_dim,
        weight,
        bias,
    }
}

fn linear_from_snapshot<B: Backend>(snapshot: &LinearSnapshot, device: &B::Device) -> Linear<B> {
    let weight = Param::from_data(
        TensorData::new(snapshot.weight.clone(), [snapshot.in_dim, snapshot.out_dim]),
        device,
    );
    let bias = if snapshot.bias.is_empty() {
        None
    } else {
        Some(Param::from_data(
            TensorData::new(snapshot.bias.clone(), [snapshot.out_dim]),
            device,
        ))
    };
    Linear { weight, bias }
}

pub fn policy_network_from_snapshot<B: Backend>(
    snapshot: &PolicySnapshot,
    device: &B::Device,
) -> PolicyNetwork<B> {
    PolicyNetwork {
        mlp: Mlp {
            input: linear_from_snapshot(&snapshot.input, device),
            hidden: linear_from_snapshot(&snapshot.hidden, device),
            output: linear_from_snapshot(&snapshot.output, device),
        },
        action_limit: snapshot.action_limit,
    }
}

pub fn value_network_from_snapshot<B: Backend>(
    snapshot: &ValueSnapshot,
    device: &B::Device,
) -> ValueNetwork<B> {
    ValueNetwork {
        mlp: Mlp {
            input: linear_from_snapshot(&snapshot.input, device),
            hidden: linear_from_snapshot(&snapshot.hidden, device),
            output: linear_from_snapshot(&snapshot.output, device),
        },
    }
}

pub fn obs_tensor<B: Backend>(device: &B::Device, observations: &[[f32; 4]]) -> Tensor<B, 2> {
    let flat = observations
        .iter()
        .flat_map(|observation| observation.iter().copied())
        .collect::<Vec<_>>();
    Tensor::<B, 2>::from_data(TensorData::new(flat, [observations.len(), 4]), device)
}

pub fn scalar_tensor<B: Backend>(device: &B::Device, values: &[f32]) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(values.to_vec(), [values.len(), 1]), device)
}
