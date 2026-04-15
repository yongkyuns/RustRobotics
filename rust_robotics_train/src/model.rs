use burn::{
    module::Module,
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSnapshot {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LinearSnapshot {
    fn from_linear<B: Backend>(linear: &Linear<B>) -> Self {
        let weight = linear.weight.val().to_data().to_vec::<f32>().unwrap();
        let bias = linear
            .bias
            .as_ref()
            .map(|bias| bias.val().to_data().to_vec::<f32>().unwrap())
            .unwrap_or_default();
        let [in_dim, out_dim] = linear.weight.shape().dims();

        Self {
            in_dim,
            out_dim,
            weight,
            bias,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_dim];
        for out_idx in 0..self.out_dim {
            let mut sum = self.bias.get(out_idx).copied().unwrap_or(0.0);
            for in_idx in 0..self.in_dim {
                sum += input[in_idx] * self.weight[in_idx * self.out_dim + out_idx];
            }
            output[out_idx] = sum;
        }
        output
    }

    pub fn to_linear<B: Backend>(&self, device: &B::Device) -> Linear<B> {
        let weight = Param::from_data(
            TensorData::new(self.weight.clone(), [self.in_dim, self.out_dim]),
            device,
        );
        let bias = if self.bias.is_empty() {
            None
        } else {
            Some(Param::from_data(
                TensorData::new(self.bias.clone(), [self.out_dim]),
                device,
            ))
        };
        Linear { weight, bias }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicySnapshot {
    pub input: LinearSnapshot,
    pub hidden: LinearSnapshot,
    pub output: LinearSnapshot,
    pub action_limit: f32,
    pub action_std: f32,
}

impl PolicySnapshot {
    pub fn from_policy_network<B: Backend>(policy: &PolicyNetwork<B>, action_std: f32) -> Self {
        Self {
            input: LinearSnapshot::from_linear(&policy.mlp.input),
            hidden: LinearSnapshot::from_linear(&policy.mlp.hidden),
            output: LinearSnapshot::from_linear(&policy.mlp.output),
            action_limit: policy.action_limit,
            action_std,
        }
    }

    pub fn action_std(&self) -> f32 {
        self.action_std
    }

    pub fn to_policy_network<B: Backend>(&self, device: &B::Device) -> PolicyNetwork<B> {
        PolicyNetwork {
            mlp: Mlp {
                input: self.input.to_linear(device),
                hidden: self.hidden.to_linear(device),
                output: self.output.to_linear(device),
            },
            action_limit: self.action_limit,
        }
    }

    pub fn act(&self, observation: [f32; 4]) -> f32 {
        let hidden0 = relu_vec(self.input.forward(&observation));
        let hidden1 = relu_vec(self.hidden.forward(&hidden0));
        let out = self.output.forward(&hidden1);
        out[0].tanh() * self.action_limit
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueSnapshot {
    pub input: LinearSnapshot,
    pub hidden: LinearSnapshot,
    pub output: LinearSnapshot,
}

impl ValueSnapshot {
    pub fn from_value_network<B: Backend>(value: &ValueNetwork<B>) -> Self {
        Self {
            input: LinearSnapshot::from_linear(&value.mlp.input),
            hidden: LinearSnapshot::from_linear(&value.mlp.hidden),
            output: LinearSnapshot::from_linear(&value.mlp.output),
        }
    }

    pub fn to_value_network<B: Backend>(&self, device: &B::Device) -> ValueNetwork<B> {
        ValueNetwork {
            mlp: Mlp {
                input: self.input.to_linear(device),
                hidden: self.hidden.to_linear(device),
                output: self.output.to_linear(device),
            },
        }
    }
}

fn relu_vec(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(|value| value.max(0.0)).collect()
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
