use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
pub enum Models {
    NeuralNetwork {
        id: String,
        input_columns: Option<Vec<String>>,
        input_from: Option<String>,
        hyperparams: HPNeuralNetwork,
        output: ModelOutput,
    },
    GradientBoostedDecisionTree {
        id: String,
        input_columns: Option<Vec<String>>,
        input_from: Option<String>,
        target_columns: Vec<String>,
        hyperparams: HPGradientBoostedDecisionTree,
        output: ModelOutput,
    },
    QLearning {
        id: String,
        input_columns: Option<Vec<String>>,
        input_from: Option<String>,
        hyperparams: HPQLearning,
        output: ModelOutput,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPNeuralNetwork {
    pub layers: Vec<u64>,
    pub activation: String,
    pub optimizer: Option<String>,
    pub epochs: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPGradientBoostedDecisionTree {
    pub n_trees: u64,
    pub learning_rate: f32,
    pub max_depth: u64,
    pub subsample: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPQLearning {
    pub environment: String,
    pub learning_rate: f32,
    pub discount: f32,
    pub episodes: u64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelOutput {
    pub output_mode: String,
}
