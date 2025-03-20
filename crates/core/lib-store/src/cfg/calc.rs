use serde::{Deserialize, Serialize};

/// Models is an enum which represents every current model available  in Use-AI.rs and Context needed
/// for the Operations.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
pub enum Models {
	NeuralNetwork {
		id: usize,
		input_columns: Option<Vec<String>>,
		input_from: Option<String>,
		hyperparams: HPNeuralNetwork,
		mode: Mode,
	},
	GradientBoostedDecisionTree {
		id: usize,
		input_columns: Option<Vec<String>>,
		input_from: Option<String>,
		target_columns: Vec<String>,
		hyperparams: GbdtRules,
		mode: Mode,
	},
	QLearning {
		id: usize,
		input_columns: Option<Vec<String>>,
		input_from: Option<String>,
		hyperparams: HPQLearning,
		mode: Mode,
	},
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Mode {
	Train,
	Generate,
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
	pub sub_sample: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPQLearning {
	pub environment: String,
	pub learning_rate: f32,
	pub discount: f32,
	pub episodes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbdtRules {
	pub n_trees: u32,
	pub learning_rate: f32,
	pub max_depth: u32,
}
