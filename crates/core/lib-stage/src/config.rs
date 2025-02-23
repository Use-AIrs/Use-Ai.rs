use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSection {
	pub source: DataSource,
	pub scheme: Option<DataScheme>,
	pub transformer: Option<Vec<TransformationStep>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
	#[serde(rename = "type")]
	pub source_type: String,
	pub path: Option<String>,
	pub delimiter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataScheme {
	pub columns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationStep {
	#[serde(rename = "t_id")]
	pub id: usize,
	pub operation: String,
	pub params: Option<Parameters>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameters {
	pub columns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSection {
	pub final_output: Vec<String>,
}
