pub mod calc;
pub mod stage;

pub use calc::*;
pub use stage::*;

use crate::error::{Result, StoreError};

use mongodb::bson::doc;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
	pub name: Option<String>,
	pub active: Option<bool>,
	pub version: String,
	pub data: DataSection,
	pub models: Vec<Models>,
	pub output: OutputSection,
}

impl Config {
	pub fn cfg_version() -> String {
		"0.11_pre_alpha".to_string()
	}

	pub fn get_config(path: String) -> Result<Config> {
		let file = File::open(&path)?;
		let reader = BufReader::new(file);

		let config: Config = serde_json::from_reader(reader)?;
		if config.version != Self::cfg_version() {
			Err(StoreError::InvalidConfig)
		} else {
			Ok(config)
		}
	}
}
