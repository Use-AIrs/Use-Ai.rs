use crate::error::{Result, StagingError};
use lib_calculator::config::Models;
use lib_stage::config::{DataSection, OutputSection};
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
    pub fn get_config(path: String) -> Result<Config> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let config: Config = serde_json::from_reader(reader)?;
        if config.version != "0.1_pre_alpha" {
            Err(StagingError::InvalidConfig)
        } else {
            Ok(config)
        }
    }
}
