mod gbdt_gen;
mod gbdt_trainer;

use crate::operator::*;
use crate::MetaData;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::Runtime;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct GbdtOperator {
	pub target: (MetaData, Handle),
	pub table: (MetaData, Handle),
	pub ctx: GbdtRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbdtRules {
	pub n_trees: u32,
	pub learning_rate: f32,
	pub max_depth: u32,
}
