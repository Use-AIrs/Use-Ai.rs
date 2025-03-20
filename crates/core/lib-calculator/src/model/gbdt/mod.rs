mod gbdt_gen;
mod gbdt_trainer;

use crate::MetaData;

use cubecl::server::Handle;
use lib_store::cfg::GbdtRules;

#[derive(Debug)]
pub struct GbdtOperator {
	pub target: (MetaData, Handle),
	pub table: (MetaData, Handle),
	pub ctx: GbdtRules,
}
