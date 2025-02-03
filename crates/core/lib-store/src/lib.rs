use crate::error::Result;
use crate::mangodb::get_active_config;
use lib_stage::transformer;

pub mod ai_config;
pub mod error;
pub mod mangodb;

pub fn init_transformation() -> Result<()> {
    let config = get_active_config()?;
    transformer(config.data);
    Ok(())
}
