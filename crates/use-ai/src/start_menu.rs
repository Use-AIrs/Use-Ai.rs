use crate::error::Result;
use inquire::Select;
use lib_store::init_transformation;

pub fn start_menu() -> Result<()> {
	loop {
		let st_menu = vec!["Init Transformation", "Init Training", "Test Model", "Back"];

		let selection = Select::new("Executions:", st_menu)
			.with_help_message("Here you can load, list, import and create configurations.")
			.prompt()?;

		match selection {
			"Init Transformation" => {
				init_transform()?;
			},
			"Init Training" => {
				println!("Not implemented yet");
			},
			"Test Model" => {
				println!("Not implemented yet");
			},
			"Back" => {
				return Ok(());
			},
			_ => (),
		}
	}
}

fn init_transform() -> Result<()> {
	Ok(init_transformation()?)
}
