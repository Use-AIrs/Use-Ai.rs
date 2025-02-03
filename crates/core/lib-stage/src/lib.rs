use crate::config::DataSection;
use crate::data::{RawTable, Table};
use crate::error::{Result, TransformerError};

pub mod config;
pub mod data;
#[allow(unused_imports, dead_code, unused_variables)]
pub mod error;
pub mod output_guard;

pub fn transformer(instruction: DataSection) -> Result<()> {
    let data = get_data(&instruction)?;
    let result = transform(data, &instruction)?;
    println!("{:?}", result);
    Ok(())
}

fn get_data(data_source: &DataSection) -> Result<RawTable<String>> {
    let config = data_source;
    match data_source.source.source_type.as_str() {
        "csv" => RawTable::<String>::from_csv(config),
        _ => Err(TransformerError::DataTypeNotSupported),
    }
}
pub fn transform(mut table: RawTable<String>, instructions: &DataSection) -> Result<Table> {
    let transformations = match &instructions.transformer {
        Some(transformations) => transformations,
        None => return Err(TransformerError::NoTransformationsDefined),
    };

    for transformation in transformations {
        match transformation.operation.as_str() {
            "categories" => {
                let params = transformation
                    .params
                    .clone()
                    .ok_or(TransformerError::NoTransformationParams)?;

                let columns = params
                    .columns
                    .ok_or(TransformerError::TransformationParamsWrong)?;

                println!("Applying 'categories' to columns: {:?}", columns);
                table = table.categories(columns)?;
            }

            "date_conv" => {
                todo!()
            }

            other => {
                println!("Unknown operation: {}", other);
            }
        }
    }

    table.convert_to_f32_table()
}
