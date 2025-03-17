extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
	parenthesized,
	parse::{Parse, ParseStream, Result as SynResult},
	parse_macro_input,
	punctuated::Punctuated,
	Expr, Ident, Path, Token, Type,
};

struct ActionStep {
	inputs: Option<Expr>,
	op: Path,
	output: Option<Expr>,
}

fn to_var(idx: usize) -> (Ident, Ident) {
	let a = format!("a{}", idx);
	let b = format!("b{}", idx);
	(
		Ident::new(&a, Span::call_site()),
		Ident::new(&b, Span::call_site()),
	)
}

impl Parse for ActionStep {
	fn parse(input: ParseStream) -> SynResult<Self> {
		let content;
		parenthesized!(content in input);

		if !content.peek(syn::token::Paren) && !content.peek2(Token![,]) {
			let fork = content.fork();
			if let Ok(op) = fork.parse::<Path>() {
				if fork.is_empty() {
					content.parse::<Path>()?; // Consume the path
					return Ok(ActionStep {
						inputs: None,
						op,
						output: None,
					});
				}
			}
		}

		let inputs_expr: Expr = content.parse()?;
		content.parse::<Token![,]>()?;
		let op: Path = content.parse()?;
		let output = if content.peek(Token![,]) {
			content.parse::<Token![,]>()?;
			Some(content.parse::<Expr>()?)
		} else {
			None
		};

		Ok(ActionStep {
			inputs: Some(inputs_expr),
			op,
			output,
		})
	}
}

struct ActionSpaceInput {
	runtime: Type,
	steps: Punctuated<ActionStep, Token![,]>,
}

impl Parse for ActionSpaceInput {
	fn parse(input: ParseStream) -> SynResult<Self> {
		let runtime = input.parse::<Type>()?;
		input.parse::<Token![,]>()?;
		let steps = Punctuated::parse_terminated(input)?;

		Ok(ActionSpaceInput { runtime, steps })
	}
}

#[proc_macro]
pub fn action_space(input: TokenStream) -> TokenStream {
	let action_input = parse_macro_input!(input as ActionSpaceInput);
	let runtime = action_input.runtime;
	let steps = action_input.steps.into_iter().collect::<Vec<_>>();

	if steps.is_empty() {
		return syn::Error::new(
			Span::call_site(),
			"No steps provided for action_space!",
		)
		.to_compile_error()
		.into();
	}

	let mut code = TokenStream2::new();
	let mut current_output = None;

	code.extend(quote! {
		let client = #runtime::client(&Default::default());
	});

	for (i, step) in steps.iter().enumerate() {
		let op = &step.op;
		let (a, b) = to_var(i);

		if i == 0 && step.inputs.is_none() {
			return syn::Error::new(
				Span::call_site(),
				"Please specify input tensor",
			)
			.to_compile_error()
			.into();
		}

		if let Some(inputs) = &step.inputs {
			if op
				.segments
				.last()
				.unwrap()
				.ident
				.to_string()
				.starts_with("Exec")
			{
				code.extend(quote! {
					let (#a, #b) = #op::<#runtime>::exec(#inputs, &client)?;
				});
				current_output = Some(quote! { (#a, #b) });
			} else if op
				.segments
				.last()
				.unwrap()
				.ident
				.to_string()
				.starts_with("Prep")
			{
				code.extend(quote! {
					let #a = #op::<#runtime>::push(#inputs, &client)?;
				});
				current_output = Some(quote! { #a });
			} else {
				code.extend(quote! {
					let #a = #op::<#runtime>::push(#inputs, &client)?;
				});
				current_output = Some(quote! { #a });
			}
		} else {
			if let Some(prev_output) = current_output {
				if op
					.segments
					.last()
					.unwrap()
					.ident
					.to_string()
					.starts_with("Exec")
				{
					code.extend(quote! {
						let (#a, #b) = #op::<#runtime>::exec(#prev_output, &client)?;
					});
					current_output = Some(quote! { (&#a, &#b) });
				} else if op
					.segments
					.last()
					.unwrap()
					.ident
					.to_string()
					.starts_with("Prep")
				{
					code.extend(quote! {
						let #a = #op::push(#prev_output, &client)?;
					});
					current_output = Some(quote! { #a });
				} else {
					code.extend(quote! {
						let #a = #op::<#runtime>::push(#prev_output, &client)?;
					});
					current_output = Some(quote! { #a });
				}
			} else {
				return syn::Error::new(
					Span::call_site(),
					"Cannot use default input on first step",
				)
				.to_compile_error()
				.into();
			}
		}

		if let Some(output_var) = &step.output {
			code.extend(quote! {
				let #output_var = #b;
			});
		}
	}

	let last_idx = steps.len() - 1;
	let (last_a, last_b) = to_var(last_idx);
	code.extend(quote! {
		(#last_a, #last_b)
	});

	let output = quote! {
		{
			#code
		}
	};

	TokenStream::from(output)
}
