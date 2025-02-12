use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Field, Fields, ItemStruct};

fn get_fields(input: &ItemStruct) -> Result<Vec<&Field>, syn::Error> {
    match &input.fields {
        Fields::Named(fields_named) => Ok(fields_named.named.iter().collect()),
        _ => Err(syn::Error::new_spanned(
            input,
            "Expected a struct with named fields",
        )),
    }
}

#[proc_macro_attribute]
pub fn operator(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let struct_ident = input.ident.clone();

    let fields = match get_fields(&input) {
        Ok(f) => f,
        Err(err) => return err.to_compile_error().into(),
    };

    let mut tensor_bindings = Vec::new();
    let mut tensor_idents = Vec::new();

    for (count, field) in fields.iter().enumerate() {
        let field_ident = field.ident.as_ref().unwrap();

        let meta_ident = format_ident!("t{}_meta", count);
        let handle_ident = format_ident!("t{}_handle", count);
        let tensor_ident = format_ident!("t{}", count);
        tensor_idents.push(tensor_ident.clone());

        tensor_bindings.push(quote! {
            let #meta_ident = &self.#field_ident.0;
            let #handle_ident = &self.#field_ident.1;
            let #tensor_ident = unsafe {
                TensorHandleRef::<'a, R>::from_raw_parts(
                    &#handle_ident,
                    &*#meta_ident.stride,
                    &*#meta_ident.shape,
                    std::mem::size_of::<f32>()
                )
            };
        });
    }

    let ret_type = if tensor_idents.len() == 1 {
        quote! { TensorHandleRef<'a, R> }
    } else {
        let types = tensor_idents
            .iter()
            .map(|_| quote! { TensorHandleRef<'a, R> });
        quote! { (#(#types),*) }
    };

    let ret_expr = if tensor_idents.len() == 1 {
        let t0 = &tensor_idents[0];
        quote! { #t0 }
    } else {
        quote! { (#(#tensor_idents),*) }
    };

    let expanded = quote! {
        #input

        impl<R: Runtime> Operator<R> for #struct_ident {
            type Tuple<'a> = #ret_type;
            fn tensor_refs<'a>(&'a self) -> #ret_type {
                #(#tensor_bindings)*
                #ret_expr
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn ctx(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let struct_ident = input.ident.clone();

    let fields = match get_fields(&input) {
        Ok(f) => f,
        Err(err) => return err.to_compile_error().into(),
    };

    let mut ctx_bindings = Vec::new();
    let mut ctx_idents = Vec::new();
    for (count, field) in fields.iter().enumerate() {
        let field_ident = field.ident.as_ref().unwrap();
        let ctx_ident = format_ident!("c{}", count);
        ctx_idents.push(ctx_ident.clone());
        ctx_bindings.push(quote! {
            let #ctx_ident = self.#field_ident;
        });
    }

    let ret_type = if fields.len() == 1 {
        let ty = &fields[0].ty;
        quote! { #ty }
    } else {
        let types = fields.iter().map(|f| {
            let ty = &f.ty;
            quote! { #ty }
        });
        quote! { (#(#types),*) }
    };

    let ret_expr = if ctx_idents.len() == 1 {
        let c0 = &ctx_idents[0];
        quote! { #c0 }
    } else {
        quote! { (#(#ctx_idents),*) }
    };

    let expanded = quote! {
        #input

        impl Context for #struct_ident {
            type Tuple = #ret_type;
            fn ctx_ref(&self) -> #ret_type {
                #(#ctx_bindings)*
                #ret_expr
            }
        }
    };

    TokenStream::from(expanded)
}
