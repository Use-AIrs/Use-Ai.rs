mod gbdt_gen;
mod gbdt_trainer;

use crate::error::Result;
use crate::model::Operation;
use crate::operator::{Context, Operator};
use crate::MetaData;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::Runtime;
use lib_proc_macros::{ctx, operator};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
#[operator]
pub struct GbdtOperator {
    pub target: (MetaData, Handle),
    pub table: (MetaData, Handle),
    pub buffer: (MetaData, Handle),
}

#[derive(Debug, Serialize, Deserialize)]
#[ctx]
pub struct GbdtRules {
    pub n_trees: u32,
    pub learning_rate: f32,
    pub max_depth: u32,
}

pub fn gbdt<R: Runtime, Op: Operation<R>>(_op: GbdtOperator, _ctx: GbdtRules) {
    todo!()
}

#[derive(Debug)]
pub struct XGBoostModel {
    pub trees: Vec<DecisionTree>,
    pub rate: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub subsamples: f64,
    pub max_depth: usize,
    pub min_child_weight: f64,
}

#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: TreeNode,
}


#[derive(Debug, Clone)]
pub struct TreeNode {
    split_feature: Option<usize>,
    split_threshold: Option<f64>,
    left_child: Option<Box<TreeNode>>,
    right_child: Option<Box<TreeNode>>,
    leaf_value: Option<f64>,
}

impl TreeNode {

    fn new_leaf(value: f64) -> Self {
        TreeNode {
            split_feature: None,
            split_threshold: None,
            left_child: None,
            right_child: None,
            leaf_value: Some(value),
        }
    }
}

#[derive(Debug, Clone)]
struct DataSet {
    features: Vec<f64>,
    labels: Vec<f64>,
}

impl XGBoostModel {
    fn new(
        rate: f64,
        gamma: f64,
        lambda: f64,
        subsamples: f64,
        max_depth: usize,
        min_child_weight: f64,
    ) -> Self {
        XGBoostModel {
            trees: Vec::new(),
            rate,
            gamma,
            lambda,
            subsamples,
            max_depth,
            min_child_weight,
        }
    }
}