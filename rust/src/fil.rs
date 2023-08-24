use std::path::Path;

use crate::{
    errors::CumlError,
    sys::{
        bindings::FILModelHandle,
        fil::{fil_free_model, fil_get_num_class, fil_load_model, fil_predict},
    },
};
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum ModelType {
    // XGBoost xgboost model (binary model file)
    XGBoost = 0,
    // XGBoostJSON xgboost model (json model file)
    XGBoostJSON = 1,
    // LightGBM lighgbm model (binary model file)
    LightGBM = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum Algo {
    // AlgoAuto choose the algorithm automatically; currently chooses NAIVE for sparse forests
    //  and BatchTreeReorg for dense ones
    AlgoAuto = 0,
    // Naive naive algorithm: 1 thread block predicts 1 row; the row is cached in
    //  shared memory, and the trees are distributed cyclically between threads
    Naive = 1,
    // TreeReorg tree reorg algorithm: same as naive, but the tree nodes are rearranged
    //  into a more coalescing-friendly layout: for every node position,
    //  nodes of all trees at that position are stored next to each other
    TreeReorg = 2,
    // BatchTreeReorg batch tree reorg algorithm: same as tree reorg, but predictions multiple rows (up to 4)
    //  in a single thread block
    BatchTreeReorg = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum StorageType {
    // Auto decide automatically; currently always builds dense forests
    Auto = 0,
    // Dense import the forest as dense
    Dense = 1,
    // Sparse import the forest as sparse (currently always with 16-byte nodes)
    Sparse = 2,
    // Sparse8 (experimental) import the forest as sparse with 8-byte nodes; can fail if
    //  8-byte nodes are not enough to store the forest, e.g. there are too many
    //  nodes in a tree or too many features; note that the number of bits used to
    //  store the child or feature index can change in the future; this can affect
    //  whether a particular forest can be imported as SPARSE8 */
    Sparse8 = 3,
}

#[derive(Debug, Clone)]
pub struct Model {
    model: FILModelHandle,
    num_class: usize,
}

impl Model {
    pub fn new<P: AsRef<Path>>(
        model_type: ModelType,
        model_path: P,
        algo: Algo,
        classification: bool,
        threshold: f32,
        storage_type: StorageType,
        block_per_sm: i32,
        thread_per_tree: i32,
        n_items: i32,
    ) -> Result<Self, CumlError> {
        let model = fil_load_model(
            model_type,
            model_path,
            algo,
            classification,
            threshold,
            storage_type,
            block_per_sm,
            thread_per_tree,
            n_items,
        )?;
        let num_class = fil_get_num_class(model)?;
        Ok(Self { model, num_class })
    }

    pub fn predict<'a>(
        &self,
        data: &'a [f32],
        num_row: usize,
        output_class_probabilities: bool,
    ) -> Result<Vec<f32>, CumlError> {
        let result = fil_predict(
            self.model,
            data,
            num_row,
            output_class_probabilities,
            self.num_class,
        )?;

        Ok(result)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        fil_free_model(self.model).expect("fail to free model");
    }
}
