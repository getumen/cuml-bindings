use crate::{errors::CumlError, metric::Metric, sys::clustering};

#[derive(Debug, Clone)]
pub struct AgglomerativeClusteringResult {
    num_cluster: i32,
    labels: Vec<i32>,
    children: Vec<i32>,
}

impl AgglomerativeClusteringResult {
    pub fn num_cluster(&self) -> i32 {
        self.num_cluster
    }

    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    pub fn children(&self) -> &[i32] {
        &self.children
    }
}

pub fn agglomerative_clustering<'a>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    pairwise_conn: bool,
    metric: Metric,
    n_neighbors: i32,
    init_n_clusters: i32,
) -> Result<AgglomerativeClusteringResult, CumlError> {
    let result = clustering::agglomerative_clustering(
        data,
        num_row,
        num_col,
        pairwise_conn,
        metric as i32,
        n_neighbors,
        init_n_clusters,
    )?;

    Ok(AgglomerativeClusteringResult {
        num_cluster: result.0,
        labels: result.1,
        children: result.2,
    })
}

pub fn dbscan<'a>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    min_pts: i32,
    eps: f64,
    metric: Metric,
    max_bytes_per_batch: usize,
    verbosity: i32,
) -> Result<Vec<i32>, CumlError> {
    clustering::dbscan(
        data,
        num_row,
        num_col,
        min_pts,
        eps,
        metric as i32,
        max_bytes_per_batch,
        verbosity,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum KmeansInitMethod {
    KMeansPlusPlus = 0,
    Random = 1,
    Array = 2,
}

#[derive(Debug, Clone)]
pub struct KmeansResult {
    labels: Vec<i32>,
    centroids: Vec<f32>,
    inertia: f32,
    n_iter: i32,
}

impl KmeansResult {
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    pub fn inertia(&self) -> f32 {
        self.inertia
    }

    pub fn n_iter(&self) -> i32 {
        self.n_iter
    }
}

pub fn kmeans<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    sample_weight: Option<&'b [f32]>,
    k: i32,
    max_iter: i32,
    tol: f64,
    init_method: KmeansInitMethod,
    metric: Metric,
    seed: i32,
    verbosity: i32,
) -> Result<KmeansResult, CumlError> {
    let result = clustering::kmeans(
        data,
        num_row,
        num_col,
        sample_weight,
        k,
        max_iter,
        tol,
        init_method as i32,
        metric as i32,
        seed,
        verbosity,
    )?;

    Ok(KmeansResult {
        labels: result.0,
        centroids: result.1,
        inertia: result.2,
        n_iter: result.3,
    })
}
