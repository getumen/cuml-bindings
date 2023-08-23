use anyhow::anyhow;

use crate::errors::CumlError;

use super::bindings::{AgglomerativeClusteringFit, DbscanFit, KmeansFit};

pub fn agglomerative_clustering<'a>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    pairwise_conn: bool,
    metric: i32,
    n_neighbors: i32,
    init_n_clusters: i32,
) -> Result<(i32, Vec<i32>, Vec<i32>), CumlError> {
    let mut num_cluster = 0i32;
    let mut labels = vec![0; num_row];
    let mut children = vec![0; (num_row - 1) * 2];

    let result = unsafe {
        AgglomerativeClusteringFit(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            pairwise_conn,
            metric as i32,
            n_neighbors,
            init_n_clusters,
            &mut num_cluster,
            labels.as_mut_ptr() as *mut i32,
            children.as_mut_ptr() as *mut i32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to AgglomerativeClusteringFit"))?
    }

    Ok((num_cluster, labels, children))
}

pub fn dbscan<'a>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    min_pts: i32,
    eps: f64,
    metric: i32,
    max_bytes_per_batch: usize,
    verbosity: i32,
) -> Result<Vec<i32>, CumlError> {
    let mut out = vec![0; num_row];
    let result = unsafe {
        DbscanFit(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            min_pts,
            eps,
            metric,
            max_bytes_per_batch,
            verbosity,
            out.as_mut_ptr() as *mut i32,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to dbscan"))?
    }

    Ok(out)
}

pub fn kmeans<'a, 'b>(
    data: &'a [f32],
    num_row: usize,
    num_col: usize,
    sample_weight: Option<&'b [f32]>,
    k: i32,
    max_iter: i32,
    tol: f64,
    init_method: i32,
    metric: i32,
    seed: i32,
    verbosity: i32,
) -> Result<(Vec<i32>, Vec<f32>, f32, i32), CumlError> {
    let mut labels = vec![0; num_row];
    let mut centroids = vec![0f32; k as usize * num_col];
    let mut inertia = 0f32;
    let mut n_iter = 0i32;

    let num_row = num_row as i32;
    let num_col = num_col as i32;

    let result = unsafe {
        KmeansFit(
            data.as_ptr() as *const f32,
            num_row,
            num_col,
            sample_weight.map_or(std::ptr::null(), |x| x.as_ptr()),
            k,
            max_iter,
            tol,
            init_method,
            metric,
            seed,
            verbosity,
            labels.as_mut_ptr() as *mut i32,
            centroids.as_mut_ptr() as *mut f32,
            &mut inertia,
            &mut n_iter,
        )
    };

    if result != 0 {
        Err(anyhow!("fail to dbscan"))?
    }

    Ok((labels, centroids, inertia, n_iter))
}
