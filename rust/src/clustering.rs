use crate::{
    errors::CumlError,
    log_level::LogLevel,
    metric::Metric,
    sys::{clustering, device_resource::DeviceResource},
};

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

pub struct AgglomerativeClustering {
    device_resource: DeviceResource,
    pairwise_conn: bool,
    metric: Metric,
    n_neighbors: i32,
    init_n_clusters: i32,
}

impl AgglomerativeClustering {
    pub fn new(
        pairwise_conn: bool,
        metric: Metric,
        n_neighbors: i32,
        init_n_clusters: i32,
    ) -> Self {
        let device_resource = DeviceResource::new();

        Self {
            device_resource,
            pairwise_conn,
            metric,
            n_neighbors,
            init_n_clusters,
        }
    }

    pub fn fit(
        &self,
        data: &[f32],
        num_row: usize,
        num_col: usize,
    ) -> Result<AgglomerativeClusteringResult, CumlError> {
        let mut labels = vec![0; num_row];
        let mut children = vec![0; 2 * (num_row - 1)];
        let num_cluster = clustering::agglomerative_clustering(
            &self.device_resource,
            &data,
            num_row,
            num_col,
            self.pairwise_conn,
            self.metric as i32,
            self.n_neighbors,
            self.init_n_clusters,
            &mut labels,
            &mut children,
        )?;

        Ok(AgglomerativeClusteringResult {
            num_cluster,
            labels,
            children,
        })
    }
}

pub struct DBScan {
    device_resource: DeviceResource,
    min_pts: i32,
    eps: f64,
    metric: Metric,
    max_bytes_per_batch: usize,
    verbosity: LogLevel,
}

impl DBScan {
    pub fn new(
        min_pts: i32,
        eps: f64,
        metric: Metric,
        max_bytes_per_batch: usize,
        verbosity: LogLevel,
    ) -> Self {
        let device_resource = DeviceResource::new();
        Self {
            device_resource,
            min_pts,
            eps,
            metric,
            max_bytes_per_batch,
            verbosity,
        }
    }

    pub fn fit(&self, data: &[f32], num_row: usize, num_col: usize) -> Result<Vec<i32>, CumlError> {
        let mut labels = vec![0; num_row];

        clustering::dbscan(
            &self.device_resource,
            &data,
            num_row,
            num_col,
            self.min_pts,
            self.eps,
            self.metric as i32,
            self.max_bytes_per_batch,
            self.verbosity as i32,
            &mut labels,
        )?;

        Ok(labels)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmeansInitMethod {
    KMeansPlusPlus = 0,
    Random = 1,
    // TODO: implement Array
    // Array = 2,
}

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

pub struct Kmeans {
    device_resource: DeviceResource,
    k: i32,
    max_iter: i32,
    tol: f64,
    init_method: KmeansInitMethod,
    metric: Metric,
    seed: i32,
    verbosity: LogLevel,
}

impl Kmeans {
    pub fn new(
        k: i32,
        max_iter: i32,
        tol: f64,
        init_method: KmeansInitMethod,
        metric: Metric,
        seed: i32,
        verbosity: LogLevel,
    ) -> Self {
        let device_resource = DeviceResource::new();
        Self {
            device_resource,
            k,
            max_iter,
            tol,
            init_method,
            metric,
            seed,
            verbosity,
        }
    }

    pub fn fit(
        &self,
        data: &[f32],
        num_row: usize,
        num_col: usize,
        sample_weight: Option<&[f32]>,
    ) -> Result<KmeansResult, CumlError> {
        let mut labels = vec![0; num_row];
        let mut centroids = vec![0f32; self.k as usize * num_col];

        let (inertia, n_iter) = clustering::kmeans(
            &self.device_resource,
            &data,
            num_row,
            num_col,
            self.k,
            self.max_iter,
            self.tol,
            self.init_method as i32,
            self.metric as i32,
            self.seed,
            self.verbosity as i32,
            &mut labels,
            &mut centroids,
        )?;

        Ok(KmeansResult {
            labels,
            centroids,
            inertia,
            n_iter,
        })
    }
}
