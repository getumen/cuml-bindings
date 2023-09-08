use std::path::PathBuf;

fn main() {
    let bindings = bindgen::Builder::default()
        .raw_line("#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused)]")
        .header("../include/cuml4c/agglomerative_clustering.h")
        .header("../include/cuml4c/dbscan.h")
        .header("../include/cuml4c/fil.h")
        .header("../include/cuml4c/kmeans.h")
        .header("../include/cuml4c/memory_resource.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("unable to generate bindings");

    let out_path = PathBuf::from("src");
    bindings
        .write_to_file(out_path.join("sys/bindings.rs"))
        .expect("Couldn't write bindings!");
}
