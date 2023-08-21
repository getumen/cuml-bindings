file(REMOVE_RECURSE
  "libcuml4c.pdb"
  "libcuml4c.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cuml4c.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
