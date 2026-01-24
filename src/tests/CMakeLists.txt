cmake_minimum_required(VERSION 3.15)
project(CVLM_Catch2_Test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get R paths
execute_process(COMMAND Rscript -e "cat(R.home('lib'))" OUTPUT_VARIABLE R_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND Rscript -e "cat(R.home('include'))" OUTPUT_VARIABLE R_INC_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND Rscript -e "cat(system.file('include', package='Rcpp'))" OUTPUT_VARIABLE RCPP_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND Rscript -e "cat(system.file('include', package='RcppEigen'))" OUTPUT_VARIABLE RCPP_EIGEN_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND Rscript -e "cat(system.file('include', package='RcppParallel'))" OUTPUT_VARIABLE RCPP_PARALLEL_INC OUTPUT_STRIP_TRAILING_WHITESPACE)

# Link against catch2
find_package(Catch2 3 REQUIRED)
add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# Link against R
target_link_directories(tests PRIVATE ${R_LIB_DIR})
target_link_libraries(tests PRIVATE R)

# Include R and Rcpp headers as system includes
target_include_directories(tests SYSTEM PRIVATE ${R_INC_DIR})
target_include_directories(tests SYSTEM PRIVATE ${RCPP_INC})
target_include_directories(tests SYSTEM PRIVATE ${RCPP_EIGEN_INC})
target_include_directories(tests SYSTEM PRIVATE ${RCPP_PARALLEL_INC})
