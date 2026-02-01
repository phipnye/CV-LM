# cvLM 2.0.0

### Major Changes & Breaking Updates

-   **New Default Behavior**: Data centering (`center = TRUE`) is now the default. This ensures that the intercept is not penalized in ridge regression, aligning the package with standard statistical methodologies.
-   **API Cleanup**: Removed the `verbose` argument from `grid.search`. The new C++ backend evaluates the lambda grid analytically, rendering progress bars unnecessary.
-   **Refined Object Inheritance**: For `lm` and `glm` methods, `subset` and `na.action` are now handled by the model object prior to cross-validation, ensuring consistency with the original model fit.

### Performance & Engine Overhaul

The engine has been transitioned from RcppEigen to RcppArmadillo, allowing the package to leverage high-performance LAPACK and BLAS libraries for large-scale matrix operations.

-   **SVD-Powered Grid Search**: `grid.search` has been entirely rewritten in C++. It now utilizes a single **Singular Value Decomposition (SVD)** to evaluate the entire $\lambda$ grid analytically.
    -   **Efficiency**: Reduces computational complexity from $O(np^2)$ per grid point to $O(\min(n, p))$ after the initial decomposition.
-   **Parallel Computation**: Refined and further integrated `RcppParallel` to distribute workloads.
    -   For K-fold CV, threads are distributed across folds.
    -   For GCV/LOOCV grid searches, threads are distributed across the $\lambda$ grid.
-   **Optimized LOOCV/GCV**: Implemented closed-form solutions for Leave-One-Out and Generalized Cross-Validation using the hat-matrix diagonal, avoiding $n$ model refits.

### Numerical Robustness

-   **OLS Evolution**: Transitioned from Column-Pivoted QR decomposition to **Complete Orthogonal Decomposition (COD)**. This enables the computation of the unique minimum $L_2$ norm solution for column rank-deficient or underdetermined ($p > n$) systems.
-   **Ridge Evolution**: Transitioned from Cholesky-based methods to **Singular Value Decomposition (SVD)**. This avoids the numerical risks associated with forming the cross-product matrix $X^TX$ and ensures stability in ill-conditioned settings.
-   **Precision Control**: Added a `tol` (tolerance) parameter to define the threshold for numerical rank estimation during COD and SVD operations.

### Internal Improvements

-   **Template Metaprogramming**: Re-engineered core logic to utilize generic, templated C++ code, shifting significant computational evaluation to compile-time and reducing runtime overhead.
-   **C++17 Migration**: Upgraded the package build standard to C++17, enabling more expressive syntax and modern compiler optimizations.
-   **Memory Optimization**: Refactored multi-threaded workers to utilize pre-allocated buffers. By eliminating heap allocations within "hot loops" (specifically during data training and out-of-sample evaluation), the engine achieves significantly higher throughput and lower latency.
-   **Armadillo Expression Tuning**: Optimized the use of Armadillo expression templates to maximize lazy evaluation. This minimizes the creation of temporary objects and allows the compiler to generate more efficient SIMD-augmented computation loops.
-   **Comprehensive Testing Suite**:
    -   **R Integration**: Implemented extensive `testthat` suites to validate `cvLM` and `grid.search` against manual matrix algebra and established packages like `boot`.
    -   **Numerical Validation**: Tests specifically target edge cases including ill-conditioned, rank-deficient, and high-dimensional ($p > n$) datasets.
-    **Zero-Copy Interoperability**: Utilizes Armadillo’s advanced memory mapping to interface directly with R-allocated memory, ensuring zero-copy data passing between R and C++.
