# test_that("grid.search returns consistent CV value", {
#   dat <- make_data()
#   
#   grid <- grid.search(
#     dat$formula,
#     dat$df,
#     K = 5,
#     max.lambda = 5,
#     precision = 0.5,
#     seed = 123
#   )
#   
#   cv_fixed <- cvLM(
#     dat$formula,
#     dat$df,
#     K.vals = 5,
#     lambda = grid$lambda,
#     seed = 123
#   )$CV
#   
#   expect_equal(grid$CV, cv_fixed, tolerance = 1e-10)
# })
