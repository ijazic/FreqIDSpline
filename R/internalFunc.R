## function for sandwich estimator. sandwichEstCheese() is an Rcpp export
sandwich.var <- function(fit, y1, y2, delta1, delta2, x1, x2, x3, wts, b1.pred, b2.pred, b3.pred){
  x <- fit$x
  jac <- fit$jac
  A.inv.mat <- solve(jac)
  
  B.mat <- sandwichEstCheese(xvec = as.double(x),
                             y1      = as.double(y1),
                             y2      = as.double(y2),
                             delta1      = as.integer(delta1),
                             delta2      = as.integer(delta2),
                             x1          = as.matrix(x1),
                             x2          = as.matrix(x2),
                             x3          = as.matrix(x3),
                             wts         = as.double(wts),
                             m1pred   = as.matrix(b1.pred),
                             m2pred   = as.matrix(b2.pred),
                             m3pred   = as.matrix(b3.pred))
  
  varcov.mat <- A.inv.mat%*%B.mat%*%A.inv.mat
  return(varcov.mat)
}