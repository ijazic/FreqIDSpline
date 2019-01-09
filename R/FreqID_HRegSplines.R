#' The function to fit flexible parametric B-spline models for the frequentist anlaysis of semi-competing risks data
#' arising from cohort or nested case-control studies.
#'
#'
#' Independent semi-competing risks data can be analyzed using hierarchical models. As currently implemented, the semi-Markov 
#' assumption can be adopted for the conditional hazard function for time to the terminal event given time to non-terminal event.
#' This function is a flexible parametric analog of \code{FreqID_HReg} in the R package \code{SemiCompRisks} that
#' accommodates user-specified weights, permitting the analysis of semi-competing risks data arising from a nested case-
#' control study.
#' 
#'
#' @return \code{FreqID_HRegSplines} returns a list with two elements. The first element provides maximum likelihood
#' or maximum penalized likelihood estimates of the parameters, in the order 
#' (\eqn{\log\theta, \beta_1, \beta_2, \beta_3, \phi_1, \phi_2, \phi_3}). The second element is a vector of sandwich 
#' estimates of the standard error, in the same order.
#'  
#'
#' @examples
#'data(IDSplineData)
#'form <- Formula(y1 + delta1 | y2 + delta2 ~ cov1 + cov2 | cov1 + cov2 | cov1 + cov2)

#'### starting values for the baseline hazard components (phi)
#'### based on a spline approximation to a BayesID fit in SemiCompRisks. other strategies possible.
#'### each is length nKnots[i]+1
#'phi1.start <- c(-9.488466, -7.372098, -6.656212, -5.972695, -5.622274, -5.417296, -5.340710)
#'phi2.start <- c(-8.641889, -6.842977, -6.234473, -5.653484, -5.355626, -5.181395, -5.116297)
#'phi3.start <- c(-4.563376, -3.004653, -3.035086, -2.379959, -2.498012, -2.083923, -2.279812)

#'### starting values
#'startvals <- c(log(3), c(1,1), c(-1,-1), c(-0.5,-0.5), phi1.start, phi2.start, phi3.start) 
#'### 
#'fit1<- FreqID_HRegSplines(Formula=form, data=IDSplineData, wts=rep(1,nrow(IDSplineData)), 
#'nKnots=rep(6,3), startvals=startvals, penalty=FALSE)
#'
#' @param Formula a \code{Formula} object of the form \eqn{y_1+\delta_1 | y_2+\delta_2 \sim x_1 | x_2 | x_3}. 
#' See the \code{SemiCompRisks} package for more details.
#' @param data a data.frame in which to interpret the variables named in \code{Formula}.
#' @param wts a vector of weights for each individual in \code{data}. Must be of length \code{nrow(data)}.
#' @param nKnots a vector of length 3 giving the number of knots used in the B-spline specification of each baseline hazard. The
#' length of \eqn{\phi_i, i\in\{1,2,3\}} in \code{startvals} is equal to \code{nKnots[i]+1}, which includes an intercept.
#' @param startvals a vector of starting values in the order (\eqn{\log\theta, \beta_1, \beta_2, \beta_3,
#' \phi_1, \phi_2, \phi_3}), where \eqn{\beta} represent regression coefficients and \eqn{\phi} represent the B-spline coefficients.
#' @param penalty an indicator of whether penalized maximum likelihood estimates are to be used. Defaults to \code{FALSE}.
#' @param Kappa a vector of length 3 providing values of smoothing parameters. Defaults to \code{c(0,0,0)}.
#' @param na.action how NAs are treated. See \code{model.frame}.
#' @param subset a specification of the rows to be used: defaults to all rows. See \code{model.frame}.
#' 
#' @rdname FreqID_HRegSplines
#' @export
#######################################################################################################

FreqID_HRegSplines <- function(Formula, data, wts, nKnots, startvals, penalty=FALSE, Kappa=c(0,0,0), na.action = "na.fail",
                               subset=NULL)
{
  if(na.action != "na.fail" & na.action != "na.omit")
  {
    stop("na.action should be either na.fail or na.omit")
  }
  if(length(wts)!=nrow(data)){
    stop("wts must have a length equal to the number of rows in data")
  }
  
  form2 <- as.Formula(paste(Formula[2], Formula[1], Formula[3], sep = ""))    
  data <- model.frame(form2, data=data, na.action = na.action, subset = subset)
  
  ##
  time1 <- model.part(Formula, data=data, lhs=1)
  time2 <- model.part(Formula, data=data, lhs=2)
  
  Y <- cbind(time1[1], time1[2], time2[1], time2[2])
  y1     <- as.vector(Y[,1])
  delta1 <- as.vector(Y[,2])
  y2     <- as.vector(Y[,3])
  delta2 <- as.vector(Y[,4])
  
  Xmat1 <- model.frame(formula(Formula, lhs=0, rhs=1), data=data)
  Xmat2 <- model.frame(formula(Formula, lhs=0, rhs=2), data=data)
  Xmat3 <- model.frame(formula(Formula, lhs=0, rhs=3), data=data)
  
  db1 <- ncol(Xmat1); db2 <- ncol(Xmat2); db3 <- ncol(Xmat3)
  b.len <- db1+db2+db3 # number of beta coefficients
    ##
    ### predicted splines for each of the 3 baseline hazards. 
    b1 <- bSpline(y1, knots=seq(0,max(y1)+0.01, length=(nKnots[1]-1))[2:(nKnots[1]-2)], degree = 3, intercept=TRUE, Boundary.knots=c(0,max(y1)+0.01))
    b2 <- bSpline(y1, knots=seq(0,max(y1)+0.01, length=(nKnots[2]-1))[2:(nKnots[2]-2)], degree = 3, intercept=TRUE, Boundary.knots=c(0,max(y1)+0.01))
    ## semi-Markov
    b3 <- bSpline((y2-y1)[delta1==1], knots=seq(-0.01,max(y2-y1)+0.01, length=(nKnots[3]-1))[2:(nKnots[3]-2)], degree = 3, intercept=TRUE, Boundary.knots=c(-0.01,max(y2-y1)+0.01))
    b1.pred <- predict(b1, y1)
    b2.pred <- predict(b2, y1)
    b3.pred <- predict(b3, y2-y1)
    
    ### code for computing penalty components involving 2nd derivatives of B-splines (see SM A.4)
    if(penalty==TRUE){
      Kappa1 <- Kappa[1]; Kappa2 <- Kappa[2]; Kappa3 <- Kappa[3]
      b1.deriv.2 <- bSpline(y1, knots=seq(0,max(y1)+0.01, length=(nKnots[1]-1))[2:(nKnots[1]-2)], degree = 3, intercept=TRUE, Boundary.knots=c(0,max(y1)+0.01), derivs=2)
      b2.deriv.2 <- bSpline(y1, knots=seq(0,max(y1)+0.01, length=(nKnots[2]-1))[2:(nKnots[2]-2)], degree = 3, intercept=TRUE, Boundary.knots=c(0,max(y1)+0.01), derivs=2)
      b3.deriv.2 <- bSpline((y2-y1)[delta1==1], knots=seq(-0.01,max(y2-y1)+0.01, length=(nKnots[3]-1))[2:(nKnots[3]-2)], degree = 3, intercept=TRUE, Boundary.knots=c(-0.01,max(y2-y1)+0.01), derivs=2)
      
      prod.fn <- Vectorize(function(val, p1, p2, h){
        pred <- switch(h, "1" = predict(b1.deriv.2, val), "2" = predict(b2.deriv.2, val), "3" = predict(b3.deriv.2, val))
        return(pred[p1]*pred[p2])
      }, vectorize.args="val")
      
      pen.int.fn <- function(pos1, pos2, haz){
        new.prod.fn <- function(val){
          prod.fn(val, p1=pos1, p2=pos2, h=haz)
        }
        ul <- ifelse(haz==3, max(y2-y1)+0.01, max(y1)+0.01)
        return(integrate(new.prod.fn, lower=0, upper=ul))
      }
      
      pen.int.mat <- lapply(1:3, function(gg) matrix(NA, nrow=(nKnots[gg]+1), ncol=(nKnots[gg]+1)))
      
      for(hazard in 1:3){
        combos <- cbind(sapply(1:(nKnots[hazard]+1), function(i) c(i,i)), combn(c(1:(nKnots[hazard]+1)),2))
        for(col in 1:ncol(combos)){
          pen.int.mat[[hazard]][combos[1,col],combos[2,col]] <- pen.int.fn(pos1=combos[1,col], pos2=combos[2,col], haz=hazard)$value
          if(col>nKnots[hazard]){
            pen.int.mat[[hazard]][combos[2,col],combos[1,col]] <- pen.int.mat[[hazard]][combos[1,col],combos[2,col]]
          }
        }
      }
      
    } else{
      pen.int.mat <- list(0,0,0)
      Kappa1=0; Kappa2=0; Kappa3=0
    }	
    
    fit0 <- try(nleqslv(x = startvals, fn = spModelAll, method="Newton", jacobian=TRUE, control=list(maxit=250),
                        y1      = as.double(y1),
                        y2      = as.double(y2),
                        delta1      = as.integer(delta1),
                        delta2      = as.integer(delta2),
                        x1          = as.matrix(Xmat1),
                        x2          = as.matrix(Xmat2),
                        x3          = as.matrix(Xmat3),
                        wts         = as.double(wts),
                        m1pred   = as.matrix(b1.pred),
                        m2pred   = as.matrix(b2.pred),
                        m3pred   = as.matrix(b3.pred),
                        penalty = penalty,
                        penaltyMat1 = as.matrix(pen.int.mat[[1]]),
                        penaltyMat2 = as.matrix(pen.int.mat[[2]]),
                        penaltyMat3 = as.matrix(pen.int.mat[[3]]),
                        kappa1 = Kappa1,
                        kappa2 = Kappa2,
                        kappa3 = Kappa3))
  #}
  
  ##
  if(fit0$termcd==1)
  {
    varcov.mat <- sandwich.var(fit=fit0, y1=y1, y2=y2, delta1=delta1, delta2=delta2, 
                               x1=Xmat1, x2=Xmat2, x3=Xmat3, wts=wts, b1.pred=b1.pred, b2.pred=b2.pred, b3.pred=b3.pred)
    sdvec <- sqrt(diag(varcov.mat))
    ##
    
    return(list(fit0$x, sdvec))
  }
  
  ##
  #invisible()
}

