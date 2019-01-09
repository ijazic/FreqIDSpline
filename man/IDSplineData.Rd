\name{IDSplineData}
\alias{IDSplineData}
\docType{data}
\title{
  A simulated semi-competing risks data set
}
\description{
  Simulated semi-competing risks data
}
\usage{data(IDSplineData)}
\format{
  a data frame with 10000 observations on the following 14 variables.
  \describe{
    \item{\code{y1}}{the time to non-terminal event}
    \item{\code{delta1}}{the censoring indicators for the non-terminal event time; 1=event observed, 0=censored/truncated}
    \item{\code{y2}}{the time to terminal event}
    \item{\code{delta2}}{the censoring indicators for the terminal event time; 1=event observed, 0=censored}
    \item{\code{cov1}}{a vector of binary covariates}
    \item{\code{cov2}}{a vector of binary covariates}
  }
}
%\details{
  %%  ~~ If necessary, more details than the __description__ above ~~
    %}
%\source{
  %%  ~~ reference to a publication or URL from which the data were obtained ~~
    %}
%\references{
  %%  ~~ possibly secondary sources and usages ~~
    %}
\examples{
  data(IDSplineData)
}
\keyword{datasets}
