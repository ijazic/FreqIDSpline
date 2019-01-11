// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-
//
// scoreEqs.cpp: Score equations for the illness-death model adopting a flexible 
// 				 B-spline specification of the baseline hazards (optionally penalized)


#include <RcppGSL.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

double loglam0fn(int rowInd, RcppGSL::matrix<double> G_mpred, RcppGSL::vector<double> G_eta){
	double val_out;
	RcppGSL::VectorView mInd = gsl_matrix_row(G_mpred, rowInd);
	double dotProd = 0;
	gsl_blas_ddot(mInd, G_eta, &dotProd);
	val_out = dotProd;
	return val_out;
}

double loglam0penalfn(int rowInd, RcppGSL::matrix<double> G_mderiv2pred, RcppGSL::vector<double> G_eta){
	double val_out;
	RcppGSL::VectorView mInd = gsl_matrix_row(G_mderiv2pred, rowInd);
	double dotProd = 0;
	gsl_blas_ddot(mInd, G_eta, &dotProd);
	val_out = pow(dotProd,2);
	return val_out;
}

double lam0fn(int rowInd, RcppGSL::matrix<double> G_mpred, RcppGSL::vector<double> G_eta){
	double val_out;
	RcppGSL::VectorView mInd = gsl_matrix_row(G_mpred, rowInd);
	double dotProd = 0;
	gsl_blas_ddot(mInd, G_eta, &dotProd);
	val_out = exp(dotProd);
	return val_out;
}

double lam0fnEta(int rowInd, int etaInd, RcppGSL::matrix<double> G_mpred, RcppGSL::vector<double> G_eta){
	double val_out;
	RcppGSL::VectorView mInd = gsl_matrix_row(G_mpred, rowInd);
	double dotProd = 0;
	gsl_blas_ddot(mInd, G_eta, &dotProd);
	val_out = exp(dotProd)*mInd[etaInd];
	return val_out;
}

// score equation u1: individual
double u1_ind(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   Rcpp::NumericVector wts, Rcpp::NumericVector AVec, Rcpp::NumericVector commonVec, double h)
{
	double scoreValSum;
	double term1 = delta1[ind] * delta2[ind]/(1+exp(h));
	double term2 = log(1+exp(h) * AVec[ind])/exp(2*h);
	double termLastUnique = AVec[ind];
	scoreValSum = exp(h)*wts[ind]*(term1+term2-commonVec[ind]*termLastUnique);
	return scoreValSum;
} 

// score equation u1: all
double u1(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   Rcpp::NumericVector wts, Rcpp::NumericVector AVec, Rcpp::NumericVector commonVec, double h, int nObs)
{
	std::vector<double> scoreVals(nObs);
	double scoreValSum, term1, term2, termLastUnique;
	int ind;
	for(ind = 0; ind < nObs; ind++)
		{
			term1 = delta1[ind] * delta2[ind]/(1+exp(h));
			term2 = log(1+exp(h) * AVec[ind])/exp(2*h);
			termLastUnique = AVec[ind];
			scoreVals[ind] = exp(h)*wts[ind]*(term1+term2-commonVec[ind]*termLastUnique);
		}
	scoreValSum =  std::accumulate(scoreVals.begin(), scoreVals.end(), 0.0);
	return scoreValSum;
}


// score equation u2: individual
std::vector<double> u2_ind(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x1, RcppGSL::vector<double> G_beta1, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda01, Rcpp::NumericVector commonVec, double h, int nCov1)
{
	std::vector<double> scoreVals(nCov1);
	double cov1;
	RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
	double beta1Tx=0; 
	gsl_blas_ddot(G_beta1, x1Ind, &beta1Tx);
	double coef = wts[ind]*(delta1[ind] - commonVec[ind]*Lambda01[ind]*exp(h+beta1Tx));
	for(cov1 = 0; cov1 < nCov1; cov1++)
		{
			scoreVals[cov1] = coef*x1Ind[cov1];
		}
	return scoreVals;	
}

// score equation u2: all
std::vector<double> u2(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x1, RcppGSL::vector<double> G_beta1, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda01, Rcpp::NumericVector commonVec, double h, int nCov1, int nObs)
{
	std::vector<double> scoreVals(nCov1);
	std::vector<double> scoreValsOld(nCov1);
	double cov1, ind, beta1Tx, coef;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
			beta1Tx=0; 
			gsl_blas_ddot(G_beta1, x1Ind, &beta1Tx);
			coef = wts[ind]*(delta1[ind] - commonVec[ind]*Lambda01[ind]*exp(h+beta1Tx));
			for(cov1 = 0; cov1 < nCov1; cov1++)
				{
					scoreVals[cov1] = scoreValsOld[cov1]+coef*x1Ind[cov1];
					scoreValsOld[cov1] = scoreVals[cov1];
				}
		}		
	return scoreVals;	
}

/*// [[Rcpp::export]]
Rcpp::NumericVector u2_indA(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   Rcpp::NumericMatrix x1, Rcpp::NumericVector beta1, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda01, Rcpp::NumericVector commonVec, double h, int nCov)
{
	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);
	RcppGSL::vector<double> G_beta1 = Rcpp::as< RcppGSL::vector<double> >(beta1);

	double cov1;
	Rcpp::NumericVector scoreValSum(nCov);
	RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
	double beta1Tx=0; 
	gsl_blas_ddot(G_beta1, x1Ind, &beta1Tx);
	double coef = wts[ind]*(delta1[ind] - commonVec[ind]*Lambda01[ind]*exp(h+beta1Tx));
	for(cov1 = 0; cov1 < nCov; cov1++)
		{
			scoreValSum[cov1] = coef*x1Ind[cov1];
		}
	G_x1.free();
	G_beta1.free();	
	return scoreValSum;
}*/

// score equation u3: individual
std::vector<double> u3_ind(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x2, RcppGSL::vector<double> G_beta2, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda02, Rcpp::NumericVector commonVec, double h, int nCov2)
{
	std::vector<double> scoreVals(nCov2);
	double cov2;
	RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, ind);
	double beta2Tx=0; 
	gsl_blas_ddot(G_beta2, x2Ind, &beta2Tx);
	double coef = wts[ind]*((1-delta1[ind])*delta2[ind] - commonVec[ind]*Lambda02[ind]*exp(h+beta2Tx));
	for(cov2 = 0; cov2 < nCov2; cov2++)
		{
			scoreVals[cov2] = coef*x2Ind[cov2];
		}
	return scoreVals;	
}

// score equation u3: all
std::vector<double> u3(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x2, RcppGSL::vector<double> G_beta2, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda02, Rcpp::NumericVector commonVec, double h, int nCov2, int nObs)
{
	std::vector<double> scoreVals(nCov2);
	std::vector<double> scoreValsOld(nCov2);
	double cov2, ind, beta2Tx, coef;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, ind);
			beta2Tx=0; 
			gsl_blas_ddot(G_beta2, x2Ind, &beta2Tx);
			coef = wts[ind]*((1-delta1[ind])*delta2[ind] - commonVec[ind]*Lambda02[ind]*exp(h+beta2Tx));
			for(cov2 = 0; cov2 < nCov2; cov2++)
				{
					scoreVals[cov2] = scoreValsOld[cov2]+coef*x2Ind[cov2];
					scoreValsOld[cov2] = scoreVals[cov2];
				}
		}		
	return scoreVals;	
}

// score equation u4: individual
std::vector<double> u4_ind(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x3, RcppGSL::vector<double> G_beta3, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda03, Rcpp::NumericVector commonVec, double h, int nCov3)
{
	std::vector<double> scoreVals(nCov3);
	double cov3;
	RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, ind);
	double beta3Tx=0; 
	gsl_blas_ddot(G_beta3, x3Ind, &beta3Tx);
	double coef = wts[ind]*(delta1[ind]*delta2[ind] - commonVec[ind]*Lambda03[ind]*exp(h+beta3Tx));
	for(cov3 = 0; cov3 < nCov3; cov3++)
		{
			scoreVals[cov3] = coef*x3Ind[cov3];
		}
	return scoreVals;	
}

// score equation u4: all
std::vector<double> u4(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x3, RcppGSL::vector<double> G_beta3, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda03, Rcpp::NumericVector commonVec, double h, int nCov3, int nObs)
{
	std::vector<double> scoreVals(nCov3);
	std::vector<double> scoreValsOld(nCov3);
	double cov3, ind, beta3Tx, coef;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, ind);
			beta3Tx=0; 
			gsl_blas_ddot(G_beta3, x3Ind, &beta3Tx);
			coef = wts[ind]*(delta1[ind]*delta2[ind] - commonVec[ind]*Lambda03[ind]*exp(h+beta3Tx));
			for(cov3 = 0; cov3 < nCov3; cov3++)
				{
					scoreVals[cov3] = scoreValsOld[cov3]+coef*x3Ind[cov3];
					scoreValsOld[cov3] = scoreVals[cov3];
				}
		}		
	return scoreVals;	
}

// score equation u5: individual
std::vector<double> u5_ind_nopenal(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x1, RcppGSL::vector<double> G_beta1, RcppGSL::vector<double> G_eta1, RcppGSL::matrix<double> G_Lambda01Eta, 
			   RcppGSL::matrix<double> G_m1pred, Rcpp::NumericVector wts, Rcpp::NumericVector commonVec, double h, int nKnots1)
{
	std::vector<double> scoreVals(nKnots1);
	double spl1;
	RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
	RcppGSL::VectorView m1Ind = gsl_matrix_row(G_m1pred, ind); 
	RcppGSL::VectorView Lambda01EtaInd = gsl_matrix_row(G_Lambda01Eta, ind); 
	double beta1Tx=0; 
	gsl_blas_ddot(G_beta1, x1Ind, &beta1Tx);
	double coef1 = delta1[ind];
	double coef2 = commonVec[ind]*exp(h+beta1Tx);
	for(spl1 = 0; spl1 < nKnots1; spl1++)
		{
			scoreVals[spl1]=wts[ind]*(coef1*m1Ind[spl1] - coef2*Lambda01EtaInd[spl1]);
		}
	
	return scoreVals;	
}

// score equation u5: all, no penalty
std::vector<double> u5_nopenal(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x1, RcppGSL::vector<double> G_beta1, RcppGSL::vector<double> G_eta1, RcppGSL::matrix<double> G_Lambda01Eta, 
			   RcppGSL::matrix<double> G_m1pred, Rcpp::NumericVector wts, Rcpp::NumericVector commonVec, double h, int nKnots1, int nObs)
{
	std::vector<double> scoreVals(nKnots1);
	std::vector<double> scoreValsOld(nKnots1);
	double spl1, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
			RcppGSL::VectorView m1Ind = gsl_matrix_row(G_m1pred, ind); 
			RcppGSL::VectorView Lambda01EtaInd = gsl_matrix_row(G_Lambda01Eta, ind); 
			double beta1Tx=0; 
			gsl_blas_ddot(G_beta1, x1Ind, &beta1Tx);
			coef1 = delta1[ind];
			coef2 = commonVec[ind]*exp(h+beta1Tx);
			for(spl1 = 0; spl1 < nKnots1; spl1++)
				{
					scoreVals[spl1]=scoreValsOld[spl1]+wts[ind]*(coef1*m1Ind[spl1] - coef2*Lambda01EtaInd[spl1]);
					scoreValsOld[spl1] = scoreVals[spl1];
				}
		}
	
	return scoreVals;		
}

// score equation u5: all, penalty
std::vector<double> u5_penal(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x1, RcppGSL::vector<double> G_beta1, RcppGSL::vector<double> G_eta1, RcppGSL::matrix<double> G_Lambda01Eta, 
			   RcppGSL::matrix<double> G_m1pred, Rcpp::NumericVector wts, bool penalty, double kappa1, RcppGSL::matrix<double> G_penaltyMat1, 
			   Rcpp::NumericVector commonVec, double h, int nKnots1, int nObs)
{
	std::vector<double> scoreVals(nKnots1);
	std::vector<double> scoreValsOld(nKnots1);
	double spl1, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
			RcppGSL::VectorView m1Ind = gsl_matrix_row(G_m1pred, ind); 
			RcppGSL::VectorView Lambda01EtaInd = gsl_matrix_row(G_Lambda01Eta, ind); 
			double beta1Tx=0; 
			gsl_blas_ddot(G_beta1, x1Ind, &beta1Tx);
			coef1 = delta1[ind];
			coef2 = commonVec[ind]*exp(h+beta1Tx);
			for(spl1 = 0; spl1 < nKnots1; spl1++)
				{
					scoreVals[spl1]=scoreValsOld[spl1]+wts[ind]*(coef1*m1Ind[spl1] - coef2*Lambda01EtaInd[spl1]);
					scoreValsOld[spl1] = scoreVals[spl1];
				}
		}		
 	if(penalty==TRUE)
		{
			for(spl1 = 0; spl1 < nKnots1; spl1++)
				{
					RcppGSL::VectorView penaltySpl = gsl_matrix_row(G_penaltyMat1, spl1);
					double dotprod;
					gsl_blas_ddot(G_eta1, penaltySpl, &dotprod);
					scoreVals[spl1] = scoreValsOld[spl1]-2*kappa1*dotprod;
				}
			return scoreVals;	
		} else
		{
			return scoreVals;
		}
}

// score equation u6: individual
std::vector<double> u6_ind_nopenal(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x2, RcppGSL::vector<double> G_beta2, RcppGSL::vector<double> G_eta2, RcppGSL::matrix<double> G_Lambda02Eta, 
			   RcppGSL::matrix<double> G_m2pred, Rcpp::NumericVector wts, Rcpp::NumericVector commonVec, double h, int nKnots2)
{
	std::vector<double> scoreVals(nKnots2);
	double spl2;
	RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, ind);
	RcppGSL::VectorView m2Ind = gsl_matrix_row(G_m2pred, ind); 
	RcppGSL::VectorView Lambda02EtaInd = gsl_matrix_row(G_Lambda02Eta, ind); 
	double beta2Tx=0; 
	gsl_blas_ddot(G_beta2, x2Ind, &beta2Tx);
	double coef1 = (1-delta1[ind])*delta2[ind];
	double coef2 = commonVec[ind]*exp(h+beta2Tx);
	for(spl2 = 0; spl2 < nKnots2; spl2++)
		{
			scoreVals[spl2]=wts[ind]*(coef1*m2Ind[spl2] - coef2*Lambda02EtaInd[spl2]);
		}
	
	return scoreVals;	
}

// score equation u6: all, no penalty
std::vector<double> u6_nopenal(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x2, RcppGSL::vector<double> G_beta2, RcppGSL::vector<double> G_eta2, RcppGSL::matrix<double> G_Lambda02Eta, 
			   RcppGSL::matrix<double> G_m2pred, Rcpp::NumericVector wts, Rcpp::NumericVector commonVec, double h, int nKnots2, int nObs)
{
	std::vector<double> scoreVals(nKnots2);
	std::vector<double> scoreValsOld(nKnots2);
	double spl2, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, ind);
			RcppGSL::VectorView m2Ind = gsl_matrix_row(G_m2pred, ind); 
			RcppGSL::VectorView Lambda02EtaInd = gsl_matrix_row(G_Lambda02Eta, ind); 
			double beta2Tx=0; 
			gsl_blas_ddot(G_beta2, x2Ind, &beta2Tx);
			coef1 = (1-delta1[ind])*delta2[ind];
			coef2 = commonVec[ind]*exp(h+beta2Tx);
			for(spl2 = 0; spl2 < nKnots2; spl2++)
				{
					scoreVals[spl2]=scoreValsOld[spl2]+wts[ind]*(coef1*m2Ind[spl2] - coef2*Lambda02EtaInd[spl2]);
					scoreValsOld[spl2] = scoreVals[spl2];
				}
		}
	
	return scoreVals;		
}

// score equation u6: all, penalty
std::vector<double> u6_penal(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x2, RcppGSL::vector<double> G_beta2, RcppGSL::vector<double> G_eta2, RcppGSL::matrix<double> G_Lambda02Eta, 
			   RcppGSL::matrix<double> G_m2pred, Rcpp::NumericVector wts, bool penalty, double kappa2, RcppGSL::matrix<double> G_penaltyMat2, 
			   Rcpp::NumericVector commonVec, double h, int nKnots2, int nObs)
{
	std::vector<double> scoreVals(nKnots2);
	std::vector<double> scoreValsOld(nKnots2);
	double spl2, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, ind);
			RcppGSL::VectorView m2Ind = gsl_matrix_row(G_m2pred, ind); 
			RcppGSL::VectorView Lambda02EtaInd = gsl_matrix_row(G_Lambda02Eta, ind); 
			double beta2Tx=0; 
			gsl_blas_ddot(G_beta2, x2Ind, &beta2Tx);
			coef1 = (1-delta1[ind])*delta2[ind];
			coef2 = commonVec[ind]*exp(h+beta2Tx);
			for(spl2 = 0; spl2 < nKnots2; spl2++)
				{
					scoreVals[spl2]=scoreValsOld[spl2]+wts[ind]*(coef1*m2Ind[spl2] - coef2*Lambda02EtaInd[spl2]);
					scoreValsOld[spl2] = scoreVals[spl2];
				}
		}		
 	if(penalty==TRUE)
		{
			for(spl2 = 0; spl2 < nKnots2; spl2++)
				{
					RcppGSL::VectorView penaltySpl = gsl_matrix_row(G_penaltyMat2, spl2);
					double dotprod;
					gsl_blas_ddot(G_eta2, penaltySpl, &dotprod);
					scoreVals[spl2] = scoreValsOld[spl2]-2*kappa2*dotprod;
				}
			return scoreVals;	
		} else
		{
			return scoreVals;
		}
}

// score equation u7: individual
std::vector<double> u7_ind_nopenal(int ind, Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector yDiff, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x3, RcppGSL::vector<double> G_beta3, RcppGSL::vector<double> G_eta3, RcppGSL::matrix<double> G_Lambda03Eta, 
			   RcppGSL::matrix<double> G_m3pred, Rcpp::NumericVector wts, Rcpp::NumericVector commonVec, double h, int nKnots3)
{
	std::vector<double> scoreVals(nKnots3);
	double spl3;
	RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, ind);
	RcppGSL::VectorView m3Ind = gsl_matrix_row(G_m3pred, ind); 
	RcppGSL::VectorView Lambda03EtaInd = gsl_matrix_row(G_Lambda03Eta, ind); 
	double beta3Tx=0; 
	gsl_blas_ddot(G_beta3, x3Ind, &beta3Tx);
	double coef1 = delta1[ind]*delta2[ind];
	double coef2 = commonVec[ind]*exp(h+beta3Tx);
	for(spl3 = 0; spl3 < nKnots3; spl3++)
		{
			scoreVals[spl3]=wts[ind]*(coef1*m3Ind[spl3] - coef2*Lambda03EtaInd[spl3]);
		}
	
	return scoreVals;	
}

// score equation u7: all, no penalty
std::vector<double> u7_nopenal(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector yDiff, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x3, RcppGSL::vector<double> G_beta3, RcppGSL::vector<double> G_eta3, RcppGSL::matrix<double> G_Lambda03Eta, 
			   RcppGSL::matrix<double> G_m3pred, Rcpp::NumericVector wts, Rcpp::NumericVector commonVec, double h, int nKnots3, int nObs)
{
	std::vector<double> scoreVals(nKnots3);
	std::vector<double> scoreValsOld(nKnots3);
	double spl3, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			if(yDiff[ind]!=0)
				{
					RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, ind);
					RcppGSL::VectorView m3Ind = gsl_matrix_row(G_m3pred, ind); 
					RcppGSL::VectorView Lambda03EtaInd = gsl_matrix_row(G_Lambda03Eta, ind); 
					double beta3Tx=0; 
					gsl_blas_ddot(G_beta3, x3Ind, &beta3Tx);
					coef1 = delta1[ind]*delta2[ind];
					coef2 = commonVec[ind]*exp(h+beta3Tx);
					for(spl3 = 0; spl3 < nKnots3; spl3++)
						{
							scoreVals[spl3]=scoreValsOld[spl3]+wts[ind]*(coef1*m3Ind[spl3] - coef2*Lambda03EtaInd[spl3]);
							scoreValsOld[spl3] = scoreVals[spl3];
						}
				}
		}
	
	return scoreVals;		
}


// score equation u7: all, penalty
std::vector<double> u7_penal(Rcpp::NumericVector y1, Rcpp::NumericVector y2, Rcpp::NumericVector yDiff, Rcpp::NumericVector delta1, Rcpp::NumericVector delta2,
			   RcppGSL::matrix<double> G_x3, RcppGSL::vector<double> G_beta3, RcppGSL::vector<double> G_eta3, RcppGSL::matrix<double> G_Lambda03Eta, 
			   RcppGSL::matrix<double> G_m3pred, Rcpp::NumericVector wts, bool penalty, double kappa3, RcppGSL::matrix<double> G_penaltyMat3, 
			   Rcpp::NumericVector commonVec, double h, int nKnots3, int nObs)
{
	std::vector<double> scoreVals(nKnots3);
	std::vector<double> scoreValsOld(nKnots3);
	double spl3, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			if(yDiff[ind]!=0)
				{
					RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, ind);
					RcppGSL::VectorView m3Ind = gsl_matrix_row(G_m3pred, ind); 
					RcppGSL::VectorView Lambda03EtaInd = gsl_matrix_row(G_Lambda03Eta, ind); 
					double beta3Tx=0; 
					gsl_blas_ddot(G_beta3, x3Ind, &beta3Tx);
					coef1 = delta1[ind]*delta2[ind];
					coef2 = commonVec[ind]*exp(h+beta3Tx);
					for(spl3 = 0; spl3 < nKnots3; spl3++)
						{
							scoreVals[spl3]=scoreValsOld[spl3]+wts[ind]*(coef1*m3Ind[spl3] - coef2*Lambda03EtaInd[spl3]);
							scoreValsOld[spl3] = scoreVals[spl3];
						}
				}
		}		
 	if(penalty==TRUE)
		{
			for(spl3 = 0; spl3 < nKnots3; spl3++)
				{
					RcppGSL::VectorView penaltySpl = gsl_matrix_row(G_penaltyMat3, spl3);
					double dotprod;
					gsl_blas_ddot(G_eta3, penaltySpl, &dotprod);
					scoreVals[spl3] = scoreValsOld[spl3]-2*kappa3*dotprod;
				}
			return scoreVals;	
		} else
		{
			return scoreVals;
		}
}

// sorts vector
Rcpp::NumericVector RcppSort(Rcpp::NumericVector x1) {
   Rcpp::NumericVector y = clone(x1);
   std::sort(y.begin(), y.end());
   return y;
}

// sorts vector x in order of vector y
Rcpp::NumericVector sortByVec(Rcpp::NumericVector x1, Rcpp::NumericVector y) {
    // First create a vector of indices
    Rcpp::IntegerVector idx = seq_along(x1) - 1;
    // Then sort that vector by the values of y
    std::sort(idx.begin(), idx.end(), [&](int i, int j){return y[i] < y[j];});
    // And return x in that order
    return x1[idx];
}

	
	
// score equation u1: all
double u1Ex(Rcpp::NumericVector xVals, double a, double b, int nObs)
{
	std::vector<double> scoreVals(nObs);
	double scoreValSum, val;
	int ind;
	for(ind = 0; ind < nObs; ind++)
		{
			scoreVals[ind] = xVals[ind];
		}
	scoreValSum =  std::accumulate(scoreVals.begin(), scoreVals.end(), 0.0);
	val = scoreValSum*a + pow(b,2)/2 - 5;
	return val;
}



// score equation u2: all
double u2Ex(Rcpp::NumericVector xvals, double a, double b, int nObs)
{
	std::vector<double> scoreVals(nObs);
	double scoreValSum, val;
	int ind;
	for(ind = 0; ind < nObs; ind++)
		{
			scoreVals[ind] = -log(xvals[ind]);
		}
	scoreValSum =  std::accumulate(scoreVals.begin(), scoreVals.end(), 0.0);
	val = scoreValSum*a + b;
	return val;
}



// possibly extend later to different knots for different HRss

// [[Rcpp::export]] 
Rcpp::NumericVector spModelAll(Rcpp::NumericVector xvec, const Rcpp::NumericVector y1, const Rcpp::NumericVector y2, 
					Rcpp::NumericVector delta1, Rcpp::NumericVector delta2, Rcpp::NumericMatrix x1, Rcpp::NumericMatrix x2, 
					Rcpp::NumericMatrix x3, Rcpp::NumericVector wts, Rcpp::NumericMatrix m1pred, 
					Rcpp::NumericMatrix m2pred, Rcpp::NumericMatrix m3pred, bool penalty, Rcpp::NumericMatrix penaltyMat1, 
					Rcpp::NumericMatrix penaltyMat2, Rcpp::NumericMatrix penaltyMat3, double kappa1, double kappa2, double kappa3)
{
	int aa, gg1, gg2, gg3, i, ii, iia, iib, ii2, jj, jja, jjb, jj2, kk;
	int nObs, nCov1, nCov2, nCov3, nKnots1, nKnots2, nKnots3;
	
	nObs = x1.nrow(); 
	nCov1 = x1.ncol(); 
	nCov2 = x2.ncol(); 
	nCov3 = x3.ncol();
	nKnots1 = m1pred.ncol(); 
	nKnots2 = m2pred.ncol(); 
	nKnots3 = m3pred.ncol();
	
	double h(1);//, a(1), b(1);
	Rcpp::NumericVector beta1(nCov1), beta2(nCov2), beta3(nCov3), eta1(nKnots1), eta2(nKnots2), eta3(nKnots3);
	 
	// initialize all parameters
	h = xvec[0];
	for(ii = 0; ii < nCov1; ii++){
  		beta1[ii] = xvec[1+ ii];
  	}
  	for(iia = 0; iia < nCov2; iia++){
  		beta2[iia] = xvec[1+nCov1+ iia];
  	}
  	for(iib = 0; iib < nCov3; iib++){
  		beta3[iib] = xvec[1+nCov1+nCov2+ iib];
  	}
	for(jj = 0; jj < nKnots1; jj++){
  		eta1[jj] = xvec[1+nCov1+nCov2+nCov3+ jj];
  	}
  	for(jja = 0; jja < nKnots2; jja++){
  		eta2[jja] = xvec[1+nCov1+nCov2+nCov3+nKnots1+ jja];
  	}
  	for(jjb = 0; jjb < nKnots3; jjb++){
  		eta3[jjb] = xvec[1+nCov1+nCov2+nCov3+nKnots1+nKnots2+ jjb];
  	}

 	// make GSL versions as necessary
 	RcppGSL::vector<double> G_beta1 = Rcpp::as< RcppGSL::vector<double> >(beta1);
 	RcppGSL::vector<double> G_beta2 = Rcpp::as< RcppGSL::vector<double> >(beta2);
 	RcppGSL::vector<double> G_beta3 = Rcpp::as< RcppGSL::vector<double> >(beta3);
 	
 	RcppGSL::vector<double> G_eta1 = Rcpp::as< RcppGSL::vector<double> >(eta1);
 	RcppGSL::vector<double> G_eta2 = Rcpp::as< RcppGSL::vector<double> >(eta2);
 	RcppGSL::vector<double> G_eta3 = Rcpp::as< RcppGSL::vector<double> >(eta3);
 	
 	RcppGSL::matrix<double> G_m1pred = Rcpp::as< RcppGSL::matrix<double> >(m1pred);
 	RcppGSL::matrix<double> G_m2pred = Rcpp::as< RcppGSL::matrix<double> >(m2pred);
 	RcppGSL::matrix<double> G_m3pred = Rcpp::as< RcppGSL::matrix<double> >(m3pred);
 	
 	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);
 	RcppGSL::matrix<double> G_x2 = Rcpp::as< RcppGSL::matrix<double> >(x2);
 	RcppGSL::matrix<double> G_x3 = Rcpp::as< RcppGSL::matrix<double> >(x3);
 	
 	RcppGSL::matrix<double> G_penaltyMat1 = Rcpp::as< RcppGSL::matrix<double> >(penaltyMat1);
 	RcppGSL::matrix<double> G_penaltyMat2 = Rcpp::as< RcppGSL::matrix<double> >(penaltyMat2);
 	RcppGSL::matrix<double> G_penaltyMat3 = Rcpp::as< RcppGSL::matrix<double> >(penaltyMat3);

    /**************************************************
    // Computing cumulative baseline hazard functions
    **************************************************/
 
 	// initialize vectors for BH and cumulative BH
 	Rcpp::NumericVector Lam01Terms(nObs), Lam02Terms(nObs), Lam03Terms(nObs);
	Rcpp::NumericVector Lambda01(nObs), Lambda02(nObs), Lambda03(nObs);
	
	Rcpp::NumericVector yDiff = y2-y1;
	Rcpp::NumericVector isDiffNonzero(nObs);
	int diffZeroCount = 0;
	
	// compute BH for each person
	for(i = 0; i < nObs; i++)
  	{
    	Lam01Terms[i] = lam0fn(i,G_m1pred, G_eta1);
    	Lam02Terms[i] = lam0fn(i,G_m2pred, G_eta2);
    	if(yDiff[i]==0) {
    		Lam03Terms[i] = 1;
    		isDiffNonzero[i] = 0;
    		diffZeroCount++;
    	} else {
    		Lam03Terms[i] = lam0fn(i,G_m3pred, G_eta3);
    		isDiffNonzero[i] = 1;
    	}		
  	}
  	
  	Rcpp::NumericVector y1Sort = RcppSort(y1);
  	Rcpp::NumericVector yDiffSort = RcppSort(yDiff);
  	Rcpp::NumericVector Lam01TermsSort = sortByVec(Lam01Terms, y1);
  	Rcpp::NumericVector Lam02TermsSort = sortByVec(Lam02Terms, y1);
  	Rcpp::NumericVector Lam03TermsSort = sortByVec(Lam03Terms, yDiff);
  	Rcpp::NumericVector yDiffSortNew(nObs-diffZeroCount), Lam03TermsSortNew(nObs-diffZeroCount);
  	  	
  	for(i = diffZeroCount; i < nObs; i++)
  	{
    	yDiffSortNew[i-diffZeroCount] = yDiffSort[i];
    	Lam03TermsSortNew[i-diffZeroCount] = Lam03TermsSort[i];	
  	}

  	// insert values for 0, and maximum values
  	y1Sort.insert(y1Sort.begin(), 0);
  	y1Sort.insert(y1Sort.end(), y1Sort[nObs]+0.1);
  	yDiffSortNew.insert(yDiffSortNew.begin(), 0);
  	yDiffSortNew.insert(yDiffSortNew.end(), yDiffSortNew[nObs-diffZeroCount]+0.1);
  	Lam01TermsSort.insert(Lam01TermsSort.begin(), Lam01TermsSort[0]);
  	Lam02TermsSort.insert(Lam02TermsSort.begin(), Lam02TermsSort[0]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.begin(), Lam03TermsSortNew[0]);
  	Lam01TermsSort.insert(Lam01TermsSort.end(), Lam01TermsSort[nObs]);
  	Lam02TermsSort.insert(Lam02TermsSort.end(), Lam02TermsSort[nObs]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.end(), Lam03TermsSortNew[nObs-diffZeroCount]);	
	
    // interpolation for Lambda01 
    gsl_interp_accel *acc01 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda01 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda01, y1Sort.begin(), Lam01TermsSort.begin(), y1Sort.size());
	
	double result01;
	for(gg1 = 0; gg1 < nObs; gg1++)
		{
			gsl_interp_eval_integ_e(interpLambda01,y1Sort.begin(),Lam01TermsSort.begin(),0,y1[gg1],acc01, &result01);
			Lambda01[gg1]=result01;
		}

	gsl_interp_free(interpLambda01);
	gsl_interp_accel_free(acc01); 
	
	// interpolation for Lambda02
    gsl_interp_accel *acc02 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda02 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda02, y1Sort.begin(), Lam02TermsSort.begin(), y1Sort.size());

	double result02;
	for(gg2 = 0; gg2 < nObs; gg2++)
		{
			gsl_interp_eval_integ_e(interpLambda02,y1Sort.begin(),Lam02TermsSort.begin(),0,y1[gg2],acc02, &result02);
			Lambda02[gg2]=result02;
		}

	gsl_interp_free(interpLambda02);
	gsl_interp_accel_free(acc02);

	
	// interpolation for Lambda03 
   	gsl_interp_accel *acc03 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda03 = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
	gsl_interp_init(interpLambda03, yDiffSortNew.begin(), Lam03TermsSortNew.begin(), yDiffSortNew.size());

	double result03;
	for(gg3 = 0; gg3 < nObs; gg3++)
		{
			if(yDiff[gg3]==0) {
    			Lambda03[gg3] = 0;
    		} else {
				gsl_interp_eval_integ_e(interpLambda03,yDiffSortNew.begin(),Lam03TermsSortNew.begin(),0,yDiff[gg3],acc03, &result03);
				Lambda03[gg3]=result03; 
			}	
		}

	gsl_interp_free(interpLambda03);
	gsl_interp_accel_free(acc03);
	
	
	/******************************************************************************************
    // Computing integrals that get used in the eta score equations (int lambda0j(s)*Bjl(s))
    ******************************************************************************************/
    
    // initialize vectors for terms (will get reused) and matrices for integrals themselves
 	Rcpp::NumericVector Lam01TermsEta(nObs), Lam02TermsEta(nObs), Lam03TermsEta(nObs);
 	Rcpp::NumericVector Lam01TermsEtaSort(nObs), Lam02TermsEtaSort(nObs), Lam03TermsEtaSort(nObs), Lam03TermsEtaSortNew(nObs-diffZeroCount);
	Rcpp::NumericMatrix Lambda01Eta(nObs,nKnots1), Lambda02Eta(nObs,nKnots2), Lambda03Eta(nObs,nKnots3);
	double result01Eta, result02Eta, result03Eta;
	
	// we still have yDiff, isDiffNonzero, and diffZeroCount from before
	// as well as y1Sort, yDiffSort, yDiffSortNew
	
	// all integrals for Lambda01
	for(kk = 0; kk < nKnots1; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		Lam01TermsEta[i] = lam0fnEta(i, kk, G_m1pred, G_eta1);	
  		}
  		
  		// fill sort vectors
  		Lam01TermsEtaSort = sortByVec(Lam01TermsEta, y1);
  		
  		// insert values for 0, and maximum values
  		Lam01TermsEtaSort.insert(Lam01TermsEtaSort.begin(), Lam01TermsEtaSort[0]);
  		Lam01TermsEtaSort.insert(Lam01TermsEtaSort.end(), Lam01TermsEtaSort[nObs]);

		// interpolation + integration for Lambda01Eta 
    	gsl_interp_accel *acc01Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda01Eta = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
		gsl_interp_init(interpLambda01Eta, y1Sort.begin(), Lam01TermsEtaSort.begin(), y1Sort.size());
	
		for(gg1 = 0; gg1 < nObs; gg1++)
			{
				gsl_interp_eval_integ_e(interpLambda01Eta,y1Sort.begin(),Lam01TermsEtaSort.begin(),0,y1[gg1],acc01Eta, &result01Eta);
				Lambda01Eta(gg1,kk)=result01Eta;
			}

		gsl_interp_free(interpLambda01Eta);
		gsl_interp_accel_free(acc01Eta); 	
	}	
	
	
	// all integrals for Lambda02
	for(kk = 0; kk < nKnots2; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		Lam02TermsEta[i] = lam0fnEta(i, kk, G_m2pred, G_eta2);	
  		}
  		
  		// fill sort vectors
  		Lam02TermsEtaSort = sortByVec(Lam02TermsEta, y1);
  		
  		// insert values for 0, and maximum values
  		Lam02TermsEtaSort.insert(Lam02TermsEtaSort.begin(), Lam02TermsEtaSort[0]);
  		Lam02TermsEtaSort.insert(Lam02TermsEtaSort.end(), Lam02TermsEtaSort[nObs]);
				
		// interpolation + integration for Lambda02Eta
    	gsl_interp_accel *acc02Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda02Eta = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
		gsl_interp_init(interpLambda02Eta, y1Sort.begin(), Lam02TermsEtaSort.begin(), y1Sort.size());

		for(gg2 = 0; gg2 < nObs; gg2++)
			{
				gsl_interp_eval_integ_e(interpLambda02Eta,y1Sort.begin(),Lam02TermsEtaSort.begin(),0,y1[gg2],acc02Eta, &result02Eta);
				Lambda02Eta(gg2,kk)=result02Eta;
			}

		gsl_interp_free(interpLambda02Eta);
		gsl_interp_accel_free(acc02Eta);
	}
	
	
	// all integrals for Lambda03
	for(kk = 0; kk < nKnots3; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		if(yDiff[i]==0) {
    			Lam03TermsEta[i] = 1;
    		} else {
    			Lam03TermsEta[i] = lam0fnEta(i, kk, G_m3pred, G_eta3);
    		}		
  		}
  		
  		// fill sort vectors
  		Lam03TermsEtaSort = sortByVec(Lam03TermsEta, yDiff);	
  		for(i = diffZeroCount; i < nObs; i++)
  		{
    		Lam03TermsEtaSortNew[i-diffZeroCount] = Lam03TermsEtaSort[i];	
  		}
  		
  		// insert values for 0, and maximum values
  		Lam03TermsEtaSortNew.insert(Lam03TermsEtaSortNew.begin(), Lam03TermsEtaSortNew[0]);
  		Lam03TermsEtaSortNew.insert(Lam03TermsEtaSortNew.end(), Lam03TermsEtaSortNew[nObs-diffZeroCount]);
	
		// interpolation + integration for Lambda03Eta
   	 	gsl_interp_accel *acc03Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda03Eta = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
		gsl_interp_init(interpLambda03Eta, yDiffSortNew.begin(), Lam03TermsEtaSortNew.begin(), yDiffSortNew.size());


		for(gg3 = 0; gg3 < nObs; gg3++)
			{
				if(yDiff[gg3]==0) {
    				Lambda03Eta(gg3,kk)= 0;
   		 		} else {
					gsl_interp_eval_integ_e(interpLambda03Eta,yDiffSortNew.begin(),Lam03TermsEtaSortNew.begin(),0,yDiff[gg3],acc03Eta, &result03Eta);
					Lambda03Eta(gg3,kk)=result03Eta;
				}	
			}

		gsl_interp_free(interpLambda03Eta);
		gsl_interp_accel_free(acc03Eta); 		
	}	

	
	// Put into GSL form
	RcppGSL::matrix<double> G_Lambda01Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda01Eta);
 	RcppGSL::matrix<double> G_Lambda02Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda02Eta);
 	RcppGSL::matrix<double> G_Lambda03Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda03Eta);
    
    /**************************************************
    // Computing score equations
    **************************************************/
	
	// compute A.vec and common.vec for each person
	Rcpp::NumericVector aVec(nObs);
	Rcpp::NumericVector commonVec(nObs);
	for(aa = 0; aa < nObs; aa++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, aa);
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, aa);
			RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, aa);
			double lp1 = 0;
			double lp2 = 0;
			double lp3 = 0;
			gsl_blas_ddot(x1Ind, G_beta1, &lp1);
			gsl_blas_ddot(x2Ind, G_beta2, &lp2);
			gsl_blas_ddot(x3Ind, G_beta3, &lp3);
			
			aVec[aa] = Lambda01[aa]*exp(lp1) + Lambda02[aa]*exp(lp2) + Lambda03[aa]*exp(lp3);
			commonVec[aa] = (exp(-h)+delta1[aa]+delta2[aa])/(1+exp(h)*aVec[aa]);
		} 
		
	/*double res_old=0;
	double res_new, term;
	for(i1 = 0; i1 < nObs; i1++)
		{
			 term = u1_ind(i1,y1, y2, delta1, delta2, wts, aVec, commonVec, h);
			 res_new = res_old + term;
			 res_old = res_new;
		}
	Rprintf("u1a, = %f \n", res_old);*/
	
	/*a = x2start[0];
	b = x2start[1];
	
	Rprintf("a = %f \n", a);
	Rprintf("b = %f \n", b);
	
	double u1Out = u1Ex(fakeXvals, a, b, nObs);
	Rprintf("u1 = %f \n", u1Out);		   
	
	double u2Out = u2Ex(fakeXvals, a, b, nObs);
	Rprintf("u2 = %f \n", u2Out);

  	
	// final score vector
	Rcpp::NumericVector scoreVec(2);
	scoreVec[0] = u1Out;
	scoreVec[1] = u2Out; */
	
	double u1Out = u1(y1, y2, delta1, delta2, wts, aVec, commonVec, h, nObs);
			   
	//Rprintf("u1a, = %f \n", u1Out);		   
	
	std::vector<double>  u2Out(nCov1);
	u2Out = u2(y1, y2, delta1, delta2, G_x1, G_beta1, wts, Lambda01, commonVec, h, nCov1, nObs);
	
	std::vector<double>  u3Out(nCov2);
	u3Out = u3(y1, y2, delta1, delta2, G_x2, G_beta2, wts, Lambda02, commonVec, h, nCov2, nObs);
	
	std::vector<double>  u4Out(nCov3);
	u4Out = u4(y1, y2, delta1, delta2, G_x3, G_beta3, wts, Lambda03, commonVec, h, nCov3, nObs);
	
	std::vector<double>  u5Out(nKnots1);
	if(penalty==FALSE)
		{
			u5Out = u5_nopenal(y1, y2, delta1, delta2, G_x1, G_beta1, G_eta1, G_Lambda01Eta, G_m1pred, wts, commonVec, h, nKnots1, nObs);
		}else{
			u5Out = u5_penal(y1, y2, delta1, delta2, G_x1, G_beta1, G_eta1, G_Lambda01Eta, G_m1pred, wts, penalty, kappa1, G_penaltyMat1, commonVec, h, nKnots1, nObs);
		}		
	
	std::vector<double>  u6Out(nKnots2);
	if(penalty==FALSE)
		{
			u6Out = u6_nopenal(y1, y2, delta1, delta2, G_x2, G_beta2, G_eta2, G_Lambda02Eta, G_m2pred, wts, commonVec, h, nKnots2, nObs);
		}else{
			u6Out = u6_penal(y1, y2, delta1, delta2, G_x2, G_beta2, G_eta2, G_Lambda02Eta, G_m2pred, wts, penalty, kappa2, G_penaltyMat2, commonVec, h, nKnots2, nObs);
		}	
	
	std::vector<double>  u7Out(nKnots3);
	if(penalty==FALSE)
		{
			u7Out = u7_nopenal(y1, y2, yDiff, delta1, delta2, G_x3, G_beta3, G_eta3, G_Lambda03Eta, G_m3pred, wts, commonVec, h, nKnots3, nObs);
		}else{
			u7Out = u7_penal(y1, y2, yDiff, delta1, delta2, G_x3, G_beta3, G_eta3, G_Lambda03Eta, G_m3pred, wts, penalty, kappa3, G_penaltyMat3, commonVec, h, nKnots3, nObs);
		} 
	
	// final score vector
	Rcpp::NumericVector scoreVec(1+nCov1+nCov2+nCov3+nKnots1+nKnots2+nKnots3);
	scoreVec[0] = u1Out;
	for(ii2 = 0; ii2 < nCov1; ii2++){
  		scoreVec[1 + ii2] = u2Out[ii2];
  	}
  	for(ii2 = 0; ii2 < nCov2; ii2++){
  		scoreVec[1+ nCov1 + ii2] = u3Out[ii2];
  	}
  	for(ii2 = 0; ii2 < nCov3; ii2++){
  		scoreVec[1+ nCov1 + nCov2 + ii2] = u4Out[ii2];
  	}
	for(jj2 = 0; jj2 < nKnots1; jj2++){
  		scoreVec[1+ nCov1 + nCov2 + nCov3 + jj2] = u5Out[jj2];
  	} 
  	for(jj2 = 0; jj2 < nKnots2; jj2++){
  		scoreVec[1+ nCov1 + nCov2 + nCov3 + nKnots1 + jj2] = u6Out[jj2];
  	} 
  	for(jj2 = 0; jj2 < nKnots3; jj2++){
  		scoreVec[1+ nCov1 + nCov2 + nCov3 + nKnots1 + nKnots2 + jj2] = u7Out[jj2];
  	} 
	
	return scoreVec;

}


// [[Rcpp::export]] 
Rcpp::NumericMatrix sandwichEstCheese(Rcpp::NumericVector xvec, const Rcpp::NumericVector y1, const Rcpp::NumericVector y2, 
					Rcpp::NumericVector delta1, Rcpp::NumericVector delta2, Rcpp::NumericMatrix x1, Rcpp::NumericMatrix x2, 
					Rcpp::NumericMatrix x3, Rcpp::NumericVector wts, Rcpp::NumericMatrix m1pred, 
					Rcpp::NumericMatrix m2pred, Rcpp::NumericMatrix m3pred)
{
	int aa, gg1, gg2, gg3, i, i1, i2, ii, ii2, j1, j2, jj, jj2, kk, obs;
	
	int nObs, nCov1, nCov2, nCov3, nKnots1, nKnots2, nKnots3;
	
	nObs = x1.nrow(); 
	nCov1 = x1.ncol(); 
	nCov2 = x2.ncol(); 
	nCov3 = x3.ncol();
	nKnots1 = m1pred.ncol(); 
	nKnots2 = m2pred.ncol(); 
	nKnots3 = m3pred.ncol();
	
	double h(1);//, a(1), b(1);
	Rcpp::NumericVector beta1(nCov1), beta2(nCov2), beta3(nCov3), eta1(nKnots1), eta2(nKnots2), eta3(nKnots3);
	 
	// initialize all parameters
	h = xvec[0];
	for(ii = 0; ii < nCov1; ii++){
  		beta1[ii] = xvec[1+ ii];
  	}
  	for(ii = 0; ii < nCov2; ii++){
  		beta2[ii] = xvec[1+nCov1+ ii];
  	}
  	for(ii = 0; ii < nCov3; ii++){
  		beta3[ii] = xvec[1+nCov1+nCov2+ ii];
  	}
	for(jj = 0; jj < nKnots1; jj++){
  		eta1[jj] = xvec[1+nCov1+nCov2+nCov3+ jj];
  	}
  	for(jj = 0; jj < nKnots2; jj++){
  		eta2[jj] = xvec[1+nCov1+nCov2+nCov3+nKnots1+ jj];
  	}
  	for(jj = 0; jj < nKnots3; jj++){
  		eta3[jj] = xvec[1+nCov1+nCov2+nCov3+nKnots1+nKnots2+ jj];
  	}


 	// make GSL versions as necessary
 	RcppGSL::vector<double> G_beta1 = Rcpp::as< RcppGSL::vector<double> >(beta1);
 	RcppGSL::vector<double> G_beta2 = Rcpp::as< RcppGSL::vector<double> >(beta2);
 	RcppGSL::vector<double> G_beta3 = Rcpp::as< RcppGSL::vector<double> >(beta3);
 	
 	RcppGSL::vector<double> G_eta1 = Rcpp::as< RcppGSL::vector<double> >(eta1);
 	RcppGSL::vector<double> G_eta2 = Rcpp::as< RcppGSL::vector<double> >(eta2);
 	RcppGSL::vector<double> G_eta3 = Rcpp::as< RcppGSL::vector<double> >(eta3);
 	
 	RcppGSL::matrix<double> G_m1pred = Rcpp::as< RcppGSL::matrix<double> >(m1pred);
 	RcppGSL::matrix<double> G_m2pred = Rcpp::as< RcppGSL::matrix<double> >(m2pred);
 	RcppGSL::matrix<double> G_m3pred = Rcpp::as< RcppGSL::matrix<double> >(m3pred);
 	
 	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);
 	RcppGSL::matrix<double> G_x2 = Rcpp::as< RcppGSL::matrix<double> >(x2);
 	RcppGSL::matrix<double> G_x3 = Rcpp::as< RcppGSL::matrix<double> >(x3);

    /**************************************************
    // Computing cumulative baseline hazard functions
    **************************************************/
 
 	// initialize vectors for BH and cumulative BH
 	Rcpp::NumericVector Lam01Terms(nObs), Lam02Terms(nObs), Lam03Terms(nObs);
	Rcpp::NumericVector Lambda01(nObs), Lambda02(nObs), Lambda03(nObs);
	
	Rcpp::NumericVector yDiff = y2-y1;
	Rcpp::NumericVector isDiffNonzero(nObs);
	int diffZeroCount = 0;
	
	// compute BH for each person
	for(i = 0; i < nObs; i++)
  	{
    	Lam01Terms[i] = lam0fn(i,G_m1pred, G_eta1);
    	Lam02Terms[i] = lam0fn(i,G_m2pred, G_eta2);
    	if(yDiff[i]==0) {
    		Lam03Terms[i] = 1;
    		isDiffNonzero[i] = 0;
    		diffZeroCount++;
    	} else {
    		Lam03Terms[i] = lam0fn(i,G_m3pred, G_eta3);
    		isDiffNonzero[i] = 1;
    	}		
  	}
  	
  	Rcpp::NumericVector y1Sort = RcppSort(y1);
  	Rcpp::NumericVector yDiffSort = RcppSort(yDiff);
  	Rcpp::NumericVector Lam01TermsSort = sortByVec(Lam01Terms, y1);
  	Rcpp::NumericVector Lam02TermsSort = sortByVec(Lam02Terms, y1);
  	Rcpp::NumericVector Lam03TermsSort = sortByVec(Lam03Terms, yDiff);
  	Rcpp::NumericVector yDiffSortNew(nObs-diffZeroCount), Lam03TermsSortNew(nObs-diffZeroCount);
  	  	
  	for(i = diffZeroCount; i < nObs; i++)
  	{
    	yDiffSortNew[i-diffZeroCount] = yDiffSort[i];
    	Lam03TermsSortNew[i-diffZeroCount] = Lam03TermsSort[i];	
  	}

  	// insert values for 0, and maximum values
  	y1Sort.insert(y1Sort.begin(), 0);
  	y1Sort.insert(y1Sort.end(), y1Sort[nObs]+0.1);
  	yDiffSortNew.insert(yDiffSortNew.begin(), 0);
  	yDiffSortNew.insert(yDiffSortNew.end(), yDiffSortNew[nObs-diffZeroCount]+0.1);
  	Lam01TermsSort.insert(Lam01TermsSort.begin(), Lam01TermsSort[0]);
  	Lam02TermsSort.insert(Lam02TermsSort.begin(), Lam02TermsSort[0]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.begin(), Lam03TermsSortNew[0]);
  	Lam01TermsSort.insert(Lam01TermsSort.end(), Lam01TermsSort[nObs]);
  	Lam02TermsSort.insert(Lam02TermsSort.end(), Lam02TermsSort[nObs]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.end(), Lam03TermsSortNew[nObs-diffZeroCount]);	
	
    // interpolation for Lambda01 
    gsl_interp_accel *acc01 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda01 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda01, y1Sort.begin(), Lam01TermsSort.begin(), y1Sort.size());
	
	double result01;
	for(gg1 = 0; gg1 < nObs; gg1++)
		{
			gsl_interp_eval_integ_e(interpLambda01,y1Sort.begin(),Lam01TermsSort.begin(),0,y1[gg1],acc01, &result01);
			Lambda01[gg1]=result01;
		}

	gsl_interp_free(interpLambda01);
	gsl_interp_accel_free(acc01); 
	
	// interpolation for Lambda02
    gsl_interp_accel *acc02 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda02 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda02, y1Sort.begin(), Lam02TermsSort.begin(), y1Sort.size());

	double result02;
	for(gg2 = 0; gg2 < nObs; gg2++)
		{
			gsl_interp_eval_integ_e(interpLambda02,y1Sort.begin(),Lam02TermsSort.begin(),0,y1[gg2],acc02, &result02);
			Lambda02[gg2]=result02;
		}

	gsl_interp_free(interpLambda02);
	gsl_interp_accel_free(acc02);
	
	// interpolation for Lambda03 
   	gsl_interp_accel *acc03 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda03 = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
	gsl_interp_init(interpLambda03, yDiffSortNew.begin(), Lam03TermsSortNew.begin(), yDiffSortNew.size());

	double result03;
	for(gg3 = 0; gg3 < nObs; gg3++)
		{
			if(yDiff[gg3]==0) {
    			Lambda03[gg3] = 0;
    		} else {
				gsl_interp_eval_integ_e(interpLambda03,yDiffSortNew.begin(),Lam03TermsSortNew.begin(),0,yDiff[gg3],acc03, &result03);
				Lambda03[gg3]=result03; 
			}	
		}

	gsl_interp_free(interpLambda03);
	gsl_interp_accel_free(acc03);
	
	
	/******************************************************************************************
    // Computing integrals that get used in the eta score equations (int lambda0j(s)*Mjl(s))
    ******************************************************************************************/
    
    // initialize vectors for terms (will get reused) and matrices for integrals themselves
 	Rcpp::NumericVector Lam01TermsEta(nObs), Lam02TermsEta(nObs), Lam03TermsEta(nObs);
 	Rcpp::NumericVector Lam01TermsEtaSort(nObs), Lam02TermsEtaSort(nObs), Lam03TermsEtaSort(nObs), Lam03TermsEtaSortNew(nObs-diffZeroCount);
	Rcpp::NumericMatrix Lambda01Eta(nObs,nKnots1), Lambda02Eta(nObs,nKnots2), Lambda03Eta(nObs,nKnots3);
	double result01Eta, result02Eta, result03Eta;
	
	// we still have yDiff, isDiffNonzero, and diffZeroCount from before
	// as well as y1Sort, yDiffSort, yDiffSortNew
	
	// all integrals for Lambda01
	for(kk = 0; kk < nKnots1; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		Lam01TermsEta[i] = lam0fnEta(i, kk, G_m1pred, G_eta1);	
  		}
  		
  		// fill sort vectors
  		Lam01TermsEtaSort = sortByVec(Lam01TermsEta, y1);
  		
  		// insert values for 0, and maximum values
  		Lam01TermsEtaSort.insert(Lam01TermsEtaSort.begin(), Lam01TermsEtaSort[0]);
  		Lam01TermsEtaSort.insert(Lam01TermsEtaSort.end(), Lam01TermsEtaSort[nObs]);

		// interpolation + integration for Lambda01Eta 
    	gsl_interp_accel *acc01Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda01Eta = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
		gsl_interp_init(interpLambda01Eta, y1Sort.begin(), Lam01TermsEtaSort.begin(), y1Sort.size());
	
		for(gg1 = 0; gg1 < nObs; gg1++)
			{
				gsl_interp_eval_integ_e(interpLambda01Eta,y1Sort.begin(),Lam01TermsEtaSort.begin(),0,y1[gg1],acc01Eta, &result01Eta);
				Lambda01Eta(gg1,kk)=result01Eta;
			}

		gsl_interp_free(interpLambda01Eta);
		gsl_interp_accel_free(acc01Eta); 	
	}	
	
	
	// all integrals for Lambda02
	for(kk = 0; kk < nKnots2; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		Lam02TermsEta[i] = lam0fnEta(i, kk, G_m2pred, G_eta2);	
  		}
  		
  		// fill sort vectors
  		Lam02TermsEtaSort = sortByVec(Lam02TermsEta, y1);
  		
  		// insert values for 0, and maximum values
  		Lam02TermsEtaSort.insert(Lam02TermsEtaSort.begin(), Lam02TermsEtaSort[0]);
  		Lam02TermsEtaSort.insert(Lam02TermsEtaSort.end(), Lam02TermsEtaSort[nObs]);
				
		// interpolation + integration for Lambda02Eta
    	gsl_interp_accel *acc02Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda02Eta = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
		gsl_interp_init(interpLambda02Eta, y1Sort.begin(), Lam02TermsEtaSort.begin(), y1Sort.size());

		for(gg2 = 0; gg2 < nObs; gg2++)
			{
				gsl_interp_eval_integ_e(interpLambda02Eta,y1Sort.begin(),Lam02TermsEtaSort.begin(),0,y1[gg2],acc02Eta, &result02Eta);
				Lambda02Eta(gg2,kk)=result02Eta;
			}

		gsl_interp_free(interpLambda02Eta);
		gsl_interp_accel_free(acc02Eta);
	}
	
	// all integrals for Lambda03
	for(kk = 0; kk < nKnots3; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		if(yDiff[i]==0) {
    			Lam03TermsEta[i] = 1;
    		} else {
    			Lam03TermsEta[i] = lam0fnEta(i, kk, G_m3pred, G_eta3);
    		}		
  		}
  		
  		// fill sort vectors
  		Lam03TermsEtaSort = sortByVec(Lam03TermsEta, yDiff);	
  		for(i = diffZeroCount; i < nObs; i++)
  		{
    		Lam03TermsEtaSortNew[i-diffZeroCount] = Lam03TermsEtaSort[i];	
  		}
  		
  		// insert values for 0, and maximum values
  		Lam03TermsEtaSortNew.insert(Lam03TermsEtaSortNew.begin(), Lam03TermsEtaSortNew[0]);
  		Lam03TermsEtaSortNew.insert(Lam03TermsEtaSortNew.end(), Lam03TermsEtaSortNew[nObs-diffZeroCount]);
	
		// interpolation + integration for Lambda03Eta
   	 	gsl_interp_accel *acc03Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda03Eta = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
		gsl_interp_init(interpLambda03Eta, yDiffSortNew.begin(), Lam03TermsEtaSortNew.begin(), yDiffSortNew.size());


		for(gg3 = 0; gg3 < nObs; gg3++)
			{
				if(yDiff[gg3]==0) {
    				Lambda03Eta(gg3,kk)= 0;
   		 		} else {
					gsl_interp_eval_integ_e(interpLambda03Eta,yDiffSortNew.begin(),Lam03TermsEtaSortNew.begin(),0,yDiff[gg3],acc03Eta, &result03Eta);
					Lambda03Eta(gg3,kk)=result03Eta;
				}	
			}

		gsl_interp_free(interpLambda03Eta);
		gsl_interp_accel_free(acc03Eta); 		
	}	

	
	// Put into GSL form
	RcppGSL::matrix<double> G_Lambda01Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda01Eta);
 	RcppGSL::matrix<double> G_Lambda02Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda02Eta);
 	RcppGSL::matrix<double> G_Lambda03Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda03Eta);
    
    /**************************************************
    // Computing score equations
    **************************************************/
	
	// compute A.vec and common.vec for each person
	Rcpp::NumericVector aVec(nObs);
	Rcpp::NumericVector commonVec(nObs);
	for(aa = 0; aa < nObs; aa++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, aa);
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, aa);
			RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, aa);
			double lp1 = 0;
			double lp2 = 0;
			double lp3 = 0;
			gsl_blas_ddot(x1Ind, G_beta1, &lp1);
			gsl_blas_ddot(x2Ind, G_beta2, &lp2);
			gsl_blas_ddot(x3Ind, G_beta3, &lp3);
			
			aVec[aa] = Lambda01[aa]*exp(lp1) + Lambda02[aa]*exp(lp2) + Lambda03[aa]*exp(lp3);
			commonVec[aa] = (exp(-h)+delta1[aa]+delta2[aa])/(1+exp(h)*aVec[aa]);
		} 
		
	/*double res_old=0;
	double res_new, term;
	for(i1 = 0; i1 < nObs; i1++)
		{
			 term = u1_ind(i1,y1, y2, delta1, delta2, wts, aVec, commonVec, h);
			 res_new = res_old + term;
			 res_old = res_new;
		}
	Rprintf("u1a, = %f \n", res_old);*/
	
	/*a = x2start[0];
	b = x2start[1];
	
	Rprintf("a = %f \n", a);
	Rprintf("b = %f \n", b);
	
	double u1Out = u1Ex(fakeXvals, a, b, nObs);
	Rprintf("u1 = %f \n", u1Out);		   
	
	double u2Out = u2Ex(fakeXvals, a, b, nObs);
	Rprintf("u2 = %f \n", u2Out);

  	
	// final score vector
	Rcpp::NumericVector scoreVec(2);
	scoreVec[0] = u1Out;
	scoreVec[1] = u2Out; */
	
	// initialize final matrix
	int nScore = 1+ nCov1 + nCov2 + nCov3 + nKnots1 + nKnots2 + nKnots3;
	Rcpp::NumericMatrix IHess(nScore,nScore);
	Rcpp::NumericMatrix IHessOld(nScore,nScore);
	
	// compute score contributions for each individual
	for(obs = 0; obs < nObs; obs++)
		{
			double u1IndOut = u1_ind(obs, y1, y2, delta1, delta2, wts, aVec, commonVec, h);
			
			std::vector<double>  u2IndOut(nCov1);
			u2IndOut = u2_ind(obs, y1, y2, delta1, delta2, G_x1, G_beta1, wts, Lambda01, commonVec, h, nCov1);
			
			std::vector<double>  u3IndOut(nCov2);
			u3IndOut = u3_ind(obs, y1, y2, delta1, delta2, G_x2, G_beta2, wts, Lambda02, commonVec, h, nCov2);
			
			std::vector<double>  u4IndOut(nCov3);
			u4IndOut = u4_ind(obs, y1, y2, delta1, delta2, G_x3, G_beta3, wts, Lambda03, commonVec, h, nCov3);
			
			std::vector<double>  u5IndOut(nKnots1);
			u5IndOut = u5_ind_nopenal(obs, y1, y2, delta1, delta2, G_x1, G_beta1, G_eta1, G_Lambda01Eta, G_m1pred, wts, commonVec, h, nKnots1);
		
			std::vector<double>  u6IndOut(nKnots2);
			u6IndOut = u6_ind_nopenal(obs, y1, y2, delta1, delta2, G_x2, G_beta2, G_eta2, G_Lambda02Eta, G_m2pred, wts, commonVec, h, nKnots2);
		
			std::vector<double>  u7IndOut(nKnots3);
			u7IndOut = u7_ind_nopenal(obs, y1, y2, yDiff, delta1, delta2, G_x3, G_beta3, G_eta3, G_Lambda03Eta, G_m3pred, wts, commonVec, h, nKnots3);
		
			Rcpp::NumericVector scoreVecInd(nScore);
			scoreVecInd[0] = u1IndOut;
			for(ii2 = 0; ii2 < nCov1; ii2++){
  				scoreVecInd[1 + ii2] = u2IndOut[ii2];
  			}
  			for(ii2 = 0; ii2 < nCov2; ii2++){
  				scoreVecInd[1+ nCov1 + ii2] = u3IndOut[ii2];
  			}
  			for(ii2 = 0; ii2 < nCov3; ii2++){
  				scoreVecInd[1+ nCov1 + nCov2 + ii2] = u4IndOut[ii2];
  			}
			for(jj2 = 0; jj2 < nKnots1; jj2++){
  				scoreVecInd[1+ nCov1 + nCov2 + nCov3 + jj2] = u5IndOut[jj2];
  			} 
  			for(jj2 = 0; jj2 < nKnots2; jj2++){
  				scoreVecInd[1+ nCov1 + nCov2 + nCov3 + nKnots1 + jj2] = u6IndOut[jj2];
  			} 
  			for(jj2 = 0; jj2 < nKnots3; jj2++){
  				scoreVecInd[1+ nCov1 + nCov2 + nCov3 + nKnots1 + nKnots2 + jj2] = u7IndOut[jj2];
  			} 
  			
  			// slot individual contributions into final matrix
  			for(i1 = 0; i1 < nScore; i1++){
  				for(j1 = i1; j1<nScore; j1++){
  					IHess(i1,j1) = IHessOld(i1,j1)+scoreVecInd(i1)*scoreVecInd(j1);
  					IHessOld(i1,j1) = IHess(i1,j1);
  				}
  			}
		}
		
		// fill in lower triangle of final matrix
		for(i2 = 1; i2 < nScore; i2++){
  			for(j2 = 0; j2 < i2; j2++){
  				IHess(i2,j2) = IHess(j2,i2);
  			}
  		}

	return IHess;

}



// [[Rcpp::export]] 
double logLikNoPenal(Rcpp::NumericVector xvec, const Rcpp::NumericVector y1, const Rcpp::NumericVector y2, 
					Rcpp::NumericVector delta1, Rcpp::NumericVector delta2, Rcpp::NumericMatrix x1, Rcpp::NumericMatrix x2, 
					Rcpp::NumericMatrix x3, Rcpp::NumericVector wts, Rcpp::NumericMatrix m1pred, 
					Rcpp::NumericMatrix m2pred, Rcpp::NumericMatrix m3pred)
{
	int aa, gg1, gg2, gg3, i, ii, jj;
	
	int nObs, nCov1, nCov2, nCov3, nKnots1, nKnots2, nKnots3;
	
	nObs = x1.nrow(); 
	nCov1 = x1.ncol(); 
	nCov2 = x2.ncol(); 
	nCov3 = x3.ncol();
	nKnots1 = m1pred.ncol(); 
	nKnots2 = m2pred.ncol(); 
	nKnots3 = m3pred.ncol();
	
	double h(1);//, a(1), b(1);
	Rcpp::NumericVector beta1(nCov1), beta2(nCov2), beta3(nCov3), eta1(nKnots1), eta2(nKnots2), eta3(nKnots3);
	 
	// initialize all parameters
	h = xvec[0];
	for(ii = 0; ii < nCov1; ii++){
  		beta1[ii] = xvec[1+ ii];
  	}
  	for(ii = 0; ii < nCov2; ii++){
  		beta2[ii] = xvec[1+nCov1+ ii];
  	}
  	for(ii = 0; ii < nCov3; ii++){
  		beta3[ii] = xvec[1+nCov1+nCov2+ ii];
  	}
	for(jj = 0; jj < nKnots1; jj++){
  		eta1[jj] = xvec[1+nCov1+nCov2+nCov3+ jj];
  	}
  	for(jj = 0; jj < nKnots2; jj++){
  		eta2[jj] = xvec[1+nCov1+nCov2+nCov3+nKnots1+ jj];
  	}
  	for(jj = 0; jj < nKnots3; jj++){
  		eta3[jj] = xvec[1+nCov1+nCov2+nCov3+nKnots1+nKnots2+ jj];
  	}


 	// make GSL versions as necessary
 	RcppGSL::vector<double> G_beta1 = Rcpp::as< RcppGSL::vector<double> >(beta1);
 	RcppGSL::vector<double> G_beta2 = Rcpp::as< RcppGSL::vector<double> >(beta2);
 	RcppGSL::vector<double> G_beta3 = Rcpp::as< RcppGSL::vector<double> >(beta3);
 	
 	RcppGSL::vector<double> G_eta1 = Rcpp::as< RcppGSL::vector<double> >(eta1);
 	RcppGSL::vector<double> G_eta2 = Rcpp::as< RcppGSL::vector<double> >(eta2);
 	RcppGSL::vector<double> G_eta3 = Rcpp::as< RcppGSL::vector<double> >(eta3);
 	
 	RcppGSL::matrix<double> G_m1pred = Rcpp::as< RcppGSL::matrix<double> >(m1pred);
 	RcppGSL::matrix<double> G_m2pred = Rcpp::as< RcppGSL::matrix<double> >(m2pred);
 	RcppGSL::matrix<double> G_m3pred = Rcpp::as< RcppGSL::matrix<double> >(m3pred);
 	
 	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);
 	RcppGSL::matrix<double> G_x2 = Rcpp::as< RcppGSL::matrix<double> >(x2);
 	RcppGSL::matrix<double> G_x3 = Rcpp::as< RcppGSL::matrix<double> >(x3);
 	
    /**************************************************
    // Computing cumulative baseline hazard functions
    **************************************************/
 
 	// initialize vectors for log BH, BH and cumulative BH
 	Rcpp::NumericVector LogLam01Terms(nObs), LogLam02Terms(nObs), LogLam03Terms(nObs);
 	Rcpp::NumericVector Lam01Terms(nObs), Lam02Terms(nObs), Lam03Terms(nObs);
	Rcpp::NumericVector Lambda01(nObs), Lambda02(nObs), Lambda03(nObs);
	
	Rcpp::NumericVector yDiff = y2-y1;
	Rcpp::NumericVector isDiffNonzero(nObs);
	int diffZeroCount = 0;
	
	// compute BH for each person
	for(i = 0; i < nObs; i++)
  	{
  		LogLam01Terms[i] = loglam0fn(i,G_m1pred, G_eta1);
    	LogLam02Terms[i] = loglam0fn(i,G_m2pred, G_eta2);
    	Lam01Terms[i] = lam0fn(i,G_m1pred, G_eta1);
    	Lam02Terms[i] = lam0fn(i,G_m2pred, G_eta2);
    	if(yDiff[i]==0) {
    		LogLam03Terms[i] = 0;
    		Lam03Terms[i] = 1;
    		isDiffNonzero[i] = 0;
    		diffZeroCount++;
    	} else {
    		LogLam03Terms[i] = loglam0fn(i,G_m3pred, G_eta3);
    		Lam03Terms[i] = lam0fn(i,G_m3pred, G_eta3);
    		isDiffNonzero[i] = 1;
    	}		
  	}
  	
  	Rcpp::NumericVector y1Sort = RcppSort(y1);
  	Rcpp::NumericVector yDiffSort = RcppSort(yDiff);
  	Rcpp::NumericVector Lam01TermsSort = sortByVec(Lam01Terms, y1);
  	Rcpp::NumericVector Lam02TermsSort = sortByVec(Lam02Terms, y1);
  	Rcpp::NumericVector Lam03TermsSort = sortByVec(Lam03Terms, yDiff);
  	Rcpp::NumericVector yDiffSortNew(nObs-diffZeroCount), Lam03TermsSortNew(nObs-diffZeroCount);
  	  	
  	for(i = diffZeroCount; i < nObs; i++)
  	{
    	yDiffSortNew[i-diffZeroCount] = yDiffSort[i];
    	Lam03TermsSortNew[i-diffZeroCount] = Lam03TermsSort[i];	
  	}

  	// insert values for 0, and maximum values
  	y1Sort.insert(y1Sort.begin(), 0);
  	y1Sort.insert(y1Sort.end(), y1Sort[nObs]+0.1);
  	yDiffSortNew.insert(yDiffSortNew.begin(), 0);
  	yDiffSortNew.insert(yDiffSortNew.end(), yDiffSortNew[nObs-diffZeroCount]+0.1);
  	Lam01TermsSort.insert(Lam01TermsSort.begin(), Lam01TermsSort[0]);
  	Lam02TermsSort.insert(Lam02TermsSort.begin(), Lam02TermsSort[0]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.begin(), Lam03TermsSortNew[0]);
  	Lam01TermsSort.insert(Lam01TermsSort.end(), Lam01TermsSort[nObs]);
  	Lam02TermsSort.insert(Lam02TermsSort.end(), Lam02TermsSort[nObs]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.end(), Lam03TermsSortNew[nObs-diffZeroCount]);	
	
    // interpolation for Lambda01 
    gsl_interp_accel *acc01 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda01 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda01, y1Sort.begin(), Lam01TermsSort.begin(), y1Sort.size());
	
	double result01;
	for(gg1 = 0; gg1 < nObs; gg1++)
		{
			gsl_interp_eval_integ_e(interpLambda01,y1Sort.begin(),Lam01TermsSort.begin(),0,y1[gg1],acc01, &result01);
			Lambda01[gg1]=result01;
		}

	gsl_interp_free(interpLambda01);
	gsl_interp_accel_free(acc01); 
	
	// interpolation for Lambda02
    gsl_interp_accel *acc02 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda02 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda02, y1Sort.begin(), Lam02TermsSort.begin(), y1Sort.size());

	double result02;
	for(gg2 = 0; gg2 < nObs; gg2++)
		{
			gsl_interp_eval_integ_e(interpLambda02,y1Sort.begin(),Lam02TermsSort.begin(),0,y1[gg2],acc02, &result02);
			Lambda02[gg2]=result02;
		}

	gsl_interp_free(interpLambda02);
	gsl_interp_accel_free(acc02);
	
	// interpolation for Lambda03 
   	gsl_interp_accel *acc03 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda03 = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
	gsl_interp_init(interpLambda03, yDiffSortNew.begin(), Lam03TermsSortNew.begin(), yDiffSortNew.size());

	double result03;
	for(gg3 = 0; gg3 < nObs; gg3++)
		{
			if(yDiff[gg3]==0) {
    			Lambda03[gg3] = 0;
    		} else {
				gsl_interp_eval_integ_e(interpLambda03,yDiffSortNew.begin(),Lam03TermsSortNew.begin(),0,yDiff[gg3],acc03, &result03);
				Lambda03[gg3]=result03; 
			}	
		}

	gsl_interp_free(interpLambda03);
	gsl_interp_accel_free(acc03);
	
	/**************************************************
    // Computing score equations
    **************************************************/
	
	// compute A.vec and common.vec for each person
	Rcpp::NumericVector aVec(nObs);
	Rcpp::NumericVector commonVec(nObs);
	for(aa = 0; aa < nObs; aa++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, aa);
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, aa);
			RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, aa);
			double lp1 = 0;
			double lp2 = 0;
			double lp3 = 0;
			gsl_blas_ddot(x1Ind, G_beta1, &lp1);
			gsl_blas_ddot(x2Ind, G_beta2, &lp2);
			gsl_blas_ddot(x3Ind, G_beta3, &lp3);
			
			aVec[aa] = Lambda01[aa]*exp(lp1) + Lambda02[aa]*exp(lp2) + Lambda03[aa]*exp(lp3);
			commonVec[aa] = (exp(-h)+delta1[aa]+delta2[aa])/(1+exp(h)*aVec[aa]);
		} 
		
		
	// need to add in (lowercase) lambda for all of these!
	
	std::vector<double> logLikInds(nObs);
	double logLikVal;
	int ind;
	for(ind = 0; ind < nObs; ind++)
		{
			logLikInds[ind] = wts[ind]*(delta1[ind]*LogLam01Terms[ind]
						+(1-delta1[ind])*delta2[ind]*LogLam02Terms[ind]
						+delta1[ind]*delta2[ind]*log(1+exp(h))
						-(exp(-h)+delta1[ind]+delta2[ind])*log(1+exp(h)*aVec[ind]));
		}
	logLikVal =  std::accumulate(logLikInds.begin(), logLikInds.end(), 0.0);
	
	return logLikVal;

}


// [[Rcpp::export]] 
double logLikPenal(Rcpp::NumericVector xvec, const Rcpp::NumericVector y1, const Rcpp::NumericVector y2, 
					Rcpp::NumericVector delta1, Rcpp::NumericVector delta2, Rcpp::NumericMatrix x1, Rcpp::NumericMatrix x2, 
					Rcpp::NumericMatrix x3, Rcpp::NumericVector wts, Rcpp::NumericMatrix m1pred, 
					Rcpp::NumericMatrix m2pred, Rcpp::NumericMatrix m3pred, Rcpp::NumericMatrix m1deriv2pred, 
					Rcpp::NumericMatrix m2deriv2pred, Rcpp::NumericMatrix m3deriv2pred, double kappa1, double kappa2, double kappa3)
{
	int aa, gg1, gg2, gg3, i, ii, jj;
	
	int nObs, nCov1, nCov2, nCov3, nKnots1, nKnots2, nKnots3;
	
	nObs = x1.nrow(); 
	nCov1 = x1.ncol(); 
	nCov2 = x2.ncol(); 
	nCov3 = x3.ncol();
	nKnots1 = m1pred.ncol(); 
	nKnots2 = m2pred.ncol(); 
	nKnots3 = m3pred.ncol();
	
	double h(1);//, a(1), b(1);
	Rcpp::NumericVector beta1(nCov1), beta2(nCov2), beta3(nCov3), eta1(nKnots1), eta2(nKnots2), eta3(nKnots3);
	 
	// initialize all parameters
	h = xvec[0];
	for(ii = 0; ii < nCov1; ii++){
  		beta1[ii] = xvec[1+ ii];
  	}
  	for(ii = 0; ii < nCov2; ii++){
  		beta2[ii] = xvec[1+nCov1+ ii];
  	}
  	for(ii = 0; ii < nCov3; ii++){
  		beta3[ii] = xvec[1+nCov1+nCov2+ ii];
  	}
	for(jj = 0; jj < nKnots1; jj++){
  		eta1[jj] = xvec[1+nCov1+nCov2+nCov3+ jj];
  	}
  	for(jj = 0; jj < nKnots2; jj++){
  		eta2[jj] = xvec[1+nCov1+nCov2+nCov3+nKnots1+ jj];
  	}
  	for(jj = 0; jj < nKnots3; jj++){
  		eta3[jj] = xvec[1+nCov1+nCov2+nCov3+nKnots1+nKnots2+ jj];
  	}


 	// make GSL versions as necessary
 	RcppGSL::vector<double> G_beta1 = Rcpp::as< RcppGSL::vector<double> >(beta1);
 	RcppGSL::vector<double> G_beta2 = Rcpp::as< RcppGSL::vector<double> >(beta2);
 	RcppGSL::vector<double> G_beta3 = Rcpp::as< RcppGSL::vector<double> >(beta3);
 	
 	RcppGSL::vector<double> G_eta1 = Rcpp::as< RcppGSL::vector<double> >(eta1);
 	RcppGSL::vector<double> G_eta2 = Rcpp::as< RcppGSL::vector<double> >(eta2);
 	RcppGSL::vector<double> G_eta3 = Rcpp::as< RcppGSL::vector<double> >(eta3);
 	
 	RcppGSL::matrix<double> G_m1pred = Rcpp::as< RcppGSL::matrix<double> >(m1pred);
 	RcppGSL::matrix<double> G_m2pred = Rcpp::as< RcppGSL::matrix<double> >(m2pred);
 	RcppGSL::matrix<double> G_m3pred = Rcpp::as< RcppGSL::matrix<double> >(m3pred);
 	
 	RcppGSL::matrix<double> G_m1deriv2pred = Rcpp::as< RcppGSL::matrix<double> >(m1deriv2pred);
 	RcppGSL::matrix<double> G_m2deriv2pred = Rcpp::as< RcppGSL::matrix<double> >(m2deriv2pred);
 	RcppGSL::matrix<double> G_m3deriv2pred = Rcpp::as< RcppGSL::matrix<double> >(m3deriv2pred);
 	
 	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);
 	RcppGSL::matrix<double> G_x2 = Rcpp::as< RcppGSL::matrix<double> >(x2);
 	RcppGSL::matrix<double> G_x3 = Rcpp::as< RcppGSL::matrix<double> >(x3);
 	
    /**************************************************
    // Computing cumulative baseline hazard functions
    **************************************************/
 
 	// initialize vectors for log BH, BH and cumulative BH
 	Rcpp::NumericVector Lam01PenalTerms(nObs), Lam02PenalTerms(nObs), Lam03PenalTerms(nObs);
 	Rcpp::NumericVector LogLam01Terms(nObs), LogLam02Terms(nObs), LogLam03Terms(nObs);
 	Rcpp::NumericVector Lam01Terms(nObs), Lam02Terms(nObs), Lam03Terms(nObs);
	Rcpp::NumericVector Lambda01(nObs), Lambda02(nObs), Lambda03(nObs);
	
	Rcpp::NumericVector yDiff = y2-y1;
	Rcpp::NumericVector isDiffNonzero(nObs);
	int diffZeroCount = 0;
	
	// compute BH for each person
	for(i = 0; i < nObs; i++)
  	{
  		Lam01PenalTerms[i] = loglam0penalfn(i,G_m1deriv2pred, G_eta1);
  		Lam02PenalTerms[i] = loglam0penalfn(i,G_m2deriv2pred, G_eta2);
  		LogLam01Terms[i] = loglam0fn(i,G_m1pred, G_eta1);
    	LogLam02Terms[i] = loglam0fn(i,G_m2pred, G_eta2);
    	Lam01Terms[i] = lam0fn(i,G_m1pred, G_eta1);
    	Lam02Terms[i] = lam0fn(i,G_m2pred, G_eta2);
    	if(yDiff[i]==0) {
    		Lam03PenalTerms[i] = 0;
    		LogLam03Terms[i] = 0;
    		Lam03Terms[i] = 1;
    		isDiffNonzero[i] = 0;
    		diffZeroCount++;
    	} else {
    		Lam03PenalTerms[i] = loglam0penalfn(i,G_m3deriv2pred, G_eta3);
    		LogLam03Terms[i] = loglam0fn(i,G_m3pred, G_eta3);
    		Lam03Terms[i] = lam0fn(i,G_m3pred, G_eta3);
    		isDiffNonzero[i] = 1;
    	}		
  	}
  	
  	Rcpp::NumericVector y1Sort = RcppSort(y1);
  	Rcpp::NumericVector yDiffSort = RcppSort(yDiff);
  	Rcpp::NumericVector Lam01TermsSort = sortByVec(Lam01Terms, y1);
  	Rcpp::NumericVector Lam02TermsSort = sortByVec(Lam02Terms, y1);
  	Rcpp::NumericVector Lam03TermsSort = sortByVec(Lam03Terms, yDiff);
  	Rcpp::NumericVector Lam01PenalTermsSort = sortByVec(Lam01PenalTerms, y1);
  	Rcpp::NumericVector Lam02PenalTermsSort = sortByVec(Lam02PenalTerms, y1);
  	Rcpp::NumericVector Lam03PenalTermsSort = sortByVec(Lam03PenalTerms, yDiff);
  	Rcpp::NumericVector yDiffSortNew(nObs-diffZeroCount), Lam03TermsSortNew(nObs-diffZeroCount), Lam03PenalTermsSortNew(nObs-diffZeroCount);
  	  	
  	for(i = diffZeroCount; i < nObs; i++)
  	{
    	yDiffSortNew[i-diffZeroCount] = yDiffSort[i];
    	Lam03TermsSortNew[i-diffZeroCount] = Lam03TermsSort[i];
    	Lam03PenalTermsSortNew[i-diffZeroCount] = Lam03PenalTermsSort[i];	
  	}

  	// insert values for 0, and maximum values
  	y1Sort.insert(y1Sort.begin(), 0);
  	y1Sort.insert(y1Sort.end(), y1Sort[nObs]+0.1);
  	yDiffSortNew.insert(yDiffSortNew.begin(), 0);
  	yDiffSortNew.insert(yDiffSortNew.end(), yDiffSortNew[nObs-diffZeroCount]+0.1);
  	Lam01TermsSort.insert(Lam01TermsSort.begin(), Lam01TermsSort[0]);
  	Lam02TermsSort.insert(Lam02TermsSort.begin(), Lam02TermsSort[0]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.begin(), Lam03TermsSortNew[0]);
  	Lam01TermsSort.insert(Lam01TermsSort.end(), Lam01TermsSort[nObs]);
  	Lam02TermsSort.insert(Lam02TermsSort.end(), Lam02TermsSort[nObs]);
  	Lam03TermsSortNew.insert(Lam03TermsSortNew.end(), Lam03TermsSortNew[nObs-diffZeroCount]);	
  	
  	Lam01PenalTermsSort.insert(Lam01PenalTermsSort.begin(), Lam01PenalTermsSort[0]);
  	Lam02PenalTermsSort.insert(Lam02PenalTermsSort.begin(), Lam02PenalTermsSort[0]);
  	Lam03PenalTermsSortNew.insert(Lam03PenalTermsSortNew.begin(), Lam03PenalTermsSortNew[0]);
  	Lam01PenalTermsSort.insert(Lam01PenalTermsSort.end(), Lam01PenalTermsSort[nObs]);
  	Lam02PenalTermsSort.insert(Lam02PenalTermsSort.end(), Lam02PenalTermsSort[nObs]);
  	Lam03PenalTermsSortNew.insert(Lam03PenalTermsSortNew.end(), Lam03PenalTermsSortNew[nObs-diffZeroCount]);	
	
    // interpolation for Lambda01 
    gsl_interp_accel *acc01 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda01 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda01, y1Sort.begin(), Lam01TermsSort.begin(), y1Sort.size());
	
	double result01;
	for(gg1 = 0; gg1 < nObs; gg1++)
		{
			gsl_interp_eval_integ_e(interpLambda01,y1Sort.begin(),Lam01TermsSort.begin(),0,y1[gg1],acc01, &result01);
			Lambda01[gg1]=result01;
		}

	gsl_interp_free(interpLambda01);
	gsl_interp_accel_free(acc01); 
	
	// interpolation for Lambda02
    gsl_interp_accel *acc02 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda02 = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda02, y1Sort.begin(), Lam02TermsSort.begin(), y1Sort.size());

	double result02;
	for(gg2 = 0; gg2 < nObs; gg2++)
		{
			gsl_interp_eval_integ_e(interpLambda02,y1Sort.begin(),Lam02TermsSort.begin(),0,y1[gg2],acc02, &result02);
			Lambda02[gg2]=result02;
		}

	gsl_interp_free(interpLambda02);
	gsl_interp_accel_free(acc02);
	
	// interpolation for Lambda03 
   	gsl_interp_accel *acc03 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda03 = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
	gsl_interp_init(interpLambda03, yDiffSortNew.begin(), Lam03TermsSortNew.begin(), yDiffSortNew.size());

	double result03;
	for(gg3 = 0; gg3 < nObs; gg3++)
		{
			if(yDiff[gg3]==0) {
    			Lambda03[gg3] = 0;
    		} else {
				gsl_interp_eval_integ_e(interpLambda03,yDiffSortNew.begin(),Lam03TermsSortNew.begin(),0,yDiff[gg3],acc03, &result03);
				Lambda03[gg3]=result03; 
			}	
		}

	gsl_interp_free(interpLambda03);
	gsl_interp_accel_free(acc03);
	
	// interpolation for Lambda01 - penalty 
    gsl_interp_accel *acc01Penal = gsl_interp_accel_alloc();
	gsl_interp *interpLambda01Penal = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda01Penal, y1Sort.begin(), Lam01PenalTermsSort.begin(), y1Sort.size());
	
	double Lambda01Penal;
	gsl_interp_eval_integ_e(interpLambda01Penal,y1Sort.begin(),Lam01PenalTermsSort.begin(),0,*y1Sort.end(),acc01Penal, &Lambda01Penal);

	gsl_interp_free(interpLambda01Penal);
	gsl_interp_accel_free(acc01Penal);
	
	// interpolation for Lambda02 - penalty
    gsl_interp_accel *acc02Penal = gsl_interp_accel_alloc();
	gsl_interp *interpLambda02Penal = gsl_interp_alloc(gsl_interp_linear, y1Sort.size());
	gsl_interp_init(interpLambda02Penal, y1Sort.begin(), Lam02PenalTermsSort.begin(), y1Sort.size());

	double Lambda02Penal;
	gsl_interp_eval_integ_e(interpLambda02Penal,y1Sort.begin(),Lam02PenalTermsSort.begin(),0,*y1Sort.end(),acc02Penal, &Lambda02Penal);

	gsl_interp_free(interpLambda02Penal);
	gsl_interp_accel_free(acc02Penal);
	
	// interpolation for Lambda03 - penalty
   	gsl_interp_accel *acc03Penal = gsl_interp_accel_alloc();
	gsl_interp *interpLambda03Penal = gsl_interp_alloc(gsl_interp_linear, yDiffSortNew.size());
	gsl_interp_init(interpLambda03Penal, yDiffSortNew.begin(), Lam03PenalTermsSortNew.begin(), yDiffSortNew.size());

	double Lambda03Penal;
	gsl_interp_eval_integ_e(interpLambda03Penal,yDiffSortNew.begin(),Lam03PenalTermsSortNew.begin(),0,*yDiffSortNew.end(),acc03Penal, &Lambda03Penal);

	gsl_interp_free(interpLambda03Penal);
	gsl_interp_accel_free(acc03Penal);
	
	/**************************************************
    // Computing score equations
    **************************************************/
	
	// compute A.vec and common.vec for each person
	Rcpp::NumericVector aVec(nObs);
	Rcpp::NumericVector commonVec(nObs);
	for(aa = 0; aa < nObs; aa++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, aa);
			RcppGSL::VectorView x2Ind = gsl_matrix_row(G_x2, aa);
			RcppGSL::VectorView x3Ind = gsl_matrix_row(G_x3, aa);
			double lp1 = 0;
			double lp2 = 0;
			double lp3 = 0;
			gsl_blas_ddot(x1Ind, G_beta1, &lp1);
			gsl_blas_ddot(x2Ind, G_beta2, &lp2);
			gsl_blas_ddot(x3Ind, G_beta3, &lp3);
			
			aVec[aa] = Lambda01[aa]*exp(lp1) + Lambda02[aa]*exp(lp2) + Lambda03[aa]*exp(lp3);
			commonVec[aa] = (exp(-h)+delta1[aa]+delta2[aa])/(1+exp(h)*aVec[aa]);
		} 
		
		
	// need to add in (lowercase) lambda for all of these!
	
	std::vector<double> logLikInds(nObs);
	double logLikVal, logLikValPenal, penal;
	int ind;
	for(ind = 0; ind < nObs; ind++)
		{
			logLikInds[ind] = wts[ind]*(delta1[ind]*LogLam01Terms[ind]
						+(1-delta1[ind])*delta2[ind]*LogLam02Terms[ind]
						+delta1[ind]*delta2[ind]*log(1+exp(h))
						-(exp(-h)+delta1[ind]+delta2[ind])*log(1+exp(h)*aVec[ind]));
		}
	logLikVal = std::accumulate(logLikInds.begin(), logLikInds.end(), 0.0);
	penal = kappa1*Lambda01Penal + kappa2*Lambda02Penal + kappa3*Lambda03Penal;
	logLikValPenal = logLikVal - penal;
	
	return logLikValPenal;

}
	
// score equation uBetaUniv
std::vector<double> uBetaUniv(Rcpp::NumericVector y, Rcpp::NumericVector delta, RcppGSL::matrix<double> G_x1, 
					RcppGSL::vector<double> G_beta, Rcpp::NumericVector wts, Rcpp::NumericVector Lambda01, int nCov, int nObs)
{
	std::vector<double> scoreVals(nCov);
	std::vector<double> scoreValsOld(nCov);
	double cov, ind, betaTx, coef;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
			betaTx=0; 
			gsl_blas_ddot(G_beta, x1Ind, &betaTx);
			coef = wts[ind]*(delta[ind] - Lambda01[ind]*exp(betaTx));
			for(cov = 0; cov < nCov; cov++)
				{
					scoreVals[cov] = scoreValsOld[cov]+coef*x1Ind[cov];
					scoreValsOld[cov] = scoreVals[cov];
				}
		}		
	return scoreVals;	
}

// score equation uEtaUniv
std::vector<double> uEtaUniv(Rcpp::NumericVector y, Rcpp::NumericVector delta, RcppGSL::matrix<double> G_x1, 
					RcppGSL::vector<double> G_beta, RcppGSL::vector<double> G_eta, RcppGSL::matrix<double> G_Lambda0Eta, 
			   		RcppGSL::matrix<double> G_bpred, Rcpp::NumericVector wts, int nKnots, int nObs)
{
	std::vector<double> scoreVals(nKnots);
	std::vector<double> scoreValsOld(nKnots);
	double spl, ind, coef1, coef2;
	for(ind = 0; ind < nObs; ind++)
		{
			RcppGSL::VectorView x1Ind = gsl_matrix_row(G_x1, ind);
			RcppGSL::VectorView bInd = gsl_matrix_row(G_bpred, ind); 
			RcppGSL::VectorView Lambda0EtaInd = gsl_matrix_row(G_Lambda0Eta, ind); 
			double betaTx=0; 
			gsl_blas_ddot(G_beta, x1Ind, &betaTx);
			coef1 = delta[ind];
			coef2 = exp(betaTx);
			for(spl = 0; spl < nKnots; spl++)
				{
					scoreVals[spl]=scoreValsOld[spl]+wts[ind]*(coef1*bInd[spl] - coef2*Lambda0EtaInd[spl]);
					scoreValsOld[spl] = scoreVals[spl];
				}
		}
	
	return scoreVals;		
}
	

// [[Rcpp::export]] 
Rcpp::NumericVector spModelUniv(Rcpp::NumericVector xvec, int nObs, int nCov, int nKnots, const Rcpp::NumericVector y,
					Rcpp::NumericVector delta, Rcpp::NumericMatrix x1, Rcpp::NumericVector wts, Rcpp::NumericMatrix bpred)
{
	int gg, i, ii, ii2, jj, jj2, kk;
	//double a(1), b(1);
	Rcpp::NumericVector beta(nCov), eta(nKnots);
	 
	// initialize all parameters
	for(ii = 0; ii < nCov; ii++){
  		beta[ii] = xvec[ii];
  	}
	for(jj = 0; jj < nKnots; jj++){
  		eta[jj] = xvec[nCov + jj];
  	}

 	// make GSL versions as necessary
 	RcppGSL::vector<double> G_beta = Rcpp::as< RcppGSL::vector<double> >(beta);
 	RcppGSL::vector<double> G_eta = Rcpp::as< RcppGSL::vector<double> >(eta);
 	RcppGSL::matrix<double> G_bpred = Rcpp::as< RcppGSL::matrix<double> >(bpred);
  	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);

    /**************************************************
    // Computing cumulative baseline hazard functions
    **************************************************/
 
 	// initialize vectors for BH and cumulative BH
 	Rcpp::NumericVector Lam0Terms(nObs), Lambda0(nObs);
	
	// compute BH for each person
	for(i = 0; i < nObs; i++)
  	{
    	Lam0Terms[i] = lam0fn(i,G_bpred, G_eta);	
  	}
  	
  	Rcpp::NumericVector ySort = RcppSort(y);
  	Rcpp::NumericVector Lam0TermsSort = sortByVec(Lam0Terms, y);

  	// insert values for 0, and maximum values
  	ySort.insert(ySort.begin(), 0);
  	ySort.insert(ySort.end(), ySort[nObs]+0.1);
  	Lam0TermsSort.insert(Lam0TermsSort.begin(), Lam0TermsSort[0]);
  	Lam0TermsSort.insert(Lam0TermsSort.end(), Lam0TermsSort[nObs]);
	
    // interpolation for Lambda0 
    gsl_interp_accel *acc0 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda0 = gsl_interp_alloc(gsl_interp_linear, ySort.size());
	gsl_interp_init(interpLambda0, ySort.begin(), Lam0TermsSort.begin(), ySort.size());
	
	double result0;
	for(gg = 0; gg < nObs; gg++)
		{
			gsl_interp_eval_integ_e(interpLambda0,ySort.begin(),Lam0TermsSort.begin(),0,y[gg],acc0, &result0);
			Lambda0[gg]=result0;
		}

	gsl_interp_free(interpLambda0);
	gsl_interp_accel_free(acc0); 
	
	/******************************************************************************************
    // Computing integrals that get used in the eta score equations (int lambda0j(s)*Mjl(s))
    ******************************************************************************************/
    
    // initialize vectors for terms (will get reused) and matrices for integrals themselves
 	Rcpp::NumericVector Lam0TermsEta(nObs), Lam0TermsEtaSort(nObs);
	Rcpp::NumericMatrix Lambda0Eta(nObs,nKnots);
	double result0Eta;

	// looping through each knot
	for(kk = 0; kk < nKnots; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		Lam0TermsEta[i] = lam0fnEta(i, kk, G_bpred, G_eta);		
  		}
  		
  		// fill sort vectors
  		Lam0TermsEtaSort = sortByVec(Lam0TermsEta, y);
  		
  		// insert values for 0, and maximum values
  		Lam0TermsEtaSort.insert(Lam0TermsEtaSort.begin(), Lam0TermsEtaSort[0]);
  		Lam0TermsEtaSort.insert(Lam0TermsEtaSort.end(), Lam0TermsEtaSort[nObs]);

		// interpolation + integration for Lambda0Eta 
    	gsl_interp_accel *acc0Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda0Eta = gsl_interp_alloc(gsl_interp_linear, ySort.size());
		gsl_interp_init(interpLambda0Eta, ySort.begin(), Lam0TermsEtaSort.begin(), ySort.size());
	
		for(gg = 0; gg < nObs; gg++)
			{
				gsl_interp_eval_integ_e(interpLambda0Eta,ySort.begin(),Lam0TermsEtaSort.begin(),0,y[gg],acc0Eta, &result0Eta);
				Lambda0Eta(gg,kk)=result0Eta;
			}

		gsl_interp_free(interpLambda0Eta);
		gsl_interp_accel_free(acc0Eta); 
	}	
	
	// Put into GSL form
	RcppGSL::matrix<double> G_Lambda0Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda0Eta);
    
    /**************************************************
    // Computing score equations
    **************************************************/

	/*a = x2start[0];
	b = x2start[1];
	
	Rprintf("a = %f \n", a);
	Rprintf("b = %f \n", b);
	
	double u1Out = u1Ex(fakeXvals, a, b, nObs);
	Rprintf("u1 = %f \n", u1Out);		   
	
	double u2Out = u2Ex(fakeXvals, a, b, nObs);
	Rprintf("u2 = %f \n", u2Out);

  	
	// final score vector
	Rcpp::NumericVector scoreVec(2);
	scoreVec[0] = u1Out;
	scoreVec[1] = u2Out; */
	
	std::vector<double>  uBetaOut(nCov);
	uBetaOut = uBetaUniv(y, delta, G_x1, G_beta, wts, Lambda0, nCov, nObs);
	//Rprintf("uBeta, 1 = %f \n", uBetaOut[0]);
	//Rprintf("uBeta, 2 = %f \n", uBetaOut[1]);
	
	std::vector<double>  uEtaOut(nKnots);
	uEtaOut = uEtaUniv(y, delta, G_x1, G_beta, G_eta, G_Lambda0Eta, G_bpred, wts, nKnots, nObs);	
	//Rprintf("uEta, 1 = %f \n", uEtaOut[0]);
	//Rprintf("uEta, 2 = %f \n", uEtaOut[1]);
	//Rprintf("uEta, 3 = %f \n", uEtaOut[2]);
	//Rprintf("uEta, 4 = %f \n", uEtaOut[3]);
	//Rprintf("uEta, 5 = %f \n", uEtaOut[4]);
	//Rprintf("uEta, 6 = %f \n", uEtaOut[5]);
	
	// final score vector
	Rcpp::NumericVector scoreVec(nCov+nKnots);
	for(ii2 = 0; ii2 < nCov; ii2++){
  		scoreVec[ii2] = uBetaOut[ii2];
  	}
	for(jj2 = 0; jj2 < nKnots; jj2++){
  		scoreVec[nCov + jj2] = uEtaOut[jj2];
  	} 
	
	return scoreVec;

}

// [[Rcpp::export]] 
Rcpp::NumericVector spModelUnivFixedBeta(Rcpp::NumericVector xvec, Rcpp::NumericVector beta, int nObs, int nCov, int nKnots, const Rcpp::NumericVector y,
					Rcpp::NumericVector delta, Rcpp::NumericMatrix x1, Rcpp::NumericVector wts, Rcpp::NumericMatrix bpred)
{
	int gg, i, ii, ii2, kk;
	//double a(1), b(1);
	Rcpp::NumericVector eta(nKnots);
	 
	// initialize all parameters
	for(ii = 0; ii < nKnots; ii++){
  		eta[ii] = xvec[ii];
  	}

 	// make GSL versions as necessary
 	RcppGSL::vector<double> G_beta = Rcpp::as< RcppGSL::vector<double> >(beta);
 	RcppGSL::vector<double> G_eta = Rcpp::as< RcppGSL::vector<double> >(eta);
 	RcppGSL::matrix<double> G_bpred = Rcpp::as< RcppGSL::matrix<double> >(bpred);
  	RcppGSL::matrix<double> G_x1 = Rcpp::as< RcppGSL::matrix<double> >(x1);

    /**************************************************
    // Computing cumulative baseline hazard functions
    **************************************************/
 
 	// initialize vectors for BH and cumulative BH
 	Rcpp::NumericVector Lam0Terms(nObs), Lambda0(nObs);
	
	// compute BH for each person
	for(i = 0; i < nObs; i++)
  	{
    	Lam0Terms[i] = lam0fn(i,G_bpred, G_eta);	
  	}
  	
  	Rcpp::NumericVector ySort = RcppSort(y);
  	Rcpp::NumericVector Lam0TermsSort = sortByVec(Lam0Terms, y);

  	// insert values for 0, and maximum values
  	ySort.insert(ySort.begin(), 0);
  	ySort.insert(ySort.end(), ySort[nObs]+0.1);
  	Lam0TermsSort.insert(Lam0TermsSort.begin(), Lam0TermsSort[0]);
  	Lam0TermsSort.insert(Lam0TermsSort.end(), Lam0TermsSort[nObs]);
	
    // interpolation for Lambda0 
    gsl_interp_accel *acc0 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda0 = gsl_interp_alloc(gsl_interp_linear, ySort.size());
	gsl_interp_init(interpLambda0, ySort.begin(), Lam0TermsSort.begin(), ySort.size());
	
	double result0;
	for(gg = 0; gg < nObs; gg++)
		{
			gsl_interp_eval_integ_e(interpLambda0,ySort.begin(),Lam0TermsSort.begin(),0,y[gg],acc0, &result0);
			Lambda0[gg]=result0;
		}

	gsl_interp_free(interpLambda0);
	gsl_interp_accel_free(acc0); 
	
	/******************************************************************************************
    // Computing integrals that get used in the eta score equations (int lambda0j(s)*Mjl(s))
    ******************************************************************************************/
    
    // initialize vectors for terms (will get reused) and matrices for integrals themselves
 	Rcpp::NumericVector Lam0TermsEta(nObs), Lam0TermsEtaSort(nObs);
	Rcpp::NumericMatrix Lambda0Eta(nObs,nKnots);
	double result0Eta;

	// looping through each knot
	for(kk = 0; kk < nKnots; kk++)
	{
		// compute BH for each person
		for(i = 0; i < nObs; i++)
  		{
    		Lam0TermsEta[i] = lam0fnEta(i, kk, G_bpred, G_eta);		
  		}
  		
  		// fill sort vectors
  		Lam0TermsEtaSort = sortByVec(Lam0TermsEta, y);
  		
  		// insert values for 0, and maximum values
  		Lam0TermsEtaSort.insert(Lam0TermsEtaSort.begin(), Lam0TermsEtaSort[0]);
  		Lam0TermsEtaSort.insert(Lam0TermsEtaSort.end(), Lam0TermsEtaSort[nObs]);

		// interpolation + integration for Lambda0Eta 
    	gsl_interp_accel *acc0Eta = gsl_interp_accel_alloc();
		gsl_interp *interpLambda0Eta = gsl_interp_alloc(gsl_interp_linear, ySort.size());
		gsl_interp_init(interpLambda0Eta, ySort.begin(), Lam0TermsEtaSort.begin(), ySort.size());
	
		for(gg = 0; gg < nObs; gg++)
			{
				gsl_interp_eval_integ_e(interpLambda0Eta,ySort.begin(),Lam0TermsEtaSort.begin(),0,y[gg],acc0Eta, &result0Eta);
				Lambda0Eta(gg,kk)=result0Eta;
			}

		gsl_interp_free(interpLambda0Eta);
		gsl_interp_accel_free(acc0Eta); 
	}	
	
	// Put into GSL form
	RcppGSL::matrix<double> G_Lambda0Eta = Rcpp::as< RcppGSL::matrix<double> >(Lambda0Eta);
    
    /**************************************************
    // Computing score equations
    **************************************************/

	/*a = x2start[0];
	b = x2start[1];
	
	Rprintf("a = %f \n", a);
	Rprintf("b = %f \n", b);
	
	double u1Out = u1Ex(fakeXvals, a, b, nObs);
	Rprintf("u1 = %f \n", u1Out);		   
	
	double u2Out = u2Ex(fakeXvals, a, b, nObs);
	Rprintf("u2 = %f \n", u2Out);

  	
	// final score vector
	Rcpp::NumericVector scoreVec(2);
	scoreVec[0] = u1Out;
	scoreVec[1] = u2Out; */
	
	
	std::vector<double> uEtaOut(nKnots);
	uEtaOut = uEtaUniv(y, delta, G_x1, G_beta, G_eta, G_Lambda0Eta, G_bpred, wts, nKnots, nObs);	
	//Rprintf("uEta, 1 = %f \n", uEtaOut[0]);
	//Rprintf("uEta, 2 = %f \n", uEtaOut[1]);
	//Rprintf("uEta, 3 = %f \n", uEtaOut[2]);
	//Rprintf("uEta, 4 = %f \n", uEtaOut[3]);
	//Rprintf("uEta, 5 = %f \n", uEtaOut[4]);
	//Rprintf("uEta, 6 = %f \n", uEtaOut[5]);
	
	// final score vector
	Rcpp::NumericVector scoreVec(nKnots);
	for(ii2 = 0; ii2 < nKnots; ii2++){
  		scoreVec[ii2] = uEtaOut[ii2];
  	}

	return scoreVec;

}




// [[Rcpp::export]] 
Rcpp::NumericVector spModelExample(Rcpp::NumericVector params, Rcpp::NumericVector xInterp, Rcpp::NumericVector yInterp, const Rcpp::NumericVector xVals, int nObs)
{
	double a(1), b(1);
	int i, j;
	
	// INTERPOLATION SECTION
	
	// set up vectors for interpolation
	Rcpp::NumericVector yInterpTrue(11), Lambda01(11);
	
	// compute function that involves parameter
	for(i = 0; i < 11; i++)
  	{
    	yInterpTrue[i] = params[1]*yInterp[i];	
  	}
	
	// interpolation initiation
    gsl_interp_accel *acc01 = gsl_interp_accel_alloc();
	gsl_interp *interpLambda01 = gsl_interp_alloc(gsl_interp_linear, xInterp.size());
	gsl_interp_init(interpLambda01, xInterp.begin(), yInterpTrue.begin(), xInterp.size());
	
	double result01;
	for(j = 0; j < 11; j++)
		{
			// integration
			gsl_interp_eval_integ_e(interpLambda01,xInterp.begin(),yInterpTrue.begin(),0,xInterp[j],acc01, &result01);
			Lambda01[j]=result01;
		}

	gsl_interp_free(interpLambda01);
	gsl_interp_accel_free(acc01); 
	
	// END INTERPOLATION SECTION
	 
	// initialize all parameters
	a = params[0];
	b = params[1];

	
	double u1Out = u1Ex(xVals, a, b, nObs);
	//Rprintf("u1, = %f \n", u1Out);		   
	
	double u2Out = u2Ex(xVals, a, b, nObs);
	//Rprintf("u2 = %f \n", u2Out);

  	
	// final score vector
	Rcpp::NumericVector scoreVec(2);
	scoreVec[0] = u1Out;
	scoreVec[1] = u2Out;
	
	return scoreVec;

}
	
