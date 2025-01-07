data {
  int<lower=0> N; // number of subject
  int<lower=0> P_surv; // number of covariates in survival model
  matrix[N, P_surv] W; // covariates in survival model that has direct effect
  vector<lower=0>[N] surv_time; // survival time
  vector[N] status; // censoring status
  
  int<lower=0> n_basis; // We will use ten bases for B-spline representation
  int Tgrid; // a fine grid for functional approximation
  matrix[Tgrid,n_basis] BasisM; // b-splien basis matrix on the fine grid
  array[N] int TGpos; // closest position in the time grid for observed failure time
  vector<lower=0>[Tgrid] cHW; // weights for numerical integration
  cov_matrix[n_basis] PenMat;  // prior precision matrix for spline effects (penalty matrix)
  
  int<lower=0> obs; // total number of longitudinal observations
  array[obs] int subID; // subject ID
  array[obs] int longPOS; // closest position in fine grid for observed times for longitudinal data
  
  int<lower=1> K_ordi; // number of ordinal variables under study
  array[K_ordi] int<lower=0> n_ordi; // number of categories in different ordinal variables
  int<lower=0> total_cat; // total categories across all ordinal variables
  array[obs,K_ordi] int<lower=0> y_val; // matrix of longitudinal data obs by K_ordi dimensional matrix
  int<lower=0> M; // number of eigen functions for subject specific deviation
  int<lower=0> lt_cov; // number of covariates in longitudinal sub-model
  matrix[N,lt_cov] Xdesign; // time-invariant covariates for longitudinal sub-model
}

transformed data{
  vector[n_basis] mu_bh;
  array[K_ordi] int  ck;
  array[K_ordi] int  pck;

  for(k in 1:n_basis) mu_bh[k] = 0;
  for (k in 1:K_ordi) ck[k]=sum(n_ordi[1:k])-k;
  for (k in 1:K_ordi) pck[k]=sum(n_ordi[1:k]);
}

parameters {
  vector[n_basis] bh_coef; // baseline hazard coefficients
  real<lower=0> bh_sig; // variance parameter for bh_coef 
  
  vector[P_surv] gam_par; // coefficient vectors for survial covariates
  
  real<lower=-1,upper=1> rho_par; // parameter associated with latent trait in survial model

  vector[K_ordi-1] a_ordi_temp; // intercept for longitudinal model
  vector<lower=0>[K_ordi-1] b_ordi_temp; // slope for the longitudinal model
  vector<lower=0>[total_cat-K_ordi-K_ordi] delta; // shift in the intercept parameters
  
  matrix[lt_cov,n_basis] beta_par; // coefficient functions for covariates in latent trait
  vector<lower=0>[lt_cov] beta_sig; // variances for coefficient functions
  
  matrix[M,n_basis] coef_par; // b-spline coefficients that define the eigenfunctions
  vector<lower=0>[M] coef_sig; // variance parameters associated with coefficients that define eigenfunctions
  matrix[N,M] xi; // subject-specific scores for M eigen functions
}

transformed parameters {
  vector<lower=0>[N] hazF; // hazard function
  vector<lower=0>[N] cum_b_hazard; // cumulative hazard function
  vector[N] LL; // survival likelihood
  
  matrix[M,Tgrid] PsiM; // vector of eigenfunction
  vector[M] sing_val;
  vector<lower=0>[M] lmd; // vector eigenvalues
  matrix[M,M] U; // left singular vectors
  matrix[Tgrid,M] V; // right singular vectors 
  matrix[N,M] Tx; // transformed scores

  vector[total_cat-K_ordi] a_ordi; // edited by samsul
  vector<lower=0>[K_ordi] b_ordi; // edited by samsul
  matrix<lower=0, upper=1>[obs,total_cat] psi_prob; // cumulative probabilities
  matrix<lower=0, upper=1>[obs,total_cat] prob_y; // probabilities
  
  vector[obs] ltheta;
  //vector[Tgrid] stheta;
  vector[Tgrid] lbhz_fun;
  matrix[N,Tgrid] surv_theta;

  PsiM = (BasisM * coef_par')';
  sing_val = singular_values(PsiM);
  for(k in 1:M){
    if(sing_val[k]<0){
      lmd[k] = 0;
    } else{
      lmd[k] = sing_val[k];
    }
  } 
  U = svd_U(PsiM);
  V = svd_V(PsiM)*sqrt(Tgrid);
  //for(i in 1:N) Tx[i] = (xi[i]*diag_post_multiply(U,lmd*pow(sqrt(Tgrid),-1))); 
  Tx = xi*diag_post_multiply(U,lmd*pow(sqrt(Tgrid),-1)); 

  for (k in 1:K_ordi) {
    if(k==1){
      a_ordi[k] = 0;
      for (l in 2:(n_ordi[k]-1)) a_ordi[l] = a_ordi[l-1] + delta[l-1];
    } else{
      a_ordi[ck[k-1]+1] = a_ordi_temp[k-1];
      for (l in 2:(n_ordi[k]-1)) a_ordi[(ck[k-1]+l)] = a_ordi[ck[k-1]+l-1] + delta[ck[k-1]-(k-1)+l-1];
    }
  }
  b_ordi[1]=1;
  for(k in 2:K_ordi) b_ordi[k] = b_ordi_temp[k-1];

  for (i in 1:obs) {
    ltheta[i] = (Xdesign[subID[i]]*(BasisM[longPOS[i]]*beta_par')')+(V[longPOS[i]]*Tx[subID[i]]');
    for(g in pck) psi_prob[i,g] = 1;
    for(k in 1:K_ordi){
      if(k==1){
        for(l in 1:(n_ordi[k]-1)) psi_prob[i,l] = inv_logit(a_ordi[l] - b_ordi[k]*ltheta[i]);
      } else{
        for(l in 1:(n_ordi[k]-1)) psi_prob[i,(pck[k-1]+l)] = inv_logit(a_ordi[(ck[k-1]+l)] - b_ordi[k]*ltheta[i]);
      }
    }
    
    for (k in 1:K_ordi) {
      if(k==1){
        prob_y[i,k] = psi_prob[i,k];
        for (l in 2:n_ordi[k]) prob_y[i, l] = psi_prob[i, l] - psi_prob[i, (l-1)];
      } else{
        prob_y[i,(pck[k-1]+1)] = psi_prob[i,(pck[k-1]+1)];
        for (l in 2:n_ordi[k]) prob_y[i, (pck[k-1]+l)] = psi_prob[i, (pck[k-1]+l)] - psi_prob[i, (pck[k-1]+l-1)];
      }
    }
  }
  
  lbhz_fun = BasisM*bh_coef;
  for(i in 1:N){
    surv_theta[i] = (((BasisM*beta_par')*Xdesign[i]')+(V*Tx[i]'))';
    hazF[i] = exp(lbhz_fun[TGpos[i]]+(W[i]*gam_par)+(rho_par*surv_theta[i,TGpos[i]]));
    cum_b_hazard[i] = sum(segment((cHW.*exp(lbhz_fun+(W[i]*gam_par)+(rho_par*surv_theta[i])')),1,TGpos[i]));
    LL[i] = (status[i] *log(hazF[i])) + (-cum_b_hazard[i]);
  }
}

model {
  
  bh_coef~multi_normal(mu_bh,bh_sig*PenMat);
  bh_sig ~ inv_gamma(0.01, 0.01);

  for(j in 1:P_surv) {
    gam_par[j] ~ normal(0,10);
  }
  
  for(j in 1:M) coef_par[j]' ~ multi_normal(mu_bh,coef_sig[j]*PenMat);
  
  for(j in 1:M) coef_sig[j] ~ inv_gamma(0.01, 0.01);
  
  rho_par ~ uniform(-1,1);
  
  for(i in 1:N){
    for(j in 1:M){
      xi[i,j] ~ normal(0,1);
    }
  }
  
  for (i in 1:(total_cat-K_ordi-K_ordi)) delta[i] ~ normal(0, 10) T[0,]; // Truncated from 0,infty
  for (k in 1:(K_ordi-1)) {
    b_ordi_temp[k] ~ uniform(0, 10);
    a_ordi_temp[k] ~ normal(0, 10);
  }
  
  for(i in 1:lt_cov) beta_par[i]' ~ multi_normal(mu_bh,beta_sig[i]*PenMat);
  for(j in 1:lt_cov) beta_sig[j] ~ inv_gamma(0.01, 0.01);

  
  for(i in 1:obs) {
    for(k in 1:K_ordi){
      if(k==1){
        target+= log(prob_y[i,y_val[i,k]]);
      } else{
        target+= log(prob_y[i,(pck[k-1]+y_val[i,k])]);
      }
    }
  }

  target += LL;

}


generated quantities {
  vector[N] log_lik;
  vector[N] ll_i;
  ll_i=zeros_vector(N);
  for (i in 1:obs) {
    for(k in 1:K_ordi){
      if(k==1){
        ll_i[subID[i]] += log(prob_y[i,y_val[i,k]]);
      } else{
        ll_i[subID[i]] += log(prob_y[i,(pck[k-1]+y_val[i,k])]);
      }
    }
  }
  log_lik = ll_i + LL;
}
