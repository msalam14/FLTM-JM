data {
  int<lower=0> N; // number of subject
  int<lower=0> P_surv; // number of covariates in survival model
  matrix[N, P_surv] W;
  vector<lower=0>[N] surv_time; // survival time

  int<lower=0> n_basis; // We will use ten bases for B-spline representation
  int Tgrid;
  matrix[Tgrid,n_basis] BasisM;
  array[N] int TGpos;
  vector<lower=0>[Tgrid] cHW;

  int<lower=0> obs;
  array[obs] int subID; // subject ID
  array[obs] int longPOS; // closest position in fine grid for observed times for longitudinal data

  int<lower=1> K_ordi;
  array[K_ordi] int<lower=0> n_ordi;
  int<lower=0> total_cat;
  array[obs,K_ordi] int<lower=0> y_val;
  int<lower=0> M;
  int<lower=0> lt_cov;
  matrix[N,lt_cov] Xdesign;

  // summary of the posterior samples
  vector[P_surv] gam_par;
  vector[n_basis] bh_coef;

  matrix[lt_cov,n_basis] beta_par;
  matrix[Tgrid,M] V;
  real<lower=-1,upper=1> rho_par;
  vector<lower=0>[M] lmda_par;
  
  vector[total_cat-K_ordi] a_ordi; // edited by samsul
  vector<lower=0>[K_ordi] b_ordi;
}

transformed data {
  array[K_ordi] int  ck;
  array[K_ordi] int  pck;

  for (k in 1:K_ordi) ck[k]=sum(n_ordi[1:k])-k;
  for (k in 1:K_ordi) pck[k]=sum(n_ordi[1:k]);
}

parameters {
  matrix[N,M] Tx;
}

transformed parameters {
  vector<lower=0>[N] cum_b_hazard;
  vector[N] LL;

  matrix<lower=0, upper=1>[obs,total_cat] psi_prob;
  matrix<lower=0, upper=1>[obs,total_cat] prob_y;
  
  real ltheta;
  vector[Tgrid] stheta;
  vector[Tgrid] lbhz_fun;


  for (i in 1:obs) {
    ltheta = (Xdesign[subID[i]]*(BasisM[longPOS[i]]*beta_par')')+(V[longPOS[i]]*Tx[subID[i]]');
    for(g in pck) psi_prob[i,g] = 1;
    for(k in 1:K_ordi){
      if(k==1){
        for(l in 1:(n_ordi[k]-1)) psi_prob[i,l] = inv_logit(a_ordi[l] - b_ordi[k]*ltheta);
      } else{
        for(l in 1:(n_ordi[k]-1)) psi_prob[i,(pck[k-1]+l)] = inv_logit(a_ordi[(ck[k-1]+l)] - b_ordi[k]*ltheta);
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
    stheta = rho_par*(((BasisM*beta_par')*Xdesign[i]')+(V*Tx[i]'));
    cum_b_hazard[i] = sum(segment((cHW.*exp(lbhz_fun+(W[i]*gam_par)+stheta)),1,TGpos[i]));
    LL[i] = (-cum_b_hazard[i]);
  }
}

model {
  
  for(i in 1:N){
    for(j in 1:M){
      Tx[i,j] ~ normal(0,sqrt(lmda_par[j]));
    }
  } 
  
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

