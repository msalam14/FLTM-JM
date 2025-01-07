data {
  int<lower=0> N_train;
  int<lower=0> obs;
  array[obs] int subject; // subject ID
  int<lower=0> K_ordi; // number of ordinal outcomes
  array[obs, K_ordi] int<lower=0> Y_ordi;
  int<lower=0> n_ordi;
  vector[2] zero;
  vector<lower=0>[obs] time;  
  array[obs] int<lower=0> treat;
  array[N_train] int<lower=0> treat_pts;
  array[N_train] int<lower=0, upper=100> age_pts; 
  vector[N_train] tee; // observed failure time

  // estimated parameters as data
  vector[2] beta0;
  vector[2] beta1;
  
  real<lower=0> var1;
  real<lower=0> var2;
  real<lower=-1, upper=1> rho;

  real gamma;
  real nu;
  real h0;
  
  matrix[K_ordi,n_ordi-1] a_ordi;
  vector<lower=0>[K_ordi] b_ordi;
}

parameters {
  matrix[N_train,2] U;
}

transformed parameters {
  real<lower=0> sig1;
  real<lower=0> sig2;
  cov_matrix[2] Sigma_U;
  vector[obs] theta;
  // real mu_conti[obs];
  array[obs, K_ordi, n_ordi] real<lower=0, upper=1> psi;
  array[obs, K_ordi,n_ordi] real<lower=0, upper=1> prob_y;
  vector<lower=0>[N_train] h;
  vector<lower=0,upper=1>[N_train] S;
  vector[N_train] LL;
  
  for (i in 1:obs)
    theta[i] = beta0[1] + beta0[2]*treat[i] + U[subject[i], 1] + 
    (beta1[1] + beta1[2]*treat[i] + U[subject[i], 2])*time[i];
  
  for (i in 1:obs) {
    for (k in 1:K_ordi) {
      for (l in 1:(n_ordi-1)) {
        psi[i, k, l] = inv_logit(a_ordi[k, l] - b_ordi[k]*theta[i]);
      }
      psi[i, k, n_ordi] = 1;
      
      prob_y[i, k, 1] = psi[i, k, 1];
      for (l in 2:n_ordi) {prob_y[i, k, l] = psi[i, k, l] - psi[i, k, l-1];}
    }
  }
  
  // construct survival part
  for (i in 1:N_train) {
    h[i] = exp(gamma*age_pts[i] + nu*(beta0[1] + beta0[2]*treat_pts[i] + U[i, 1] + 
                                      (beta1[1] + beta1[2]*treat_pts[i] + U[i, 2])*tee[i]))*exp(h0);
    S[i] = exp(-exp(h0)*exp(gamma*age_pts[i]+nu*(beta0[1]+beta0[2]*treat_pts[i]+U[i, 1])) * 
                  (exp(nu*(beta1[1]+beta1[2]*treat_pts[i]+U[i, 2])*tee[i])-1) / (nu*(beta1[1]+beta1[2]*treat_pts[i]+U[i, 2])));
    LL[i] = log(S[i]);  // event=1 for event; 0 for censored
  }

  sig1 = sqrt(var1);
  sig2 = sqrt(var2);
  Sigma_U[1,1] = sig1*sig1;
  Sigma_U[1,2] = rho*sig1*sig2;
  Sigma_U[2,1] = Sigma_U[1,2];
  Sigma_U[2,2] = sig2*sig2;
}

model {
  //U ~ multi_normal(zero, Sigma_U);
  for(i in 1:N_train) {U[i]' ~ multi_normal(zero, Sigma_U);}

  for (i in 1:obs) {
    for (k in 1:K_ordi) {
      Y_ordi[i, k] ~ categorical(to_vector(prob_y[i, k]));
    }
  }
  
  //increment_log_prob(LL);
  target += LL;
}
