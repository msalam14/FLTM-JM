data {
  int<lower=0> N_train;
  int<lower=0> obs;
  array[obs] int subject; // subject ID
  int<lower=0> K_ordi; // number of ordinal outcomes
  // real Y_conti[obs];
  array[obs, K_ordi] int<lower=0> Y_ordi;
  int<lower=0> n_ordi;
  vector[2] zero;
  vector<lower=0>[obs] time;  
  array[obs] int<lower=0> treat;
  array[N_train] int<lower=0> treat_pts;
  array[N_train] int<lower=0, upper=100> age_pts; 
  vector[N_train] tee; // observed failure time
  array[N_train] int<lower=0> event; // event status
}
parameters {
  vector[2] beta0;
  vector[2] beta1;
  
  matrix[N_train,2] U;
  real<lower=0> var1;
  real<lower=0> var2;
  real<lower=-1, upper=1> rho;
  //real e[obs];
  //real<lower=0> var_e;
  //real<lower=0> var_conti;
  
  real gamma;
  real nu;
  real h0;
  
  // real a_conti;
  // real<lower=0> b_conti;
  real a_ordi_temp;
  real<lower=0> b_ordi_temp;
  matrix<lower=0>[K_ordi,n_ordi-2] delta;
}
transformed parameters {
  real<lower=0> sig1;
  real<lower=0> sig2;
  cov_matrix[2] Sigma_U;
  // real<lower=0> sd_e;
  // real<lower=0> sd_conti;
  matrix[K_ordi,n_ordi-1] a_ordi;
  vector<lower=0>[K_ordi] b_ordi;
  vector[obs] theta;
  // real mu_conti[obs];
  array[obs, K_ordi, n_ordi] real<lower=0, upper=1> psi;
  array[obs, K_ordi, n_ordi] real<lower=0, upper=1>prob_y;
  vector<lower=0>[N_train] h;
  vector<lower=0,upper=1>[N_train] S;
  vector[N_train] LL;
  
  a_ordi[1, 1] = 0;
  for (l in 2:(n_ordi-1)) a_ordi[1, l] = a_ordi[1, l-1] + delta[1, l-1] ;
  for (k in 2:K_ordi) {
    a_ordi[k, 1] = a_ordi_temp;
    for (l in 2:(n_ordi-1)) a_ordi[k, l] = a_ordi[k, l-1] + delta[k, l-1];
  }
  b_ordi[1] = 1;
  for (k in 2:K_ordi) b_ordi[k] = b_ordi_temp;
  
  for (i in 1:obs)
    theta[i] = beta0[1] + beta0[2]*treat[i] + U[subject[i], 1] + 
    (beta1[1] + beta1[2]*treat[i] + U[subject[i], 2])*time[i];
  
  //for (i in 1:obs)
  //  mu_conti[i] <- a_conti + b_conti*theta[i];
  
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
    LL[i] = log(pow(h[i],event[i])*S[i]);  // event=1 for event; 0 for censored
  }

  //sd_e <- sqrt(var_e);
  //sd_conti <- sqrt(var_conti);
  sig1 = sqrt(var1);
  sig2 = sqrt(var2);
  Sigma_U[1,1] = sig1*sig1;
  Sigma_U[1,2] = rho*sig1*sig2;
  Sigma_U[2,1] = Sigma_U[1,2];
  Sigma_U[2,2] = sig2*sig2;
}
model {

  
  // construct random effects
  for(i in 1:N_train) {U[i]' ~ multi_normal(zero, Sigma_U);}
  // e ~ normal(0, sd_e);
  
  //Y_conti ~ normal(mu_conti, sd_conti);
  for (i in 1:obs) {
    for (k in 1:K_ordi) {
      Y_ordi[i, k] ~ categorical(to_vector(prob_y[i, k]));
    }
  }
  

  //increment_log_prob(LL);
  target += LL;
  
  // construct the priors
  beta0 ~ normal(0, 10);
  beta1 ~ normal(0, 10);
  var1 ~ inv_gamma(0.01, 0.01);
  var2 ~ inv_gamma(0.01, 0.01);
  rho ~ uniform(-1, 1);
  //var_e ~ inv_gamma(0.01, 0.01);
  //var_conti ~ inv_gamma(0.01, 0.01);
  
  h0 ~ normal(0, 10);
  nu ~ normal(0, 10);
  gamma ~ normal(0, 10);

  for (i in 1:(n_ordi-2)) delta[1, i] ~ normal(0, 10) T[0,] ;
  for (k in 2:K_ordi) {
    b_ordi_temp ~ uniform(0, 10);
    a_ordi_temp ~ normal(0, 10);
    for (i in 1:(n_ordi-2)) delta[k, i] ~ normal(0, 10) T[0,] ;
  }
}
