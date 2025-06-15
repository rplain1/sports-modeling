functions{

}
data {
  int<lower=1> N;                  // number of observations
  int<lower=1> n_passers;          // number of passers
  array[N] int<lower=1, upper=n_passers> passer_idx;  // index for each passer
  vector[N] y;                     // observed qb_epa
}

parameters {
  real<lower=0> sigma;             // observation noise
  real<lower=0> sigma_passer;      // passer effect std dev
  vector[n_passers] passer_offset;         // raw passer effects
}

transformed parameters {
  vector[n_passers] passer_effect;
  passer_effect = passer_offset * sigma_passer;
}

model {
  // Priors
  sigma ~ normal(0, 0.5);
  sigma_passer ~ normal(0, 0.5);
  passer_offset ~ normal(-0.2, 1);

  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(passer_effect[passer_idx[n]], sigma);
  }
}
generated quantities {
  vector[N] y_rep;
  for (n in 1:N) {
    y_rep[n] = normal_rng(passer_effect[passer_idx[n]], sigma);
  }
}
