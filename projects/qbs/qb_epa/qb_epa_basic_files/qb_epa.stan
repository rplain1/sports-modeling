functions{

}
data {
  int<lower=1> N;                  // number of observations
  int<lower=1> J;                  // number of passers
  array[N] int<lower=1, upper=J> passer_idx;  // index for each passer
  vector[N] y;                     // observed qb_epa
}

parameters {
  real<lower=0> sigma;             // observation noise
  real<lower=0> sigma_passer;      // passer effect std dev
  vector[J] passer_offset;         // raw passer effects
}

transformed parameters {
  // vector[J] passer_effect;
  // passer_effect = passer_offset * sigma_passer;
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
