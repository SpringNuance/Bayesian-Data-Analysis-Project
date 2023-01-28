data {
  int<lower=1> N; // Number of URLs of a country (training)
  int<lower=1> M; // Number of URLs of a country (testing)
  // The training features
  // safety and js are categorical, following the categorical distribution
  // js = 2: js_len >= 250
  // js = 1: js_len < 250, js_obf_len < 100
  // js = 0; js_len < 250, js_obf_len >= 100
  array[N] int<lower=0,upper=2> js;
  
  // safety = 1 (safe): https = yes, whois = complete
  // safety = 2 (neutral): https = no, whois = incomplete
  // safety = 3 (neutral): https = yes, whois = complete
  // safety = 4 (dangerous): https = no, whois = incomplete
  array[N] int<lower=1,upper=4> safety;

  // The testing predicting features
  array[M] int<lower=0,upper=2> js_pred;
  array[M] int<lower=0,upper=4> safety_pred;
  
  // label for each URL: benign(0) or malicious(1)
  array[N] int<lower=0,upper=1> label; 
}

parameters {
  real js_coeff; // Slope coefficient for js
  real safety_coeff; // Slope coefficient for js
  real intercept; // Intercept coefficient for js
  real mu_js; // Slope coefficient mean for js
  real mu_safety; // Slope coefficient mean for js
  real mu_intercept; // Intercept coefficient mean for js
  real sigma_js; // Slope coefficient std for js
  real sigma_safety; // Slope coefficient std for js
  real sigma_intercept; // Intercept coefficient std for js
  simplex[4] probs_safety; // Categorical probabilities for safety, sum = 1
  simplex[3] probs_js; // Categorical probabilities for js, sum = 1
}

model {
    // prior for safety and js categorical probabilities
    for (i in 1:4){
      probs_safety[i] ~ normal(0.5, 0.4);
    }
    for (i in 1:3){
      probs_js[i] ~ normal(0.5, 0.3);
    }
    // likelihood for safety and js
    for (i in 1:N) {
      safety[i] ~ categorical(probs_safety);
    }
    for (i in 1:N) {
      js[i] ~ categorical(probs_js);
    }
    // The distribution of the mean and std for the coefficients
    // mu_js ~ gamma(3,1);
    // mu_safety ~ gamma(3,1);
    //mu_js ~ normal(0,50);
    //mu_safety ~ normal(0,50);
    //mu_intercept ~ normal(0,50);
    //sigma_js ~ gamma(1,1);
    //sigma_safety ~ gamma(1,1);
    //sigma_js ~ normal(0,20);
    //sigma_safety ~ normal(0,20);
    //sigma_intercept ~ normal(0,20);
    // weakly informative priors for the coefficients and intercept
    //js_coeff ~ normal(mu_js, sigma_js);
    //safety_coeff ~ normal(mu_safety, sigma_safety);
    //intercept ~ normal(mu_intercept, sigma_intercept);
    js_coeff ~ normal(mu_js, sigma_js);
    safety_coeff ~ normal(mu_safety, sigma_safety);
    intercept ~ normal(mu_intercept, sigma_intercept);
    //js_coeff ~ normal(0, 10);
    //safety_coeff ~ normal(0, 10);
    //intercept ~ normal(0, 10);
    // Modelling of the label based on bernoulli logistic regression by multiple variable linear regression 
    for (i in 1:N){
      label[i] ~ bernoulli(inv_logit(intercept + safety_coeff * safety[i] + js_coeff * js[i]));
    }
}

generated quantities {
    vector[N] label_train_pred;
    vector[M] label_test_pred;
    // Predictions for the training data
    for (i in 1:N){
      label_train_pred[i] = bernoulli_rng(inv_logit(intercept + safety_coeff * safety[i] + js_coeff * js[i]));
    }
    // Predictions for the testing data
    for (i in 1:M){
      label_test_pred[i] = bernoulli_rng(inv_logit(intercept + safety_coeff * safety_pred[i] + js_coeff * js_pred[i]));
    }
}
