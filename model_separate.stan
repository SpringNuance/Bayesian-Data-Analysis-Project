data {
  int<lower=1> N; // Number of URLs of a country (training)
  int<lower=1> M; // Number of URLs of a country (testing)
  array[N] int<lower=0,upper=1> js_len;
  array[N] int<lower=0,upper=1> js_obf_len;
  array[N] int<lower=0,upper=1> https;
  array[N] int<lower=0,upper=1> whois;

  // The testing predicting features
  array[M] int<lower=0,upper=1> js_len_pred;
  array[M] int<lower=0,upper=1> js_obf_len_pred;
  array[M] int<lower=0,upper=1> https_pred;
  array[M] int<lower=0,upper=1> whois_pred;
  // label for each URL: benign(0) or malicious(1)
  array[N] int<lower=0,upper=1> label; 
}

parameters {
  real<lower=0, upper=1> theta_js_len; // probability for js_len
  real<lower=0, upper=1> theta_js_obf_len; // probability for js_obf_len
  real<lower=0, upper=1> theta_https; // probability for https
  real<lower=0, upper=1> theta_whois; // probability for whois
  real js_len_coeff; // Slope coefficient for js
  real js_obf_len_coeff; // Slope coefficient for js
  real https_coeff; // Slope coefficient for js
  real whois_coeff; // Slope coefficient for js
  real intercept; // Intercept coefficient for js
}

model {
    // Prior for the probabilities
    theta_js_len ~ beta(1,10);
    theta_js_obf_len ~ beta(1,10);
    theta_https ~ beta(8,10);
    theta_whois ~ beta(7,10);
    // likelihood for safety and js
    js_len ~ bernoulli(theta_js_len);
    js_obf_len ~ bernoulli(theta_js_obf_len);
    https ~ bernoulli(theta_https);
    whois ~ bernoulli(theta_whois);
    // Weakly informative prior for the coefficients
    js_len_coeff ~ normal(0,20);
    js_obf_len_coeff ~ normal(0,20);
    https_coeff ~ normal(0,20);
    whois_coeff ~ normal(0,20);
    intercept ~ normal(0,20);
    // Modelling of the label based on bernoulli logistic regression by multiple variable linear regression 
    for (i in 1:N){
      label[i] ~ bernoulli(inv_logit(intercept + https_coeff * https[i] + whois_coeff * whois[i] + js_len_coeff * js_len[i] + js_obf_len_coeff * js_obf_len[i]));
    }
}

generated quantities {
    vector[N] label_train_pred;
    vector[M] label_test_pred;
    // Predictions for the training data
    for (i in 1:N){
      label_train_pred[i] = bernoulli_rng(inv_logit(intercept + https_coeff * https[i] + whois_coeff * whois[i] + js_len_coeff * js_len[i] + js_obf_len_coeff * js_obf_len[i]));
    }
    // Predictions for the testing data
    for (i in 1:M){
      label_test_pred[i] = bernoulli_rng(inv_logit(intercept + https_coeff * https_pred[i] + whois_coeff * whois_pred[i] + js_len_coeff * js_len_pred[i] + js_obf_len_coeff * js_obf_len_pred[i]));
    }
}
