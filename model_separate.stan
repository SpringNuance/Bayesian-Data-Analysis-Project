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
  real<lower=0, upper=1> theta_js_len;
  real<lower=0, upper=1> theta_js_obf_len;
  real<lower=0, upper=1> theta_https;
  real<lower=0, upper=1> theta_whois;
  real js_len_coeff; // Slope coefficient for js
  real js_obf_len_coeff; // Slope coefficient for js
  real https_coeff; // Slope coefficient for js
  real whois_coeff; // Slope coefficient for js
  real intercept; // Intercept coefficient for js
}

model {
    theta_js_len ~ beta(1,1);
    theta_js_obf_len ~ beta(1,1);
    theta_https ~ beta(1,1);
    theta_whois ~ beta(1,1);
    // likelihood for safety and js
    js_len ~ bernoulli(theta_js_len);
    js_obf_len ~ bernoulli(theta_js_obf_len);
    https ~ bernoulli(theta_https);
    whois ~ bernoulli(theta_whois);
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
    //js_coeff ~ normal(mu_js, sigma_js);
    //safety_coeff ~ normal(mu_safety, sigma_safety);
    //intercept ~ normal(mu_intercept, sigma_intercept);
    js_len_coeff ~ normal(0, 10);
    js_obf_len_coeff ~ normal(0, 10);
    https_coeff ~ normal(0, 10);
    whois_coeff ~ normal(0,10);
    intercept ~ normal(0, 10);
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
      label_train_pred[i] = bernoulli_rng(inv_logit(intercept + https_coeff * (https[i] + 1) + whois_coeff * (whois[i] + 1) + js_len_coeff * (js_len[i] + 1) + js_obf_len_coeff * (js_obf_len[i] + 1)));
    }
    // Predictions for the testing data
    for (i in 1:M){
      label_test_pred[i] = bernoulli_rng(inv_logit(intercept + https_coeff * (https_pred[i] + 1) + whois_coeff * (whois_pred[i] + 1) + js_len_coeff * (js_len_pred[i] + 1) + js_obf_len_coeff * (js_obf_len_pred[i] + 1)));
    }
}
