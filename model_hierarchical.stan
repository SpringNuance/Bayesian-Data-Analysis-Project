data {
  int<lower=1> Nmax; // Number of maximum URLs among all countries (training)
  int<lower=1> Mmax; // Number of maximum URLs among all countries (testing)
  int<lower=1> K; // Number of countries
  array[K] int<lower=1> N_list; // Number of URLs of each country (training)
  array[K] int<lower=1> M_list; // Number of URLs of each country (testing)
  array[K, Nmax] int<lower=0,upper=1> js_len_list;
  array[K, Nmax] int<lower=0,upper=1> js_obf_len_list;
  array[K, Nmax] int<lower=0,upper=1> https_list;
  array[K, Nmax] int<lower=0,upper=1> whois_list;
  //array[Nmax] vector[K] whois_list;
  //array[Nmax] array[K] int<lower=0,upper=1> whois_list;

  // The testing predicting features
  array[K, Mmax] int<lower=0,upper=1> js_len_pred_list;
  array[K, Mmax] int<lower=0,upper=1> js_obf_len_pred_list;
  array[K, Mmax] int<lower=0,upper=1> https_pred_list;
  array[K, Mmax] int<lower=0,upper=1> whois_pred_list;
  // label for each URL: benign(0) or malicious(1)
  array[K, Nmax] int<lower=0,upper=1> label_list; 
}

parameters {
  real<lower=0, upper=1> theta_js_len; // probability for js_len
  real<lower=0, upper=1> theta_js_obf_len; // probability for js_obf_len
  real<lower=0, upper=1> theta_https; // probability for https
  real<lower=0, upper=1> theta_whois; // probability for whois
  array[K] real js_len_coeff; // Slope coefficient for js
  array[K] real js_obf_len_coeff; // Slope coefficient for js
  array[K] real https_coeff; // Slope coefficient for js
  array[K] real whois_coeff; // Slope coefficient for js
  array[K] real intercept; // Intercept coefficient for js
  real mu0;
  real sigma0;
  real mujs;
  real musafety;
  real sigma;
}

model {
    theta_js_len ~ beta(1,10);
    theta_js_obf_len ~ beta(1,10);
    theta_https ~ beta(8,10);
    theta_whois ~ beta(7,10);
    // likelihood for safety and js
    for (k in 1:K){
        js_len_list[k, 1:N_list[k]] ~ bernoulli(theta_js_len);
        js_obf_len_list[k, 1:N_list[k]] ~ bernoulli(theta_js_obf_len);
        https_list[k, 1:N_list[k]] ~ bernoulli(theta_https);
        whois_list[k, 1:N_list[k]] ~ bernoulli(theta_whois);     
    }

    // The distribution of the mean and std for the coefficients
    // The coefficients
    mu0 ~ normal(0,20);
    sigma0 ~ normal(0,10);
    mujs ~ normal(mu0,sigma0);
    musafety ~ normal(mu0,sigma0);
    sigma ~ normal(0,10);
    for (k in 1:K){
      js_len_coeff[k] ~ normal(mujs,sigma);
      js_obf_len_coeff[k] ~ normal(mujs,sigma);
      https_coeff[k] ~ normal(musafety,sigma);
      whois_coeff[k] ~ normal(musafety,sigma);
      intercept[k] ~ normal(0,20);   
    }

    //js_coeff ~ normal(0,20);
    //safety_coeff ~ normal(0,20);
    // Modelling of the label based on bernoulli logistic regression by multiple variable linear regression 
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_list[k, i] ~ bernoulli(inv_logit(intercept[k] + https_coeff[k] * https_list[k, i] + whois_coeff[k] * whois_list[k, i] + js_len_coeff[k] * js_len_list[k, i] + js_obf_len_coeff[k] * js_obf_len_list[k, i]));
      }
    }
}

generated quantities {
    array[K, Nmax] real label_train_pred;
    array[K, Mmax] real label_test_pred;
    // Predictions for the training data
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_train_pred[k, i] = bernoulli_rng(inv_logit(intercept[k] + https_coeff[k] * https_list[k, i] + whois_coeff[k] * whois_list[k, i] + js_len_coeff[k] * js_len_list[k, i] + js_obf_len_coeff[k] * js_obf_len_list[k, i]));
      }
    }
    // Predictions for the testing data
    for (k in 1:K){
      for (i in 1:M_list[k]){
        label_test_pred[k, i] = bernoulli_rng(inv_logit(intercept[k] + https_coeff[k] * https_pred_list[k, i] + whois_coeff[k] * whois_pred_list[k, i] + js_len_coeff[k] * js_len_pred_list[k, i] + js_obf_len_coeff[k] * js_obf_len_pred_list[k, i]));
      }
    }
}
