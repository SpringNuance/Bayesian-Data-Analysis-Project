data {
  int<lower=1> Nmax; // Number of maximum URLs among all countries (training)
  int<lower=1> Mmax; // Number of maximum URLs among all countries (testing)
  int<lower=1> K; // Number of countries
  array[K] int<lower=1> N_list; // Number of URLs of each country (training)
  array[K] int<lower=1> M_list; // Number of URLs of each country (testing)
  array[Nmax, K] int<lower=0,upper=1> js_len_list;
  array[Nmax, K] int<lower=0,upper=1> js_obf_len_list;
  array[Nmax, K] int<lower=0,upper=1> https_list;
  array[Nmax, K] int<lower=0,upper=1> whois_list;
  //array[Nmax] vector[K] whois_list;
  //array[Nmax] array[K] int<lower=0,upper=1> whois_list;

  // The testing predicting features
  array[Mmax, K] int<lower=0,upper=1> js_len_pred_list;
  array[Mmax, K] int<lower=0,upper=1> js_obf_len_pred_list;
  array[Mmax, K] int<lower=0,upper=1> https_pred_list;
  array[Mmax, K] int<lower=0,upper=1> whois_pred_list;
  // label for each URL: benign(0) or malicious(1)
  array[Nmax, K] int<lower=0,upper=1> label_list; 
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
  real mu0js;
  real sigma0js;
  real mu0safety;
  real sigma0safety;
}

model {
    theta_js_len ~ beta(1,10);
    theta_js_obf_len ~ beta(1,10);
    theta_https ~ beta(8,10);
    theta_whois ~ beta(7,10);
    // likelihood for safety and js
    for (k in 1:K){
        js_len_list[1:N_list[k], k] ~ bernoulli(theta_js_len);
        js_obf_len_list[1:N_list[k], k] ~ bernoulli(theta_js_obf_len);
        https_list[1:N_list[k], k] ~ bernoulli(theta_https);
        whois_list[1:N_list[k], k] ~ bernoulli(theta_whois);     
    }

    // The distribution of the mean and std for the coefficients
    // The coefficients
    mu0js ~ normal(0,20);
    sigma0js ~ normal(0,10);
    mu0safety ~ normal(0,20);
    sigma0safety ~ normal(0,10);
    for (k in 1:K){
      js_len_coeff[k] ~ normal(mu0js,sigma0js);
      js_obf_len_coeff[k] ~ normal(mu0js,sigma0js);
      https_coeff[k] ~ normal(mu0safety,sigma0safety);
      whois_coeff[k] ~ normal(mu0safety,sigma0safety);
      intercept[k] ~ normal(0,20);   
    }

    //js_coeff ~ normal(0,20);
    //safety_coeff ~ normal(0,20);
    // Modelling of the label based on bernoulli logistic regression by multiple variable linear regression 
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_list[i, k] ~ bernoulli(inv_logit(intercept[k] + https_coeff[k] * https_list[i, k] + whois_coeff[k] * whois_list[i, k] + js_len_coeff[k] * js_len_list[i, k] + js_obf_len_coeff[k] * js_obf_len_list[i, k]));
      }
    }
}

generated quantities {
    array[Nmax, K] real label_train_pred;
    array[Mmax, K] real label_test_pred;
    // Predictions for the training data
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_train_pred[i, k] = bernoulli_rng(inv_logit(intercept[k] + https_coeff[k] * https_list[i, k] + whois_coeff[k] * whois_list[i, k] + js_len_coeff[k] * js_len_list[i, k] + js_obf_len_coeff[k] * js_obf_len_list[i, k]));
      }
    }
    // Predictions for the testing data
    for (k in 1:K){
      for (i in 1:M_list[k]){
        label_test_pred[i, k] = bernoulli_rng(inv_logit(intercept[k] + https_coeff[k] * https_pred_list[i, k] + whois_coeff[k] * whois_pred_list[i, k] + js_len_coeff[k] * js_len_pred_list[i, k] + js_obf_len_coeff[k] * js_obf_len_pred_list[i, k]));
      }
    }
}
