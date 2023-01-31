data {
  int<lower=1> Nmax; // Number of maximum URLs among all countries (training)
  int<lower=1> Mmax; // Number of maximum URLs among all countries (testing)
  int<lower=1> K; // Number of countries
  array[K] int<lower=1> N_list; // Number of URLs of each country (training)
  array[K] int<lower=1> M_list; // Number of URLs of each country (testing)
  // The training features
  array[K, Nmax] int<lower=0,upper=1> js_len_list;
  array[K, Nmax] int<lower=0,upper=1> js_obf_len_list;
  array[K, Nmax] int<lower=0,upper=1> https_list;
  array[K, Nmax] int<lower=0,upper=1> whois_list;
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
  real js_len_coeff; // Slope coefficient for js_len
  real js_obf_len_coeff; // Slope coefficient for js_obf_len
  real https_coeff; // Slope coefficient for https_coeff
  real whois_coeff; // Slope coefficient for whois_coeff
  real intercept; // Intercept coefficient
}

model {
    // Prior probabilities of the features
    theta_js_len ~ beta(1,4);
    theta_js_obf_len ~ beta(1,4);
    theta_https ~ beta(1,2);
    theta_whois ~ beta(1,2);
    // likelihood for the features
    for (k in 1:K){
        js_len_list[k, 1:N_list[k]] ~ bernoulli(theta_js_len);
        js_obf_len_list[k, 1:N_list[k]] ~ bernoulli(theta_js_obf_len);
        https_list[k, 1:N_list[k]] ~ bernoulli(theta_https);
        whois_list[k, 1:N_list[k]] ~ bernoulli(theta_whois);     
    }
    // priors of the coefficients
    js_len_coeff ~ cauchy(1,2);
    js_obf_len_coeff ~ cauchy(1,2);
    https_coeff  ~ cauchy(-1,2);
    whois_coeff ~ cauchy(-1,2);
    intercept ~ normal(0,10);        
    // Modelling of the label based on bernoulli logistic regression by multiple variable linear regression 
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_list[k, i] ~ bernoulli(inv_logit(intercept + https_coeff * https_list[k, i] + whois_coeff * whois_list[k, i] + js_len_coeff * js_len_list[k, i] + js_obf_len_coeff * js_obf_len_list[k, i]));
      }
    }
}

generated quantities {
    array[K, Nmax] real label_train_pred;
    array[K, Mmax] real label_test_pred;
    array[Nmax] real log_likelihood; 
    // Predictions for the training data
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_train_pred[k, i] = bernoulli_rng(inv_logit(intercept + https_coeff * https_list[k, i] + whois_coeff * whois_list[k, i] + js_len_coeff * js_len_list[k, i] + js_obf_len_coeff * js_obf_len_list[k, i]));
      }
    }
    // Predictions for the testing data
    for (k in 1:K){
      for (i in 1:M_list[k]){
        label_test_pred[k, i] = bernoulli_rng(inv_logit(intercept + https_coeff * https_pred_list[k, i] + whois_coeff * whois_pred_list[k, i] + js_len_coeff * js_len_pred_list[k, i] + js_obf_len_coeff * js_obf_len_pred_list[k, i]));
      }
    }
    for (k in 1:K) {
      if (N_list[k] == Nmax){
        for (i in 1:Nmax){
          log_likelihood[i] = bernoulli_lpmf(label_list[k, i] | inv_logit(intercept + https_coeff * https_list[k, i] + whois_coeff * whois_list[k, i] + js_len_coeff * js_len_list[k, i] + js_obf_len_coeff * js_obf_len_list[k, i]));     
        }
      }
    }
}
