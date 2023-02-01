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
  array[K] real<lower=0, upper=1> theta_js_len; // probability for js_len
  array[K] real<lower=0, upper=1> theta_js_obf_len; // probability for js_obf_len
  array[K] real<lower=0, upper=1> theta_https; // probability for https
  array[K] real<lower=0, upper=1> theta_whois; // probability for whois
  array[K] real js_len_coeff; // Slope coefficient for js_len
  array[K] real js_obf_len_coeff; // Slope coefficient for js_obf_len
  array[K] real https_coeff; // Slope coefficient for https_coeff
  array[K] real whois_coeff; // Slope coefficient for whois_coeff
  array[K] real intercept; // Intercept coefficient
}

model {
    // Prior probabilities of the features
    for (k in 1:K){
        theta_js_len[k] ~ beta(1,1);
        theta_js_obf_len[k] ~ beta(1,1);
        theta_https[k] ~ beta(1,1);
        theta_whois[k] ~ beta(1,1);
    }
    // likelihood for the features
    for (k in 1:K){
        js_len_list[k, 1:N_list[k]] ~ bernoulli(theta_js_len[K]);
        js_obf_len_list[k, 1:N_list[k]] ~ bernoulli(theta_js_obf_len[K]);
        https_list[k, 1:N_list[k]] ~ bernoulli(theta_https[K]);
        whois_list[k, 1:N_list[k]] ~ bernoulli(theta_whois[K]);     
    }
    // priors of the coefficients
    for (k in 1:K){
       js_len_coeff[k] ~ normal(1,10);
       js_obf_len_coeff[k] ~ normal(1,10);
       https_coeff[k]  ~ normal(-1,10);
       whois_coeff[k] ~ normal(-1,10);
       intercept[k] ~ normal(0,20);        
    }
    // Modelling of the label based on bernoulli logistic regression by multiple variable linear regression 
    for (k in 1:K){
      for (i in 1:N_list[k]){
        label_list[k, i] ~ bernoulli(inv_logit(intercept[k] + https_coeff[k] * https_list[k, i] + whois_coeff[k] * whois_list[k, i] + js_len_coeff[k] * js_len_list[k, i] + js_obf_len_coeff[k] * js_obf_len_list[k, i]));
      }
    }
}

generated quantities {
    array[Nmax] real log_likelihood; 
    for (k in 1:K) {
      if (N_list[k] == Nmax){
        for (i in 1:Nmax){
          log_likelihood[i] = bernoulli_lpmf(label_list[k, i] | inv_logit(intercept[k] + https_coeff[k] * https_list[k, i] + whois_coeff[k] * whois_list[k, i] + js_len_coeff[k] * js_len_list[k, i] + js_obf_len_coeff[k] * js_obf_len_list[k, i]));     
        }
      }
    }
}
