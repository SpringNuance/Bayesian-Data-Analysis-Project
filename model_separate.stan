data {
  int<lower=1> N; // Number of URLs of a country (training)
  int<lower=1> M; // Number of URLs of a country (testing)
  // The features
  array[N] real js_len;
  array[N] real js_obf_len;
  // safety is categorical, following the categorical distribution
  array[N] int<lower=0,upper=4> safety;

  // predicting features
  array[M] real js_len_pred;
  array[M] real js_obf_len_pred;
  array[M] int<lower=0,upper=4> safety_pred;
  
  // label for each URL: benign(0) or malicious(1)
  array[N] int<lower=0,upper=1> label; 
}

parameters {
  // simplex requires that the sum of its vectors elements are equal to 1. 
  simplex[4] probs;
  // degree of freedom for the inverse chi square
  real nu_js_len;
  real nu_js_obf_len;
  real js_len_coeff;
  real js_obf_len_coeff;
  real safety_coeff;
  real intercept;
}

model {
    // prior for safety
    for (i in 1:4){
      probs[i] ~ normal(0.5, 0.2);
    }
    // likelihood for safety
    for (i in 1:N) {
      safety[i] ~ categorical(probs);
    }
    // prior for js_len
    nu_js_len ~ normal(10, 5);
    // likelihood for js_len
    js_len ~ inv_chi_square(nu_js_len);
    // prior for js_obf_len
    nu_js_obf_len ~ normal(15, 7);
    // likelihood for js_obf_len
    js_obf_len ~ inv_chi_square(nu_js_obf_len);
    
    js_len_coeff ~ normal(0,10);
    js_obf_len_coeff ~ normal(0,10);
    safety_coeff ~ normal(0, 10);
    intercept ~ normal(0, 10);
 
    // Modelling of the label based on bernoulli logistic regression by the multiple linear regressions of the variable
    for (i in 1:N){
      label[i] ~ bernoulli(inv_logit(intercept + safety_coeff * safety[i] + js_len_coeff * js_len[i] + js_obf_len_coeff * js_obf_len[i]));
    }
    
}

//generated quantities {
//    vector[N] y_pred;
//    for (i in 1:N) {
//        y_pred[i] = normal_rng(alpha * x_pred[i] + beta, 1);
//    }
