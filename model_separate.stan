data {
    int N; // number of URLs
    int<lower=1,upper=10> K; // Number of countries
    vector[N] js_len;
    vector[N] js_obf_len;
    int safety[N];
    array[N] vector[K] y;// label (0 = benign, 1 = malicious)

}

parameters {
    real alpha;
    vector[4] beta_safety;
    real beta_js_len;
    real beta_js_obf_len;
}

model {
    alpha ~ normal(0, 10);
    beta_safety ~ normal(0, 10);
    beta_js_len ~ normal(0, 10);
    beta_js_obf_len ~ normal(0, 10);

    for (i in 1:N) {
        y[i] ~ bernoulli_logit(alpha + beta_safety[safety[i]] + beta_js_len * js_len[i] + beta_js_obf_len * js_obf_len[i]);
    }
}
