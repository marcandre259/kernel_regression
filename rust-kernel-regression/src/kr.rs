use ndarray::prelude::*;
use ndarray_linalg::SVD;
use std::f64::consts::PI;

pub fn gaussian_kernel(h: f64, x_input: ArrayView1<f64>, x_new: f64) -> Array1<f64> {
    // For continuous variables
    let constant: f64 = 1.0 / (2.0 * PI).sqrt();
    let scale: f64 = h.powf(2.0) * 2.0;
    x_input.mapv(|x: f64| constant * (-1.0 * (x - x_new).powf(2.0) / scale).exp())
}

pub fn aitchison_aitken_kernel(h: f64, x_input: ArrayView1<f64>, x_new: f64) -> Array1<f64> {
    // For unordered discrete variables
    let num_levels = x_input.clone();
    let mut num_levels: Vec<i64> = num_levels.to_vec().iter_mut().map(|x| *x as i64).collect();
    num_levels.sort();
    num_levels.dedup();
    let num_levels = num_levels.len() as f64;

    let kernel_values = x_input.mapv(|x| {
        if x == x_new {
            1.0 - h
        } else {
            h / (num_levels - 1.0)
        }
    });
    kernel_values
}

// We compute the generalized kernel weight with a single slice from the data to predict
pub fn gpke(
    bw: &Vec<f64>,
    data: ArrayView2<f64>,
    data_predict: ArrayView1<f64>,
    var_type: Vec<&str>,
) -> Array1<f64> {
    let mut kval = data.to_owned().clone();
    for (i, vtype) in var_type.into_iter().enumerate() {
        let data_arr = kval.slice(s![.., i]);
        let data_new = data_predict.slice(s![i]).into_scalar();
        let h = bw[i];
        let kcolumn = match vtype {
            "c" => gaussian_kernel(h, data_arr, *data_new),
            "u" => aitchison_aitken_kernel(h, data_arr, *data_new),
            _ => {
                // Convoluted way to get an array of ones with the expected dimensionality
                let ones = gaussian_kernel(h, data_arr, *data_new);
                ones.mapv(|_| 1.0)
            }
        };
        kval.slice_mut(s![.., i]).assign(&kcolumn);
    }
    kval.fold_axis(Axis(1), 1.0, |acc, &x| acc * x)
}

pub fn est_loc_constant(
    bw: &Vec<f64>,
    data_endog: ArrayView1<f64>,
    data_exog: ArrayView2<f64>,
    data_predict: ArrayView1<f64>,
    var_type: Vec<&str>,
) -> f64 {
    let kernel_products = gpke(bw, data_exog, data_predict, var_type);
    let loc_num = (data_endog.to_owned() * kernel_products.view()).sum();
    let loc_denom = kernel_products.sum();
    loc_num / loc_denom
}

// For now keep this for compatibility
pub fn loc_constant_fit(
    bw: &Vec<f64>,
    data_endog: ArrayView1<f64>,
    data_exog: ArrayView2<f64>,
    data_predict: ArrayView2<f64>,
    var_type: Vec<&str>,
) -> Vec<f64> {
    let n_predict = data_predict.len_of(Axis(0));
    let mut local_means: Vec<f64> = Vec::with_capacity(n_predict);

    for i in 0..n_predict {
        let slice_predict = data_predict.slice(s![i, ..]);
        let local_mean =
            est_loc_constant(&bw, data_endog, data_exog, slice_predict, var_type.clone());
        local_means.push(local_mean);
    }

    local_means
}

// Implement local linear regression
pub fn est_loc_linear(
    bw: &Vec<f64>,
    data_endog: ArrayView1<f64>,
    data_exog: ArrayView2<f64>,
    data_predict: ArrayView1<f64>,
    var_type: Vec<&str>,
) -> f64 {
    let exog_shape = data_exog.shape();
    let kernel_products = gpke(bw, data_exog, data_predict, var_type) / (exog_shape[0] as f64);
    let data_exog = data_exog.to_owned();

    let data_predict = data_predict.to_owned().insert_axis(Axis(0));

    // We basically take p. 38 from Nonparametric Econometrics: A Primer as a starting point for
    // a block matrix
    let data_predict_broadcast = data_predict.broadcast(exog_shape).unwrap().to_owned();

    let kernel_products_broadcast = kernel_products
        .view()
        .insert_axis(Axis(1))
        .broadcast(exog_shape)
        .unwrap()
        .to_owned();

    let m12 = &data_exog - &data_predict_broadcast;

    let m12_k: Array2<f64> = (m12.clone() * kernel_products_broadcast)
        .to_owned()
        .into_dimensionality::<Ix2>()
        .unwrap();

    let m12_t: Array2<f64> = m12.t().to_owned().into_dimensionality::<Ix2>().unwrap();
    let m22: Array2<f64> = m12_t.dot(&m12_k);
    let m12: Array2<f64> = m12_k.sum_axis(Axis(0)).insert_axis(Axis(0));

    // Defining the block m matrix
    let mut m: Array2<f64> = Array::<f64, _>::zeros((exog_shape[1] + 1, exog_shape[1] + 1))
        .into_dimensionality::<Ix2>()
        .unwrap();

    m.slice_mut(s![0, 0])
        .assign(&kernel_products.sum_axis(Axis(0)));

    m.slice_mut(s![0, 1..])
        .assign(&m12.clone().into_shape((exog_shape[1],)).unwrap());

    m.slice_mut(s![1.., 0])
        .assign(&m12.clone().into_shape((exog_shape[1],)).unwrap());

    m.slice_mut(s![1.., 1..]).assign(&m22);

    let kernel_endog = kernel_products * data_endog;

    let mut v: Array2<f64> = Array::<f64, _>::zeros((exog_shape[1] + 1, 1))
        .into_dimensionality::<Ix2>()
        .unwrap();

    v.slice_mut(s![0, 0])
        .assign(&kernel_endog.sum_axis(Axis(0)));

    let v_assign = ((data_exog - data_predict_broadcast)
        * kernel_endog
            .insert_axis(Axis(1))
            .broadcast(exog_shape)
            .unwrap())
    .sum_axis(Axis(0));

    v.slice_mut(s![1.., 0])
        .assign(&v_assign.into_shape((exog_shape[1],)).unwrap());

    let est = mp_inverse(&m)
        .dot(&v)
        .slice(s![0, 0])
        .to_owned()
        .into_scalar();

    est
}

pub fn fit_predict(
    bw: &Vec<f64>,
    data_endog: ArrayView1<f64>,
    data_exog: ArrayView2<f64>,
    data_predict: ArrayView2<f64>,
    var_type: Vec<&str>,
    reg_type: &str,
) -> Vec<f64> {
    let n_predict = data_predict.len_of(Axis(0));
    let mut local_means: Vec<f64> = Vec::with_capacity(n_predict);

    for i in 0..n_predict {
        let slice_predict = data_predict.slice(s![i, ..]);

        let local_mean = match reg_type {
            "loc_constant" => {
                est_loc_constant(&bw, data_endog, data_exog, slice_predict, var_type.clone())
            }
            "loc_linear" => {
                est_loc_linear(&bw, data_endog, data_exog, slice_predict, var_type.clone())
            }
            _ => panic!("Invalid regression type"),
        };
        local_means.push(local_mean);
    }

    local_means
}

pub fn mp_inverse(m: &Array2<f64>) -> Array2<f64> {
    let m = m.to_owned();

    let svd = m.svd(true, true).unwrap();
    let (u, s, vt) = (svd.0.unwrap(), svd.1, svd.2.unwrap());

    // Check threshold for small singular values
    let threshold = 1e-10;

    let s_inv = Array::from_diag(&s.mapv(|x| if x > threshold { 1.0 / x } else { 0.0 }));

    let m_pinv = vt.t().dot(&s_inv).dot(&u.t());

    m_pinv
}

// Create a KernelReg struct

// Implement a fit method for the KernelReg struct

// Got to add a ridge penalty
// Adapt it to limits of the time-series data
