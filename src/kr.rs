use ndarray::prelude::*;
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
