use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rust_kernel_regression::kr::{gpke, mp_inverse};
use std::f64::consts::PI;

pub fn est_loc_constant(
    bw: &Vec<f64>,
    data_endog: ArrayView1<f64>,
    data_exog: ArrayView2<f64>,
    data_predict: ArrayView1<f64>,
    var_type: Vec<&str>,
) -> PyResult<f64> {
    let kernel_products = gpke(bw, data_exog, data_predict, var_type);
    let loc_num = (data_endog.to_owned() * kernel_products.view()).sum();
    let loc_denom = kernel_products.sum();
    Ok(loc_num / loc_denom)
}

#[pyfunction]
pub unsafe fn loc_constant_fit(
    bw: &PyList,
    data_endog: &PyArray1<f64>,
    data_exog: &PyArray2<f64>,
    data_predict: &PyArray2<f64>,
    var_type: &PyList,
) -> PyResult<Vec<f64>> {
    let bw: Vec<f64> = bw.extract()?;
    let data_endog = data_endog.as_array();
    let data_exog = data_exog.as_array();
    let data_predict = data_predict.as_array();
    let var_type: Vec<&str> = var_type.extract()?;

    let n_predict = data_predict.len_of(Axis(0));
    let mut local_means: Vec<f64> = Vec::with_capacity(n_predict);

    for i in 0..n_predict {
        let slice_predict = data_predict.slice(s![i, ..]);
        let local_mean =
            est_loc_constant(&bw, data_endog, data_exog, slice_predict, var_type.clone())?;
        local_means.push(local_mean);
    }

    Ok(local_means)
}

pub fn est_loc_linear(
    bw: &Vec<f64>,
    data_endog: ArrayView1<f64>,
    data_exog: ArrayView2<f64>,
    data_predict: ArrayView1<f64>,
    var_type: Vec<&str>,
) -> PyResult<f64> {
    let exog_shape = data_exog.shape();
    let kernel_products = gpke(bw, data_exog, data_predict, var_type);
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

    let m12 = data_exog - data_predict_broadcast;

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

    v.slice_mut(s![1.., 0])
        .assign(&m12.into_shape((exog_shape[1],)).unwrap());

    let est = mp_inverse(&m)
        .dot(&v)
        .slice(s![0, 0])
        .to_owned()
        .into_scalar();

    Ok(est)
}

// Keep the loc_constant_fit for legacy reasons
#[pyfunction]
pub unsafe fn fit_predict(
    _py: Python,
    bw: &PyList,
    data_endog: &PyArray1<f64>,
    data_exog: &PyArray2<f64>,
    data_predict: &PyArray2<f64>,
    var_type: &PyList,
    reg_type: Option<&PyString>,
) -> PyResult<Vec<f64>> {
    let bw: Vec<f64> = bw.extract()?;
    let data_endog = data_endog.as_array();
    let data_exog = data_exog.as_array();
    let data_predict = data_predict.as_array();
    let var_type: Vec<&str> = var_type.extract()?;
    let reg_type = reg_type.unwrap_or(PyString::new(_py, "loc_constant"));

    let n_predict = data_predict.len_of(Axis(0));
    let mut local_means: Vec<f64> = Vec::with_capacity(n_predict);

    for i in 0..n_predict {
        let slice_predict = data_predict.slice(s![i, ..]);

        let local_mean = match reg_type.to_str()? {
            "loc_constant" => {
                est_loc_constant(&bw, data_endog, data_exog, slice_predict, var_type.clone())?
            }
            "loc_linear" => {
                est_loc_linear(&bw, data_endog, data_exog, slice_predict, var_type.clone())?
            }
            _ => panic!("Invalid regression type"),
        };
        local_means.push(local_mean);
    }

    Ok(local_means)
}

#[pymodule]
fn py_kernel_regression(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(loc_constant_fit, m)?)?;
    Ok(())
}
