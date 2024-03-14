use ndarray::array;
use rust_kernel_regression::kr::fit_predict;

fn main() {
    let bw = vec![1.0, 0.2];
    let x_train = array![[1.0, 1.0], [3.2, 3.0], [2.5, 2.0], [1.2, 2.0], [4.3, 1.0]];

    let x_new = array![[1.0, 2.0], [2.2, 3.0], [2.6, 2.0]];

    let var_type = vec!["c", "u"];

    let y_train = array![9., 9., 10., 3., 4.];

    // let output = gpke(bw, x_train, x_new, var_type);

    // let output = est_loc_constant(&bw, y_train.view(), x_train.view(), x_new.view(), var_type);

    // let output = loc_constant_fit(&bw, y_train.view(), x_train.view(), x_new.view(), var_type);
    //

    // let output = mp_inverse(&x_train.dot(&x_train.t()));

    // println!("{:#?}", output);

    //    let output = est_loc_linear(
    //        &bw,
    //        y_train.view(),
    //        x_train.view(),
    //        x_new.view().slice(s![0, ..]),
    //        var_type,
    //    );
    //
    let output = fit_predict(
        &bw,
        y_train.view(),
        x_train.view(),
        x_new.view(),
        var_type,
        "loc_linear",
    );

    println!("{:#?}", output);
}
