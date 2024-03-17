use ndarray::array;
use rust_kernel_regression::kr::{rmse, KernelReg};

fn main() {
    let bw = vec![1.0, 0.2];
    let x_train = array![[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [1.0, 2.0], [2.0, 2.0]];

    let x_new = array![[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];

    let var_type = vec!["c", "u"];

    let y_train = array![1., 2., 3., 2., 4.];

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
    //
    //

    // let data_exog_inot = x_train.view().exclude(1);

    // println!("{:#?}", data_exog_inot);
    //
    //
    // println!("Original data: {:#?}", &y_train);

    // let data_endog_inot = y_train.view().exclude(0);

    // println!("{:#?}", data_endog_inot);

    let kernel_reg = KernelReg::new(
        vec![1.0, 0.9],
        vec![String::from("c"), String::from("u")],
        String::from("loc_constant"),
    );

    let output = kernel_reg.leave_one_out(y_train.view(), x_train.view(), rmse);

    //    let output = mse(y_train.view(), y_train.view());

    println!("{:#?}", output);
}
