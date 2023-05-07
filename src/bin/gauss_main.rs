use ndarray::{Array, Array1, Array2};

fn main() {
    let x_values = Array::linspace(-0.20, 0.20, 3);
    let y_values = Array::linspace(-3.0, 2.0, 3);

    let sigma: Array2<f64> = Array1::<f64>::exponentiated_quadratic(&x_values, &x_values);

    println!("{:.5?}", sigma);
}

trait Kernel<T, D> {
    fn exponentiated_quadratic(x: T, y: T) -> D;
}

impl Kernel<&Array1<f64>, Array2<f64>> for Array1<f64> {
    fn exponentiated_quadratic(x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
        let mut outer: Array2<f64> = Array2::zeros((y.len(), x.len()));
        for (i, a) in x.iter().enumerate() {
            for (j, b) in y.iter().enumerate() {
                outer[(j, i)] = (-0.5 * (a - b).powf(2.0)).exp();
            }
        }
        outer
    }
}
