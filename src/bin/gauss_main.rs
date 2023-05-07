use ndarray::{Array, Array1};

fn main() {
    let x_values = Array::linspace(-4.0, 4.0, 10);
    let y_values = Array::linspace(-3.0, 2.0, 10);

    let kernel = exponentiated_quadratic(&Scalarray::Multiple(x_values, y_values));

    println!("{:?}", kernel);
}

enum Scalarray {
    Single(f64, f64),
    Multiple(Array1<f64>, Array1<f64>),
}

fn exponentiated_quadratic(x: &Scalarray) -> f64 {
    let mut nugget: f64;
    match x {
        Scalarray::Single(xa, xb) => {
            nugget = -0.5 * (xa - xb).powf(2.0);
        }
        Scalarray::Multiple(xa, xb) => {
            nugget = -0.5 * (xa - xb).dot(&(xa - xb));
        }
    }
    nugget.exp()
}
