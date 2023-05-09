use kernel_regression::gp::{Kernel, MultiNormal};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

const N: usize = 10;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(25);
    let x_values = Array::linspace(-2.0, 2.0, N);

    let uniform = Uniform::new(-2.0, 2.0);
    let y_values = Array::random_using(3, uniform, &mut rng);

    let sigma_xx: Array2<f64> = Array1::<f64>::exponentiated_quadratic(&x_values, &x_values);
    let sigma_xy: Array2<f64> = Array1::<f64>::exponentiated_quadratic(&x_values, &y_values);
    let mn: Array1<f64> = Array1::<f64>::zeros(N);
}
