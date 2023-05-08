use ndarray::{Array, Array1, Array2};
use ndarray_linalg::cholesky::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand_chacha::ChaCha20Rng;

const N: usize = 40;

fn main() {
    let x_values = Array::linspace(-2.0, 2.0, N);

    let sigma: Array2<f64> = Array1::<f64>::exponentiated_quadratic(&x_values, &x_values);
    let mn: Array1<f64> = Array1::<f64>::zeros(N);

    MultiNormal::new();

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

struct MultiNormal {
    mn: Array1<f64>,
    sigma: Array2<f64>,
}

impl MultiNormal {
    fn new(mn: Array1<f64>, sigma: Array2<f64>) {}

    fn sample<E>(&self, n: usize) -> Array2<f64> {
        let mut rng = ChaCha20Rng::seed_from_u64(25);

        let normal = Normal::new(0.0, 1.0).unwrap();

        let samples = Array::random_using((n, n), normal, &mut rng);

        let chol = &self.sigma.cholesky(UPLO::Lower).unwrap();

        &chol.dot(&samples).t() + &self.mn.broadcast(samples.dim()).unwrap()
    }
}
