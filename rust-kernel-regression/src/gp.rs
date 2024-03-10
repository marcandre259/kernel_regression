use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

pub trait Kernel<T, D> {
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

pub struct MultiNormal {
    mn: Array1<f64>,
    sigma: Array2<f64>,
}

impl MultiNormal {
    pub fn new(mn: Array1<f64>, sigma: Array2<f64>) -> MultiNormal {
        MultiNormal {
            mn: mn,
            sigma: sigma,
        }
    }

    pub fn sample(&self, n: usize) -> Array2<f64> {
        let mut rng = ChaCha20Rng::seed_from_u64(25);

        let (m, _) = &self.sigma.dim();

        let normal = Normal::new(0.0, 1.0).unwrap();

        let samples = Array::random_using((*m, n), normal, &mut rng);

        let scale = self._cholesky().dot(&samples);

        let shift = self._broadcast_mean(n);

        scale + shift
    }

    fn _cholesky(&self) -> Array2<f64> {
        let sigma = &self.sigma;
        let mut L = Array2::<f64>::zeros(sigma.dim());
        let (n, _) = self.sigma.dim();

        for i in 0..n {
            for j in 0..=i {
                let mut summation: f64 = 0.0;
                for k in 0..j {
                    summation += L[(i, k)] * L[(j, k)];
                }
                if i == j {
                    L[(i, j)] = (sigma[(i, i)] - summation).sqrt();
                } else {
                    L[(i, j)] = 1.0 / L[(j, j)] * (sigma[(i, j)] - summation);
                }
            }
        }

        L
    }

    fn _broadcast_mean(&self, m: usize) -> Array2<f64> {
        let (n, _) = self.sigma.dim();
        let mut b_mean: Array2<f64> = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                b_mean[(i, j)] = self.mn[i];
            }
        }
        b_mean
    }
}
