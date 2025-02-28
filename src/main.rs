use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
//use std::f64::EPSILON;

/// Aplica o algoritmo Improved Asymmetric Least Squares Smoothing (IAsLS)
fn iasls(y: &Array1<f64>, p: f64, lambda: f64, tol: f64, max_iter: usize) -> Array1<f64> {
    let n = y.len();
    let mut z = y.clone();
    let mut w = Array1::<f64>::from_elem(n, 1.0);

    for _ in 0..max_iter {
        // Monta a matriz D para suavização (segunda diferença)
        let mut d = Array2::<f64>::zeros((n - 2, n));
        for i in 0..(n - 2) {
            d[[i, i]] = 1.0;
            d[[i, i + 1]] = -2.0;
            d[[i, i + 2]] = 1.0;
        }

        // Matriz de pesos W
        let w_diag = Array2::<f64>::from_diag(&w);
        
        // Sistema linear para atualização da baseline
        let lhs = &w_diag + &d.t().dot(&d) * lambda;
        let rhs = w_diag.dot(y);
        let new_z = lhs.solve(&rhs).expect("Falha ao resolver sistema linear.");

        // Atualiza os pesos
        let diff = y - &new_z;
        let new_w = diff.mapv(|val| if val < 0.0 { 1.0 - p } else { p });

        // Critério de convergência
        if (&w - &new_w).mapv(|x| x.abs()).sum() / w.sum() < tol {
            break;
        }

        w = new_w;
        z = new_z;
    }

    z
}

fn main() {
    // Simulação de um espectro com baseline
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi / 10.0).sin() + 0.1 * (xi / 5.0).cos() + 0.2 * xi / 100.0).collect();
    let y = Array1::from(y);

    // Parâmetros do IAsLS
    let p = 0.05;         // Peso assimétrico
    let lambda = 1e2;     // Regularização
    let tol = 1e-5;       // Critério de convergência
    let max_iter = 50;    // Número máximo de iterações

    // Aplicação do algoritmo
    let baseline = iasls(&y, p, lambda, tol, max_iter);

    // Exibição dos resultados
    println!("Baseline ajustada: {:?}", baseline);
}

