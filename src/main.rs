use rand::Rng;
use std::io;
use plotters::prelude::*;

const TRAINING_SET_SIZE: usize = 5;
const TESTING_SET_SIZE: usize = 5;
const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 1;

fn main() {
    loop {
        let mut rng = rand::thread_rng();


        // Initialize weights and biases
        let weights_io: Vec<f64> = (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect(); // 2x2
        let bias_o = (0..2).map(|_| rng.gen_range(-1.0..1.0)).collect(); // 1x2

        // Inputs and target
        let train_array: Vec<Vec<f64>> = [
            [6.2, 0.30].to_vec(),
            [6.7, 0.14].to_vec(),
            [7.6, 0.40].to_vec(),
            [8.9, 0.31].to_vec(),
            [9.1, 0.68].to_vec(),
        ].to_vec();
        let train_target: Vec<f64> = [
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
        ].to_vec();


        for epoch in 0..EPOCHS { // Training for EPOCHS epochs
            for i in 0..train_array.len() {
                let inputs = &train_array[i];
                let target = train_target[i];

                // Forward pass _________________________________________________________
                let final_input: Vec<f64> = mat_vec_mul(&weights_io, &inputs, &bias_o);
                let final_output:Vec<f64> = [softmax(&final_input, 1.0), softmax(&final_input, -1.0)].to_vec();

                // Compute the error
                let cross_entropy = cross_entropy(target, final_output);
                //println!("error on epoch {}: {:.2}", epoch, cross_entropy);

                // // Backward pass ________________________________________________________

                // // Output to hidden
                // let d_weights_ho: Vec<f64> = hidden_output.iter().map(|&x| LEARNING_RATE * error * x).collect();
                // let d_bias_o = LEARNING_RATE * error;

                // // Hidden to input
                // let hidden_errors: Vec<f64> = weights_ho.iter().map(|&x| x * error).collect();
                // let d_weights_ih: Vec<f64> = (0..4).map(|i| LEARNING_RATE * hidden_errors[i/2] * inputs[i%2]).collect();
                // let d_bias_h: Vec<f64> = hidden_errors.iter().map(|&x| LEARNING_RATE * x).collect();

                // // Update weights and biases
                // update_vec(&mut weights_ho, &d_weights_ho);
                // bias_o += d_bias_o;
                // update_vec(&mut weights_ih, &d_weights_ih);
                // update_vec(&mut bias_h, &d_bias_h);
        }
    }

        // Final output after training
        let test_array: Vec<Vec<f64>> = [
            [6.2, 0.30].to_vec(),
            [6.7, 0.14].to_vec(),
            [7.6, 0.40].to_vec(),
            [8.9, 0.31].to_vec(),
            [9.1, 0.68].to_vec(),
        ].to_vec();
        let test_target: Vec<f64> = [
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
        ].to_vec();

        let mut train_total_error: f64 = 0.0;
        let mut test_total_error: f64 = 0.0;

        println!("-----------------Training set----------------");
        for i in 0..train_array.len() {
            let inputs = &train_array[i];
            let target = train_target[i];
            let final_output = feed_forward(&weights_io, &bias_o, &inputs);
            if i < test_target.len() {
                println!("Target: {:.5}, Output: {:.5}", target, final_output);
            }
            train_total_error += (target - final_output).powi(2);
        }

        println!("-----------------Testing set-----------------");
        for i in 0..test_array.len() {
            let inputs = &test_array[i];
            let target = test_target[i];
            let final_output = feed_forward(&weights_io, &bias_o, &inputs);
            println!("Target: {:.5}, Output: {:.5}", target, final_output);
            test_total_error += (target - final_output).powi(2);
        }

        println!("-------------------Info set------------------");
        let train_mse = train_total_error / train_target.len() as f64;
        let test_mse = test_total_error / test_target.len() as f64;
        println!("Training set count: {}", TRAINING_SET_SIZE);
        println!("Testing set count: {}", TESTING_SET_SIZE);
        println!("Learning rate: {}", LEARNING_RATE);
        println!("Epochs: {}", EPOCHS);
        println!("Training MSE: {:.5}", train_mse);
        println!("Testing MSE: {:.5}", test_mse);

        println!("-------------Press Enter to Retry-------------");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        assert_eq!(input.trim(), "");
        print!("{}[2J", 27 as char);
    }
}

fn mat_vec_mul(matrix: &Vec<f64>, vector: &Vec<f64>, bias: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![0.0; bias.len()];
    for i in 0..result.len() {
        result[i] = matrix[i*2] * vector[0] + matrix[i*2 + 1] * vector[1] + bias[i];
    }
    result
}

// fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
//     vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x * y).sum()
// }

fn softmax(vec1: &Vec<f64>, x: f64) -> f64 {
    let sum: f64 = vec1.iter().map(|&x| x.exp()).sum();
    x.exp() / sum
}

fn cross_entropy(target: f64, output: Vec<f64>) -> f64 {
    let mut result = 0.0;
    for i in 0..output.len() {
        result += target * output[i].ln();
    }
    -result
}

// fn update_vec(vec: &mut Vec<f64>, delta: &Vec<f64>) {
//     for (v, d) in vec.iter_mut().zip(delta.iter()) {
//         *v += *d;
//     }
// }

fn feed_forward(weights_io: &Vec<f64>, bias_o: &Vec<f64>, inputs: &Vec<f64>) -> f64 {
    let final_input: Vec<f64> = mat_vec_mul(&weights_io, &inputs, &bias_o);
    let final_output = softmax(&final_input, 1.0).max(softmax(&final_input, -1.0));
    final_output
}