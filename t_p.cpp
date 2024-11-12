#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Constants
const double SAMPLING_RATE = 50.0;
const double DT = 1.0 / SAMPLING_RATE;
const int WINDOW_LENGTH = 5;
const int POLYORDER = 2;
const int LAG_ORDER = 1;
const int BUFFER_SIZE = 20;

// Function to calculate Savitzky-Golay smoothing
vector<double> savgol_filter(const vector<double>& data) {
    vector<double> smoothed(data.size(), 0.0);
    int half_window = WINDOW_LENGTH / 2;

    for (int i = half_window; i < data.size() - half_window; ++i) {
        for (int j = -half_window; j <= half_window; ++j) {
            smoothed[i] += data[i + j];
        }
        smoothed[i] /= WINDOW_LENGTH;
    }

    return smoothed;
}

// Function to predict the next torque using a simple VAR model
double predict_next_torque(const vector<double>& torque_series, const vector<double>& angle_series, const vector<double>& velocity_series) {
    int series_length = torque_series.size();
    if (series_length < LAG_ORDER + 1) {
        return torque_series.back();
    }

    VectorXd Y(series_length - LAG_ORDER);
    MatrixXd X(series_length - LAG_ORDER, 4);

    for (int i = LAG_ORDER; i < series_length; ++i) {
        Y(i - LAG_ORDER) = torque_series[i];
        X(i - LAG_ORDER, 0) = 1.0;  // Intercept term
        X(i - LAG_ORDER, 1) = torque_series[i - LAG_ORDER];
        X(i - LAG_ORDER, 2) = angle_series[i - LAG_ORDER];
        X(i - LAG_ORDER, 3) = velocity_series[i - LAG_ORDER];
    }

    VectorXd coeffs = X.colPivHouseholderQr().solve(Y);
    VectorXd last_X(4);
    last_X << 1.0, torque_series.back(), angle_series.back(), velocity_series.back();

    return last_X.dot(coeffs);
}

int main() {
    // Load data from file
    ifstream file("240915_0907");
    if (!file.is_open()) {
        cerr << "Error: Could not open file." << endl;
        return 1;
    }

    string line;
    vector<double> time, torque, angle, velocity;
    getline(file, line); // Skip header

    while (getline(file, line)) {
        stringstream ss(line);
        double t, t_a, elbow_angle, elbow_velocity;
        ss >> t >> t_a >> elbow_angle >> elbow_velocity;

        if (t >= 0.32) {
            time.push_back(t);
            torque.push_back(t_a);
            angle.push_back(elbow_angle);
            velocity.push_back(elbow_velocity);
        }
    }

    // Buffers and series
    deque<double> torque_buffer, angle_buffer, velocity_buffer;
    vector<double> torque_series, angle_series, velocity_series, predictions;

    // Timing the execution
    auto start_time = chrono::high_resolution_clock::now();

    // Loop over data for prediction
    for (int i = 0; i < torque.size(); ++i) {
        double current_torque = torque[i];
        double current_angle = angle[i];
        double current_velocity = velocity[i];

        torque_buffer.push_back(current_torque);
        angle_buffer.push_back(current_angle);
        velocity_buffer.push_back(current_velocity);

        if (torque_buffer.size() > BUFFER_SIZE) {
            torque_buffer.pop_front();
            angle_buffer.pop_front();
            velocity_buffer.pop_front();
        }

        vector<double> torque_smoothed = savgol_filter(vector<double>(torque_buffer.begin(), torque_buffer.end()));
        vector<double> angle_smoothed = savgol_filter(vector<double>(angle_buffer.begin(), angle_buffer.end()));
        vector<double> velocity_smoothed = savgol_filter(vector<double>(velocity_buffer.begin(), velocity_buffer.end()));

        torque_series.push_back(torque_smoothed.back());
        angle_series.push_back(angle_smoothed.back());
        velocity_series.push_back(velocity_smoothed.back());

        double next_torque_pred = predict_next_torque(torque_series, angle_series, velocity_series);
        predictions.push_back(next_torque_pred);
    }

    // Calculate percentage errors
    int high_error_count = 0;
    for (int i = 0; i < predictions.size(); ++i) {
        double percentage_error = abs((torque[i] - predictions[i]) / torque[i]) * 100.0;
        if (percentage_error > 20.0) {
            high_error_count++;
        }
    }

    // Calculate total execution time
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> execution_time = end_time - start_time;

    // Output results
    cout << "Total number of timesteps: " << torque.size() << endl;
    cout << "Number of timesteps with percentage error > 20%: " << high_error_count << endl;
    cout << "Time taken to execute the code: " << execution_time.count() << " seconds" << endl;

    return 0;
}
