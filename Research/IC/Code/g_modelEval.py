import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn

from g_modelEval_aux import create_evaluation_csvs

# Configure PyTorch to use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {device}")
else:
    print("No GPU found. Running on CPU.")
    print(f"Device: {device}")

# ==================== GLOBAL CONFIGURATION ====================
# ARIMA Parameters
ARIMA_ORDER = (1, 1, 1)  # (p, d, q)

# LSTM Parameters
LSTM_EPOCHS = 100
LSTM_NEURONS = 100
LSTM_BATCH_SIZE = 1  # Batch size for training
LSTM_EARLY_STOPPING_PATIENCE = 10  
LSTM_LEARNING_RATE = 0.001

# Data Split
TRAIN_RATIO = 0.7  # 70% training, 30% testing
# ============================================================

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "optimized_results")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "eval_results")
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)
    
RUN_TITLE = "unknown_run"

def load_evaluation_data(rank):
    global RUN_TITLE
    
    baseline_path = os.path.join(OUTPUT_FOLDER, 'sentiment_baseline.csv')
    interpolated_path = os.path.join(OUTPUT_FOLDER, 'sentiment_not_corrected.csv')
    optimized_path = os.path.join(OUTPUT_FOLDER, 'sentiment_corrected.csv')
    
    # Check if files exist, if not create them
    print("Creating evaluation datasets...")
    _, _, _, run_title = create_evaluation_csvs(rank)
    RUN_TITLE = str(rank + 1) + "--" + run_title
    
    # Create directory for this run, return if exist
    if not os.path.exists(os.path.join(PLOTS_FOLDER, RUN_TITLE)):
        os.makedirs(os.path.join(PLOTS_FOLDER, RUN_TITLE))
    else:
        print(f"Run '{RUN_TITLE}' already exists. Skipping.")
        return None
    
    baseline = pd.read_csv(baseline_path, sep='|')
    interpolated = pd.read_csv(interpolated_path, sep='|')
    optimized = pd.read_csv(optimized_path, sep='|')
    
    # Convert dates
    for df in [baseline, interpolated, optimized]:
        df['date'] = pd.to_datetime(df['date'])
    
    return {
        'baseline': baseline,
        'interpolated': interpolated,
        'optimized': optimized
    }

def split_data(df, train_ratio=None):
    if train_ratio is None:
        train_ratio = TRAIN_RATIO
    size = int(len(df) * train_ratio)
    train = df.iloc[:size].copy()
    test = df.iloc[size:].copy()
    return train, test

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# ==================== ARIMA Models ====================

def arima_with_exog(train, test, order=None):
    if order is None:
        order = ARIMA_ORDER
    
    history_inflation = list(train['inflation'].values)
    history_sentiment = list(train['sentiment'].values)
    predictions = []
    errors = []
    
    for t in range(len(test)):
        try:
            # Prepare exogenous variable
            exog_train = np.array(history_sentiment).reshape(-1, 1)
            exog_test = np.array([test['sentiment'].iloc[t]]).reshape(-1, 1)
            
            model = ARIMA(history_inflation, exog=exog_train, order=order)
            model_fit = model.fit()
            
            # Forecast
            hat = model_fit.forecast(steps=1, exog=exog_test)[0]
            predictions.append(hat)
            
            # Calculate error
            observed = test['inflation'].iloc[t]
            error = abs(observed - hat)
            errors.append(error)
            
            # Update history
            history_inflation.append(observed)
            history_sentiment.append(test['sentiment'].iloc[t])
        except Exception as e:
            print(f"Error at step {t}: {e}")
            predictions.append(history_inflation[-1])
            errors.append(0)
    
    return predictions, errors

# ==================== LSTM Models ====================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def prepare_lstm_data(data):
    # Extract features: inflation and sentiment
    inflation = data['inflation'].values
    sentiment = data['sentiment'].values
    
    # Create sequences: use previous values to predict next inflation
    X, y = [], []
    for i in range(len(inflation) - 1):
        # Input: [inflation[i], sentiment[i]]
        X.append([inflation[i], sentiment[i]])
        # Output: inflation[i+1]
        y.append(inflation[i + 1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    y = y.reshape((y.shape[0], 1))
    
    return X, y

def fit_lstm(X_train, y_train, epochs=None, neurons=None, batch_size=None, patience=None, learning_rate=None):
    if epochs is None:
        epochs = LSTM_EPOCHS
    if neurons is None:
        neurons = LSTM_NEURONS
    if batch_size is None:
        batch_size = LSTM_BATCH_SIZE
    if patience is None:
        patience = LSTM_EARLY_STOPPING_PATIENCE
    if learning_rate is None:
        learning_rate = LSTM_LEARNING_RATE
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    # Create model
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=neurons).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    return model

def forecast_lstm(model, X):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()[0, 0]
    return pred

def lstm_model(train, test, epochs=None, neurons=None, batch_size=None):
    if epochs is None:
        epochs = LSTM_EPOCHS
    if neurons is None:
        neurons = LSTM_NEURONS
    if batch_size is None:
        batch_size = LSTM_BATCH_SIZE
    
    # Prepare training data
    X_train, y_train = prepare_lstm_data(train)
    
    # Fit model
    model = fit_lstm(X_train, y_train, epochs=epochs, neurons=neurons, batch_size=batch_size)
    
    # Prepare test data
    X_test, y_test = prepare_lstm_data(test)
    
    # Make predictions
    predictions = []
    errors = []
    
    for i in range(len(X_test)):
        # Forecast
        pred = forecast_lstm(model, X_test[i:i+1])
        predictions.append(pred)
        
        # Calculate error
        observed = y_test[i, 0]
        error = abs(observed - pred)
        errors.append(error)
    
    return predictions, errors, 1

# ==================== Evaluation and Plotting ====================

def evaluate_model(method_name, dataset_name, predictions, test_data, lookback=0):
    """Evaluate model performance"""
    # Adjust test_data for lookback if necessary
    true_values = test_data['inflation'].iloc[lookback:].values
    
    # Ensure predictions and true_values have the same length
    min_len = min(len(true_values), len(predictions))
    true_values = true_values[:min_len]
    predictions = predictions[:min_len]
    
    metrics = calculate_metrics(true_values, predictions)
    
    print(f"\n{method_name} - {dataset_name}:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    return metrics

def plot_predictions(method_name, dataset_name, test_data, predictions, errors, save_path, lookback=0):
    """Plot predictions vs actual values and errors"""
    # Dataset name mapping
    dataset_labels = {
        'baseline': 'Baseline',
        'interpolated': 'Without Correction',
        'optimized': 'With Correction'
    }
    dataset_label = dataset_labels.get(dataset_name, dataset_name)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Adjust data for lookback
    dates = test_data['date'].iloc[lookback:].values
    actual = test_data['inflation'].iloc[lookback:].values
    
    # Ensure all arrays have the same length
    min_len = min(len(dates), len(actual), len(predictions), len(errors))
    dates = dates[:min_len]
    actual = actual[:min_len]
    predictions = predictions[:min_len]
    errors = errors[:min_len]
    
    # Plot 1: Predictions vs Actual
    ax1.plot(dates, actual, 'b-', label='Actual Inflation', linewidth=2)
    ax1.plot(dates, predictions, 'r--', label='Predicted Inflation', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Inflation (%)', fontsize=12)
    ax1.set_title(f'{method_name} - {dataset_label}\nPredictions vs Actual Values', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Prediction Errors
    ax2.plot(dates, errors, 'g-', label='Absolute Error', linewidth=1.5)
    ax2.axhline(y=np.mean(errors), color='orange', linestyle='--', label=f'Mean Error: {np.mean(errors):.4f}', linewidth=2)
    ax2.fill_between(dates, 0, errors, alpha=0.3, color='green')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {os.path.basename(save_path)}")

def plot_comparison(results_df, save_path):
    """Plot comparison of all methods and datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['RMSE', 'MAE', 'R2', 'MSE']
    colors = {'baseline': '#FF6B6B', 'interpolated': '#4ECDC4', 'optimized': '#45B7D1'}
    dataset_labels = {'baseline': 'Baseline', 'interpolated': 'Without Correction', 'optimized': 'With Correction'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Group by method
        methods = results_df['method'].unique()
        x = np.arange(len(methods))
        width = 0.25
        
        for i, dataset in enumerate(['baseline', 'interpolated', 'optimized']):
            values = []
            for method in methods:
                value = results_df[(results_df['method'] == method) & 
                                 (results_df['dataset'] == dataset)][metric].values[0]
                values.append(value)
            
            ax.bar(x + i*width, values, width, label=dataset_labels[dataset], 
                   color=colors[dataset], alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved: {os.path.basename(save_path)}")

def run_full_evaluation():
    print("="*70)
    print("INFLATION PREDICTION MODEL EVALUATION")
    print("="*70)
    
    # Get rank amount from INPUT_FOLDERs all_optimization_results.csv
    with open(os.path.join(INPUT_FOLDER, 'all_optimization_results.csv'), 'r') as f:
        lines = f.readlines()
    
    amount_of_ranks = len(lines) - 1
    
    for rank in range(amount_of_ranks):
    
        # Rank of the optimization result to evaluate
        print(f"\nEvaluating optimization result rank: {rank}")
    
        # Load datasets
        print("\nLoading evaluation datasets...")
        datasets = load_evaluation_data(rank)
        
        if datasets is None:
            continue  # Skip if run already exists
        
        results = []
        
        # Define models to test
        models = [
            ('ARIMA', arima_with_exog),
            ('LSTM', lstm_model),
        ]
        
        # Test each model on each dataset
        for dataset_name, dataset in datasets.items():
            print(f"\n{'='*70}")
            print(f"Testing on: {dataset_name.upper()} dataset ({len(dataset)} samples)")
            print(f"{'='*70}")
            
            # Split data
            train, test = split_data(dataset)
            print(f"Train size: {len(train)}, Test size: {len(test)}")
            
            for model_name, model_func in models:
                print(f"\n--- {model_name} ---")
                
                try:
                    # Run model
                    lookback = 0
                    if 'ARIMA' in model_name:
                        predictions, errors = model_func(train, test)
                    else:  # LSTM
                        predictions, errors, lookback = model_func(train, test, epochs=LSTM_EPOCHS, neurons=LSTM_NEURONS, batch_size=LSTM_BATCH_SIZE)
                    
                    # Evaluate
                    metrics = evaluate_model(model_name, dataset_name, predictions, test, lookback)
                    
                    # Plot
                    safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
                    plot_path = os.path.join(PLOTS_FOLDER, RUN_TITLE,
                                            f"{safe_model_name}_{dataset_name}.png")
                    plot_predictions(model_name, dataset_name, test, predictions, errors, plot_path, lookback)
                    
                    # Store results
                    results.append({
                        'method': model_name,
                        'dataset': dataset_name,
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'],
                        'R2': metrics['R2'],
                        'MSE': metrics['MSE']
                    })
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        results_path = os.path.join(OUTPUT_FOLDER, PLOTS_FOLDER, RUN_TITLE, 'model_comparison_results.csv')
        results_df.to_csv(results_path, sep='|', index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {results_path}")
        
        # Create comparison plot
        comparison_path = os.path.join(OUTPUT_FOLDER, PLOTS_FOLDER, RUN_TITLE, 'overall_comparison.png')
        plot_comparison(results_df, comparison_path)
        
        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY RESULTS")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))
        
        # Find best model
        print(f"\n{'='*70}")
        print("BEST MODELS BY METRIC")
        print(f"{'='*70}")
        
        for metric in ['RMSE', 'MAE']:
            best = results_df.loc[results_df[metric].idxmin()]
            print(f"\nBest {metric}: {best['method']} on {best['dataset']} dataset")
            print(f"  {metric}: {best[metric]:.4f}")
        
        best_r2 = results_df.loc[results_df['R2'].idxmax()]
        print(f"\nBest R²: {best_r2['method']} on {best_r2['dataset']} dataset")
        print(f"  R²: {best_r2['R2']:.4f}")
        
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*70}")

if __name__ == "__main__":
    run_full_evaluation()
