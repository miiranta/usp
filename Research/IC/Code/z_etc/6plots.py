import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = SCRIPT_FOLDER

def get_hardcoded_data():
    """Return hardcoded RMSE values and model info from info_normalized.txt"""
    data = {
        'OPEN': {
            'ARIMA': {
                'BASELINE': (0.3667, ''),
                'BEST_NO_CORRECTION_SINGLE': (0.3595, 'grok-4-fast\neq3'),
                'BEST_WITH_CORRECTION_SINGLE': (0.3568, 'grok-4-fast\neq3'),
                'BEST_NO_CORRECTION_MULTIPLE': (0.3587, 'llama-4-maverick\ngrok-4-fast\neq4'),
                'BEST_WITH_CORRECTION_MULTIPLE': (0.3563, 'llama-4-maverick\ngrok-4-fast\neq4')
            },
            'LSTM': {
                'BASELINE': (0.3676, ''),
                'BEST_NO_CORRECTION_SINGLE': (0.3702, 'gpt-5\neq4'),
                'BEST_WITH_CORRECTION_SINGLE': (0.3586, 'gpt-5\neq4'),
                'BEST_NO_CORRECTION_MULTIPLE': (0.3704, 'gemma-3-27b-it\ngpt-5\ngrok-4-fast\neq4'),
                'BEST_WITH_CORRECTION_MULTIPLE': (0.3568, 'gemma-3-27b-it\ngpt-5\ngrok-4-fast\neq4')
            }
        },
        'SPECIALIST': {
            'ARIMA': {
                'BASELINE': (0.3667, ''),
                'BEST_NO_CORRECTION_SINGLE': (0.3582, 'grok-4-fast\neq3'),
                'BEST_WITH_CORRECTION_SINGLE': (0.3565, 'grok-4-fast\neq3'),
                'BEST_NO_CORRECTION_MULTIPLE': (0.3655, 'llama-4-maverick\ngrok-4-fast\neq1'),
                'BEST_WITH_CORRECTION_MULTIPLE': (0.3570, 'llama-4-maverick\ngrok-4-fast\neq1')
            },
            'LSTM': {
                'BASELINE': (0.3676, ''),
                'BEST_NO_CORRECTION_SINGLE': (0.3591, 'claude-sonnet-4\neq4'),
                'BEST_WITH_CORRECTION_SINGLE': (0.3704, 'claude-sonnet-4\neq4'),
                'BEST_NO_CORRECTION_MULTIPLE': (0.3603, 'deepseek-chat-v3.1\ngrok-4-fast\neq4'),
                'BEST_WITH_CORRECTION_MULTIPLE': (0.3554, 'deepseek-chat-v3.1\ngrok-4-fast\neq4')
            }
        },
        'CONSOLIDATED': {
            'ARIMA': {
                'BASELINE': (0.3667, ''),
                'BEST_NO_CORRECTION_SINGLE': (0.3616, 'grok-4-fast\neq3'),
                'BEST_WITH_CORRECTION_SINGLE': (0.3594, 'grok-4-fast\neq3'),
                'BEST_NO_CORRECTION_MULTIPLE': (0.3616, 'gpt-5\ngrok-4-fast\neq3'),
                'BEST_WITH_CORRECTION_MULTIPLE': (0.3603, 'gpt-5\ngrok-4-fast\neq3')
            },
            'LSTM': {
                'BASELINE': (0.3676, ''),
                'BEST_NO_CORRECTION_SINGLE': (0.3685, 'gemma-3-27b-it\neq4'),
                'BEST_WITH_CORRECTION_SINGLE': (0.3596, 'gemma-3-27b-it\neq4'),
                'BEST_NO_CORRECTION_MULTIPLE': (0.3752, 'gemma-3-27b-it\ngrok-4-fast\neq4'),
                'BEST_WITH_CORRECTION_MULTIPLE': (0.3585, 'gemma-3-27b-it\ngrok-4-fast\neq4')
            }
        }
    }
    return data

def plot_six_bar_charts(data):
    """Create a figure with 6 bar charts (2 rows x 3 columns)."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    datasets = ['OPEN', 'SPECIALIST', 'CONSOLIDATED']
    models = ['ARIMA', 'LSTM']
    
    categories = [
        'Baseline',
        'Best Single\n(No Correction)',
        'Best Single\n(With Correction)',
        'Best Multiple\n(No Correction)',
        'Best Multiple\n(With Correction)'
    ]
    
    keys = [
        'BASELINE',
        'BEST_NO_CORRECTION_SINGLE',
        'BEST_WITH_CORRECTION_SINGLE',
        'BEST_NO_CORRECTION_MULTIPLE',
        'BEST_WITH_CORRECTION_MULTIPLE'
    ]
    
    # Define colors: baseline=red, no correction=yellow, with correction=green
    # Following h_getStats.py color standard
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#f39c12', '#2ecc71']
    
    for row_idx, model in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            
            # Extract values and model info
            values = []
            model_infos = []
            for key in keys:
                val_tuple = data[dataset][model].get(key, (0, ''))
                if isinstance(val_tuple, tuple):
                    values.append(val_tuple[0])
                    model_infos.append(val_tuple[1])
                else:
                    values.append(val_tuple)
                    model_infos.append('')
            
            # Create bar chart
            x_pos = np.arange(len(categories))
            bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Set y-axis limits for better visualization first
            y_min = min(values) * 0.98
            y_max = max(values) * 1.02
            ax.set_ylim([y_min, y_max])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add model info inside bars
            for i, (bar, model_info) in enumerate(zip(bars, model_infos)):
                if model_info:
                    height = bar.get_height()
                    # Calculate position in the middle of the bar considering the y-axis range
                    bar_bottom = y_min
                    y_pos = bar_bottom + (height - bar_bottom) / 2
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                           model_info,
                           ha='center', va='center', fontsize=5.5, 
                           color='black', fontweight='normal',
                           bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                                   edgecolor='none', alpha=1))
            
            # Styling
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=0, ha='center', fontsize=10)
            ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset} - {model}', fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
    
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08, hspace=0.3, wspace=0.25)
    
    output_file_path = os.path.join(OUTPUT_FOLDER, "rmse_comparison_6plots.png")
    fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to: {output_file_path}")

def main():
    print("Creating 6-panel bar chart...")
    data = get_hardcoded_data()
    
    plot_six_bar_charts(data)
    
    print("Processing completed.")

if __name__ == "__main__":
    main()
