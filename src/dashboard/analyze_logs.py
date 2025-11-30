# src/dashboard/analyze_logs.py

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_ai_engine_log(log_file_path):
    data = []
    with open(log_file_path, 'r') as f:
        for line in f:
            # Regex to capture timestamp, model, task, and generation time
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ai_engine - INFO - (GPT-2|CTGAN|TimeGAN/LSTM) generated (\w+) in (\d+\.\d+)s', line)
            if match:
                timestamp_str, model, task, gen_time_str = match.groups()
                data.append({
                    'timestamp': pd.to_datetime(timestamp_str.replace(',', '.')),
                    'model': model,
                    'task': task,
                    'generation_time': float(gen_time_str)
                })
    return pd.DataFrame(data)

if __name__ == '__main__':
    log_file = 'logs/ai_engine.log'
    df = parse_ai_engine_log(log_file)

    # Create a directory to save plots
    output_dir = 'reports/eda'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize GPT-2 generation times
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['model'] == 'GPT-2']['generation_time'], bins=10, kde=True)
    plt.title('GPT-2 Generation Time Distribution')
    plt.xlabel('Generation Time (s)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'gpt2_generation_time_histogram.png'))
    plt.close()

    # Visualize CTGAN model training times
    plt.figure(figsize=(12, 6))
    sns.histplot(df[(df['model'] == 'CTGAN') & (df['task'] == 'customers_model_training')]['generation_time'], bins=10, kde=True)
    plt.title('CTGAN Model Training Time Distribution')
    plt.xlabel('Generation Time (s)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'ctgan_training_time_histogram.png'))
    plt.close()

    # Visualize CTGAN synthetic data generation times
    plt.figure(figsize=(12, 6))
    sns.histplot(df[(df['model'] == 'CTGAN') & (df['task'] == 'customers_synthetic_data')]['generation_time'], bins=10, kde=True)
    plt.title('CTGAN Synthetic Data Generation Time Distribution')
    plt.xlabel('Generation Time (s)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'ctgan_synthetic_data_time_histogram.png'))
    plt.close()

    print(f"Generated plots saved to {output_dir}")
