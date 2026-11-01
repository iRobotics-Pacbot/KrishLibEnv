import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pacbot_performance(csv_path, output_image_path="bin/plots/pacbot_performance.png"):
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['episode'], df['score'], marker='o', color='#2ca02c', label='Game Score', linewidth=2)
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Game Score', color='#2ca02c', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#2ca02c')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['episode'], df['total_reward'], marker='s', color='#1f77b4', label='Total Reward', linestyle='--')
    ax1_twin.set_ylabel('Total Reward', color='#1f77b4', fontsize=11)
    ax1_twin.tick_params(axis='y', labelcolor='#1f77b4')

    ax1.set_title('Score & Reward Progression', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.bar(df['episode'] - 0.2, df['steps'], width=0.4, color='#bcbd22', label='Total Steps', align='center')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Steps Count', color='#bcbd22', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#bcbd22')

    ax2_twin = ax2.twinx()
    ax2_twin.bar(df['episode'] + 0.2, df['wall_hits'], width=0.4, color='#d62728', label='Wall Hits', align='center')
    ax2_twin.set_ylabel('Wall Hits Count', color='#d62728', fontsize=11)
    ax2_twin.tick_params(axis='y', labelcolor='#d62728')

    ax2.set_title('Agent Efficiency (Steps vs. Wall Penalties)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Analysis plots saved successfully to: {output_image_path}")
    plt.show()

CSV_DIR = "bin/results"
MODEL_NAME = "dqn_pacbot"

if __name__ == "__main__":
    plot_pacbot_performance(os.path.join(CSV_DIR, f"{MODEL_NAME}_episodes.csv"))