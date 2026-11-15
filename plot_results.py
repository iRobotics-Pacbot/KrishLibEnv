import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

RES_DIR = "bin/results/"
MODEL_NAME = "kalil_pacbot"
CSR_DIR = "bin/eval_logs/"
PLOT_DIR = "bin/plots/"

def plot_post_training(
    csv_path=RES_DIR+MODEL_NAME+"_"+"episodes.csv",
    output_image_path="bin/plots/" + MODEL_NAME + "_post_training_performance.png"
):
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

def plot_training_progression():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # =====================================
    # Combine all evaluation CSV files
    # =====================================
    csv_files = glob.glob(os.path.join(CSR_DIR, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSR_DIR}")

    dataframes = []

    for file in csv_files:
        match = re.search(r"(\d+)", os.path.basename(file))
        if not match:
            continue

        df_temp = pd.read_csv(file)
        df_temp["timesteps"] = int(match.group(1))
        dataframes.append(df_temp)

    df = pd.concat(dataframes, ignore_index=True)

    # Sort evaluations chronologically
    df = df.sort_values("timesteps").reset_index(drop=True)

    # Save unified dataset
    combined_path = os.path.join(RES_DIR, "training_eval_summary.csv")
    df.to_csv(combined_path, index=False)

    x = df["timesteps"]

    # =====================================
    # 1. Score progression
    # =====================================
    plt.figure()
    plt.plot(x, df["mean_score"], marker="o", label="Mean score")

    plt.fill_between(
        x,
        df["mean_score"] - df["std_score"],
        df["mean_score"] + df["std_score"],
        alpha=0.2,
        label="Standard deviation",
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Score")
    plt.title("Score Progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "score_progression.png"))
    plt.close()

    # =====================================
    # 2. Success rate
    # =====================================
    plt.figure()
    plt.plot(x, df["success_rate"] * 100, marker="o")

    plt.xlabel("Training timesteps")
    plt.ylabel("Success rate (%)")
    plt.title("Success Rate")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "success_rate.png"))
    plt.close()

    # =====================================
    # 3. Reward progression
    # =====================================
    plt.figure()
    plt.plot(x, df["mean_total_reward"], marker="o")

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean reward")
    plt.title("Mean Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "reward_progression.png"))
    plt.close()

    # =====================================
    # 4. Gameplay performance
    # =====================================
    plt.figure()
    plt.plot(x, df["mean_wall_hits"], marker="o", label="Wall hits")
    plt.plot(x, df["mean_lives_lost"], marker="o", label="Lives lost")

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean count per episode")
    plt.title("Gameplay Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gameplay_performance.png"))
    plt.close()

    # =====================================
    # 5. Reward components
    # =====================================
    reward_components = [
        "mean_reward_score",
        "mean_reward_large_score_bonus",
        "mean_reward_score_event_bonus",
        "mean_reward_step",
        "mean_reward_life_loss",
        "mean_reward_wall_hit",
        "mean_reward_ghost_proximity",
        "mean_reward_pellet_proximity",
    ]

    plt.figure()

    for component in reward_components:
        plt.plot(
            x,
            df[component],
            marker="o",
            label=component.replace("mean_reward_", ""),
        )

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean reward component")
    plt.title("Reward Component Progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "reward_components.png"))
    plt.close()

    print(f"Combined {len(csv_files)} evaluation files.")
    print(f"Evaluations plotted: {len(df)}")
    print(f"Results saved to: {RES_DIR}")


if __name__ == "__main__":
    # plot_training_progression()
    plot_post_training()
