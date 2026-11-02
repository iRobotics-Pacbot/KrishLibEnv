import os
from stable_baselines3.common.callbacks import BaseCallback
from evaluate import OUTPUT_DIR, evaluate_model, summarize_results, print_summary, save_csv

# templated directly from Stable Baselines Documentation
# See https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

class CustomPacBotCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            print(f"\n[Step {self.n_calls}] Triggering CSV Evaluation...")
            episode_results, action_sequences = evaluate_model(self.model, self.eval_env)
            summary = summarize_results(episode_results)
            print_summary(summary, action_sequences)
            save_csv(os.path.join(OUTPUT_DIR, f"checkpoint_{self.num_timesteps}_summary.csv"), [summary])
        return True
