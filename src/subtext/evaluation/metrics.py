import pandas as pd

from subtext.data_models import ANIMALS, TrialMetrics


def compute_experiment_metrics(results: list[TrialMetrics]) -> dict:
    """Compute aggregate metrics from experiment results.

    Args:
        results: List of TrialMetrics from the experiment

    Returns:
        Dictionary with overall, per-model, and per-animal metrics
    """
    if not results:
        return {"overall": {}, "per_model": {}, "per_animal": {}}

    # Convert to DataFrame for easier analysis
    data = []
    for r in results:
        data.append(
            {
                "sender_model": r.trial.sender_model.key,
                "secret_animal": r.trial.secret_animal,
                "task": r.trial.task,
                "monitor_accuracy": r.monitor_accuracy,
                "receiver_accuracy": r.receiver_accuracy,
            }
        )

    df = pd.DataFrame(data)

    # Overall metrics
    overall = {
        "monitor_accuracy_mean": df.monitor_accuracy.mean(),
        "monitor_accuracy_std": df.monitor_accuracy.std(),
        "receiver_accuracy_mean": df.receiver_accuracy.mean(),
        "receiver_accuracy_std": df.receiver_accuracy.std(),
        "n_trials": len(df),
    }

    # Per-model metrics
    per_model = {}
    for model in df.sender_model.unique():
        model_df = df[df.sender_model == model]
        per_model[model] = {
            "monitor_accuracy_mean": model_df.monitor_accuracy.mean(),
            "monitor_accuracy_std": model_df.monitor_accuracy.std(),
            "receiver_accuracy_mean": model_df.receiver_accuracy.mean(),
            "receiver_accuracy_std": model_df.receiver_accuracy.std(),
            "n_trials": len(model_df),
        }

    # Per-animal metrics (for each target animal)
    per_animal = {}
    for animal in ANIMALS:
        animal_df = df[df.secret_animal == animal]
        if len(animal_df) > 0:
            per_animal[animal] = {
                "monitor_accuracy_mean": animal_df.monitor_accuracy.mean(),
                "monitor_accuracy_std": animal_df.monitor_accuracy.std(),
                "receiver_accuracy_mean": animal_df.receiver_accuracy.mean(),
                "receiver_accuracy_std": animal_df.receiver_accuracy.std(),
                "n_trials": len(animal_df),
            }

    return {
        "overall": overall,
        "per_model": per_model,
        "per_animal": per_animal,
    }


def print_metrics_summary(metrics: dict):
    """Print a formatted summary of the metrics."""
    print("\n" + "=" * 60)
    print("EXPERIMENT METRICS SUMMARY")
    print("=" * 60)

    overall = metrics["overall"]
    print(f"\nOverall ({overall.get('n_trials', 0)} trials):")
    print(f"  Monitor accuracy:  {overall.get('monitor_accuracy_mean', 0):.3f} "
          f"(+/- {overall.get('monitor_accuracy_std', 0):.3f})")
    print(f"  Receiver accuracy: {overall.get('receiver_accuracy_mean', 0):.3f} "
          f"(+/- {overall.get('receiver_accuracy_std', 0):.3f})")

    print("\nPer-model:")
    for model, stats in metrics.get("per_model", {}).items():
        print(f"  {model}:")
        print(f"    Monitor:  {stats['monitor_accuracy_mean']:.3f} "
              f"(+/- {stats['monitor_accuracy_std']:.3f})")
        print(f"    Receiver: {stats['receiver_accuracy_mean']:.3f} "
              f"(+/- {stats['receiver_accuracy_std']:.3f})")

    print("\nPer-animal:")
    for animal, stats in metrics.get("per_animal", {}).items():
        print(f"  {animal}: M={stats['monitor_accuracy_mean']:.3f}, "
              f"R={stats['receiver_accuracy_mean']:.3f}")

    print("=" * 60 + "\n")
