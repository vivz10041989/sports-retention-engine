import pandas as pd
import numpy as np
import os


def create_sports_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    n_fans = 1000
    np.random.seed(42)  # Ensures we all get the same "random" results

    data = {
        'fan_id': range(n_fans),
        'matches_watched_loss': np.random.randint(0, 10, n_fans),
        'app_opens_month': np.random.randint(1, 30, n_fans),
        'membership_years': np.random.randint(1, 10, n_fans)
    }
    df = pd.DataFrame(data)

    # NEW: Complex logic with a "Random Noise" factor
    # This simulates real life: some people stay even if their team loses!
    noise = np.random.normal(0, 1, n_fans)
    churn_logic = (
            (df['matches_watched_loss'] * 1.2) -
            (df['app_opens_month'] * 0.5) -
            (df['membership_years'] * 0.8) +  # Higher years now SUBTRACTS from churn risk
            noise
    )

    # 1 if churn_logic > 2, else 0
    df['churn'] = (churn_logic > 2).astype(int)

    output_path = os.path.join(data_dir, "fan_data.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Real-world 'Noisy' data generated at: {output_path}")


if __name__ == "__main__":
    create_sports_data()