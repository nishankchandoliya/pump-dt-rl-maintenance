import pandas as pd

from src import config
from src.twin_sim import mode_lifetimes, simulate


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    seeds = config.EDA_SEEDS

    runs = simulate(seeds)
    runs.to_csv(config.DATA_DIR / "twin_runs.csv", index=False)
    print(f"twin_runs.csv: {len(runs)} weekly rows from {runs['seed'].nunique()} runs to first failure")

    rows = []
    for mode in ("bearing", "impeller"):
        for load in (None, 0.60, 0.75, 0.90):
            lives = mode_lifetimes(seeds, mode, load=load)
            label = "chain" if load is None else load
            rows += [{"seed": s, "mode": mode, "load": label, "life": life} for s, life in zip(seeds, lives)]
    lives = pd.DataFrame(rows)
    lives.to_csv(config.DATA_DIR / "twin_lives.csv", index=False)
    print(f"twin_lives.csv: {len(lives)} lifetimes (2 modes x 4 load settings x {len(seeds)} seeds)")


if __name__ == "__main__":
    main()
