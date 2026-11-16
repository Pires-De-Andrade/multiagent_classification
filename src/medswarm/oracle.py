"""O0 oracle: calculate(Relevant Entities) vs Ground Truth. Flags rows whose label the official code can't reproduce."""
import io
from contextlib import redirect_stdout

from medswarm.calculators import calculate, is_correct
from medswarm.data import ROOT, load

OUT = ROOT / "data" / "oracle"


def oracle_answer(row):
    try:
        with redirect_stdout(io.StringIO()):  # some implementations print
            return calculate(row["Calculator ID"], row["entities"])
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        df = load(split)
        df["oracle_answer"] = df.apply(oracle_answer, axis=1)
        df["oracle_consistent"] = df.apply(lambda r: is_correct(r["oracle_answer"], r), axis=1)
        df["oracle_answer"] = df["oracle_answer"].astype(str)
        df.drop(columns=["entities"]).to_parquet(OUT / f"{split}.parquet")
        by_calc = df.groupby("Calculator Name")["oracle_consistent"].agg(["mean", "size"])
        print(f"\n== {split}: {df['oracle_consistent'].mean():.1%} consistent ({len(df)} rows)")
        print(by_calc.sort_values("mean").to_string(float_format="{:.0%}".format))
        errors = df["oracle_answer"].astype(str).str.startswith("ERROR").sum()
        print(f"implementation errors: {errors}")


if __name__ == "__main__":
    main()
