"""Phase 1: look at TraffiDent before building anything. Read-only; prints counts.

    python explore_traffident.py

Answers: which districts have enough freeway sensors, which incident types exist (are there real
road closures?), and whether the sensor list matches the traffic data's node order.
"""
import numpy as np
import pandas as pd

from config import RAW

TD = RAW / "traffident"
pd.set_option("display.width", 160)


def load():
    sensors = pd.read_csv(TD / "sensor_meta_feature.csv", sep="\t")
    incidents = pd.read_csv(next(TD.glob("incidents_y2023*.csv")), low_memory=False)
    order = np.load(TD / "node_order.npy")
    return sensors, incidents, order


def is_closure(description):
    """CLOSURE of a Road, CLOSURE-Closure of a Road, SIG Alert (CHP's alert for lanes blocked 30+ min)."""
    return description.str.contains(r"CLOS|BLOCK|SIG ?ALERT", case=False, na=False)


def section(title):
    print(f"\n=== {title}")


def main():
    sensors, incidents, order = load()

    section("Sensors")
    print(f"{len(sensors)} sensors; node_order has {len(order)}; "
          f"same set of IDs: {set(order) == set(sensors.station_id)}")
    print(sensors.Type.value_counts().to_string())
    main_line = sensors[sensors.Type == "Mainline"]
    print("\nMainline sensors per district:")
    print(main_line.groupby("District").size().to_string())

    section("Incidents (all of California, 2023)")
    print(f"{len(incidents)} incidents; {incidents.duration.isna().mean():.0%} have no duration")
    print(incidents.duration.describe(percentiles=[.25, .5, .75, .9]).round(1).to_string())
    print("\nMost common incident codes (DESCRIPTION):")
    print(incidents.DESCRIPTION.value_counts().head(25).to_string())
    closure = is_closure(incidents.DESCRIPTION)
    print(f"\nDescriptions mentioning closure / blocked / SigAlert: {closure.sum()}")
    print(incidents.DESCRIPTION[closure].value_counts().head(15).to_string())

    section("Incidents per district (matched by freeway number + nearest mile marker)")
    inc = incidents.dropna(subset=["Fwy", "Abs PM"]).copy()
    inc["Fwy"] = inc.Fwy.astype(int)
    sens = main_line[["Fwy", "Abs PM", "District"]].dropna().copy()
    sens["Fwy"] = sens.Fwy.astype(int)
    matched = pd.merge_asof(inc.sort_values("Abs PM"), sens.sort_values("Abs PM"), on="Abs PM",
                            by="Fwy", direction="nearest", tolerance=1.0)
    print(f"{matched.District.notna().mean():.0%} of incidents lie within 1 mile of a mainline sensor")
    code = matched.DESCRIPTION.str.split("-").str[0]
    kind = np.select([is_closure(matched.DESCRIPTION), code.isin(["1179", "1180", "1181", "1182", "1183"]),
                      code.isin(["1125", "1125A"])],
                     ["closure/blocked", "collision", "hazard"], "other")
    table = pd.crosstab(matched.District, kind, margins=True)
    print(table.astype(int).to_string())


if __name__ == "__main__":
    main()
