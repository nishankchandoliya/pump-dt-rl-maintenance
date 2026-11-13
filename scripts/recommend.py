import argparse

from src import config
from src.advisor import ACTIONS, recommend
from src.dqn import load_network


def main():
    parser = argparse.ArgumentParser(description="maintenance advice for one boiler feed pump from its latest readings")
    parser.add_argument("--vibration", type=float, required=True, help="vibration, mm/s RMS")
    parser.add_argument("--temperature", type=float, required=True, help="bearing temperature, °C")
    parser.add_argument("--efficiency", type=float, required=True, help="average efficiency of the last 8 weeks, for example 0.78")
    parser.add_argument("--load", type=float, required=True, help="unit load as a fraction of rated flow, 0.60 to 0.90")
    parser.add_argument("--bearing-weeks", type=float, required=True, help="running weeks since the bearing was renewed")
    parser.add_argument("--impeller-weeks", type=float, required=True, help="running weeks since the impeller was renewed")
    parser.add_argument("--model", default=str(config.MODELS_DIR / "dqn.pt"))
    a = parser.parse_args()

    net = load_network(a.model)
    try:
        r = recommend(net, a.vibration, a.temperature, a.efficiency, a.load, a.bearing_weeks, a.impeller_weeks)
    except ValueError as error:
        parser.error(str(error))
    print(f"vibration {a.vibration:.1f} mm/s: ISO 10816-7 zone {r['vibration_zone']}")
    print(f"bearing temperature {a.temperature:.1f} °C: {r['temperature_status']} (alarm {config.TEMP_ALARM:.0f}, trip {config.TEMP_TRIP:.0f})")
    print(f"8 week efficiency {a.efficiency:.3f}: {r['efficiency_drop']:.3f} below a new pump")
    print("\nexpected discounted cost from now on (1 unit = about 23.5 lakh Rs):")
    for action, name in ACTIONS.items():
        mark = "  <- recommended" if action == r["action"] else ""
        print(f"  {name:18s} {r['cost'][action]:7.2f}{mark}")
    print(f"\nrecommendation: {r['name']}: {r['work']}. The next best action costs {r['margin']:.2f} units more")


if __name__ == "__main__":
    main()
