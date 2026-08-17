"""Read-only connectivity probe for the myCobot.

Run this before anything that moves.  It identifies which serial port and baud
rate the arm actually answers on, reports joint angles and end-effector pose,
and measures the round-trip latency of a state read — a number the simulation
needs, because the policy was trained at a control rate the hardware has to be
able to sustain.

Nothing here commands motion.

    python3 hardware/probe_arm.py
"""

from __future__ import annotations

import argparse
import json
import time

# Ports and baud rates used across myCobot variants: the Jetson/Pi models expose
# the arm on an on-board UART, the M5-stack models through a USB bridge.
CANDIDATE_PORTS = ["/dev/ttyUSB0", "/dev/ttyTHS1", "/dev/ttyAMA0", "/dev/ttyACM0"]
CANDIDATE_BAUDS = [1_000_000, 115_200]


def _plausible_angles(angles) -> bool:
    """A live arm returns six joint angles inside its mechanical range."""
    if not angles or len(angles) != 6:
        return False
    if all(abs(a) < 1e-9 for a in angles):
        return False  # all-zero usually means "no reply parsed"
    return all(-200.0 <= float(a) <= 200.0 for a in angles)


def probe(port: str, baud: int, settle: float = 2.0) -> dict | None:
    """Try one (port, baud) pair; return a report if the arm answers."""
    from pymycobot import MyCobot

    try:
        arm = MyCobot(port, baud)
    except Exception as exc:
        return {"port": port, "baud": baud, "ok": False, "error": f"open failed: {exc}"}

    time.sleep(settle)  # the controller needs a moment after the port opens

    try:
        angles = arm.get_angles()
        coords = arm.get_coords()
    except Exception as exc:
        return {"port": port, "baud": baud, "ok": False, "error": f"read failed: {exc}"}

    if not _plausible_angles(angles):
        return {"port": port, "baud": baud, "ok": False,
                "error": f"implausible reply: angles={angles}"}

    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        arm.get_coords()
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()

    return {
        "port": port,
        "baud": baud,
        "ok": True,
        "angles_deg": angles,
        "coords": coords,  # [x, y, z] mm and [rx, ry, rz] deg
        "read_latency_ms": {
            "min": round(latencies[0], 2),
            "median": round(latencies[len(latencies) // 2], 2),
            "max": round(latencies[-1], 2),
            "mean": round(sum(latencies) / len(latencies), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="skip detection and use this port")
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="print machine-readable output only")
    args = parser.parse_args()

    combos = (
        [(args.port, args.baud or 1_000_000)]
        if args.port
        else [(p, b) for p in CANDIDATE_PORTS for b in CANDIDATE_BAUDS]
    )

    results = []
    connected = None
    for port, baud in combos:
        report = probe(port, baud)
        if report is None:
            continue
        results.append(report)
        if not args.json:
            status = "OK " if report["ok"] else "-- "
            detail = report.get("error", f"angles={report.get('angles_deg')}")
            print(f"{status} {port:<16s} @ {baud:>9,d}  {detail}")
        if report["ok"]:
            connected = report
            break

    if args.json:
        print(json.dumps({"connected": connected, "attempts": results}, indent=2))
        return

    if connected is None:
        print("\nNo arm responded. Check that it is powered on and the cable is seated.")
        raise SystemExit(1)

    print(f"\nArm found on {connected['port']} @ {connected['baud']:,} baud")
    print(f"  joint angles (deg) : {[round(a, 2) for a in connected['angles_deg']]}")
    print(f"  end effector       : x={connected['coords'][0]:.1f} y={connected['coords'][1]:.1f} "
          f"z={connected['coords'][2]:.1f} mm")
    print(f"  state read latency : {connected['read_latency_ms']['median']:.1f} ms median, "
          f"{connected['read_latency_ms']['max']:.1f} ms max")


if __name__ == "__main__":
    main()
