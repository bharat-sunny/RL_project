"""Robot-side code: everything that runs on the Jetson, and nothing that doesn't.

This package is deliberately isolated from ``src/``.  It depends only on NumPy
and ``pymycobot`` — both already present on the stock myCobot/Jetson image — so
the robot never needs PyTorch, Gymnasium or a simulator installed.  The trained
actor arrives as a ``.npz`` of weights and is evaluated by
:mod:`hardware.numpy_policy`, whose agreement with the PyTorch original is proven
by ``src/export_policy.py`` before anything is deployed.

Layout, in the order a session uses it:

``probe_arm``           read-only: which serial port answers, and how fast
``calibrate_workspace`` finds the largest safely reachable box (arm moves)
``characterize``        repeatability, accuracy, latency, settling (arm moves)
``show_workspace``      walks the box corners so they can be marked (arm moves)
``deploy``              runs a policy over a target grid — Experiment 4 (arm moves)
``mycobot_driver``      the only path from a policy action to a servo command
``numpy_policy``        dependency-free inference
``recorder``            annotated video capture from inside the trial loop

Every script that moves the arm refuses to run without an explicit
``--i-am-supervising`` flag, and each one also accepts ``--dry-run`` to exercise
the full code path against a simulated arm with no hardware attached.
"""
