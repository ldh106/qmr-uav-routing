from argparse import ArgumentParser

command_line_parser = ArgumentParser()

command_line_parser.add_argument(
    "-nd",
    dest="number_of_drones",
    action="store",
    type=int,
    help="the number of drones to use in the simulation"
)

command_line_parser.add_argument(
    "-i_s",
    dest="initial_seed",
    action="store",
    type=int,
    help="the initial seed (included) to use in the simulations"
)

command_line_parser.add_argument(
    "-e_s",
    dest="end_seed",
    action="store",
    type=int,
    help=(
        "the end seed (excluded) to use in the simulations; "
        "notice that the simulations will run for seed in [i_s, e_s)"
    )
)
command_line_parser.add_argument(
    "--lr_sweep",
    action="store_true",
    help="Run learning-rate sweep: baseline + LR1p2/LR1p4/LR1p6"
)


# IMPORTANT:
# We intentionally DO NOT restrict choices here, because we want to allow
# algorithm strings with suffixes like:
#   QL_EX100, QL_W0p9, QL_B0p8
# Validation and parsing are handled in experiment_ndrones.py.
command_line_parser.add_argument(
    "-alg",
    dest="algorithm_routing",
    action="store",
    type=str,
    help="routing algorithm (e.g., QL, QL_EX100, QL_W0p9, QL_B0p8, GEO, RND)"
)
