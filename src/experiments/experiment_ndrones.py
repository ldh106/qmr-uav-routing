from src.utilities.experiments_config import *
from src.experiments.parser.parser import command_line_parser
from src.utilities import config
from src.simulation.simulator import Simulator
import os


# -----------------------------
# Optional LR sweep configuration
# -----------------------------
LR_SWEEP_SUFFIXES = ["", "LR1p2", "LR1p4", "LR1p6"]
# "" means baseline (no suffix, alpha_scale=1.0)


def parse_algorithm(alg_str: str):
    """
    Parse algorithm string into base algorithm and suffix.

    Examples:
        QL                     -> ("QL", None)
        QL_EX100               -> ("QL", "EX100")
        QL_W0p9                -> ("QL", "W0p9")
        QL_B0p8                -> ("QL", "B0p8")
        QL_LR1p6               -> ("QL", "LR1p6")
        QL_DM0p3               -> ("QL", "DM0p3")
        QL_URG                 -> ("QL", "URG")
        QL_URG0p25_G3p0         -> ("QL", "URG0p25_G3p0")
        QL_DM0p3_URG0p25_G3p0   -> ("QL", "DM0p3_URG0p25_G3p0")
        GEO                    -> ("GEO", None)
        RND                    -> ("RND", None)
    """
    if "_" not in alg_str:
        return alg_str, None
    base, suffix = alg_str.split("_", 1)
    return base, suffix


def sim_setup(n_drones, seed, algorithm_base, alg_suffix=None):
    """
    Build an instance of Simulator using the parameters from experiments_config.py
    """

    return Simulator(
        len_simulation=len_simulation,
        time_step_duration=time_step_duration,
        seed=seed,
        n_drones=n_drones,
        env_width=env_width,
        env_height=env_height,

        drone_com_range=drone_com_range,
        drone_sen_range=drone_sen_range,
        drone_speed=drone_speed,
        drone_max_buffer_size=drone_max_buffer_size,
        drone_max_energy=drone_max_energy,
        drone_retransmission_delta=drone_retransmission_delta,
        drone_communication_success=drone_communication_success,
        event_generation_delay=event_generation_delay,

        depot_com_range=depot_com_range,
        depot_coordinates=depot_coordinates,

        event_duration=event_duration,
        event_generation_prob=event_generation_prob,
        packets_max_ttl=packets_max_ttl,

        routing_algorithm=config.RoutingAlgorithm[algorithm_base],
        communication_error_type=config.ChannelError.GAUSSIAN,
        show_plot=show_plot,

        simulation_name="",
        alg_suffix=alg_suffix,
    )


def _print_run_tag(alg_base: str, alg_suffix: str | None, n_drones: int, seed: int):
    if alg_suffix:
        print(f"Running {alg_base}_{alg_suffix} with {n_drones} drones seed {seed}")
    else:
        print(f"Running {alg_base} with {n_drones} drones seed {seed}")


def launch_experiments(n_drones, in_seed, out_seed, algorithm_raw, lr_sweep=False):
    """
    Launch simulations for a given algorithm string and number of drones.

    @param n_drones: number of drones
    @param in_seed: initial seed (inclusive)
    @param out_seed: final seed (exclusive)
    @param algorithm_raw: algorithm string from CLI (e.g., QL_URG0p25_G3p0)
    @param lr_sweep: if True, automatically run LR sweep for QL
    """
    alg_base, alg_suffix = parse_algorithm(algorithm_raw)

    if alg_base not in ["QL", "GEO", "RND"]:
        raise ValueError(f"Unknown routing algorithm base: {alg_base}")

    if lr_sweep and alg_base != "QL":
        print(f"[INFO] --lr_sweep is set, but algorithm is {alg_base}. LR sweep will be ignored.")
        lr_sweep = False

    for seed in range(in_seed, out_seed):

        if lr_sweep:
            for lr_suf in LR_SWEEP_SUFFIXES:
                run_suffix = None if lr_suf == "" else lr_suf

                _print_run_tag(alg_base, run_suffix, n_drones, seed)

                simulation = sim_setup(
                    n_drones=n_drones,
                    seed=seed,
                    algorithm_base=alg_base,
                    alg_suffix=run_suffix
                )
                simulation.run()
                simulation.close()

        else:
            _print_run_tag(alg_base, alg_suffix, n_drones, seed)

            simulation = sim_setup(
                n_drones=n_drones,
                seed=seed,
                algorithm_base=alg_base,
                alg_suffix=alg_suffix
            )
            simulation.run()
            simulation.close()


if __name__ == "__main__":

    args = command_line_parser.parse_args()

    number_of_drones = args.number_of_drones
    initial_seed = args.initial_seed
    end_seed = args.end_seed
    algorithm_routing = args.algorithm_routing

    # optional flag
    lr_sweep = getattr(args, "lr_sweep", False)

    path_filename = config.EXPERIMENTS_DIR
    os.makedirs(path_filename, exist_ok=True)

    launch_experiments(
        number_of_drones,
        initial_seed,
        end_seed,
        algorithm_routing,
        lr_sweep=lr_sweep
    )

    print("Simulations completed!")
