from enum import Enum

from src.drawing import pp_draw
from src.entities.uav_entities import *
from src.simulation.metrics import Metrics
from src.utilities import config, utilities
from src.routing_algorithms.net_routing import MediumDispatcher
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import math
import time

"""
Simulation class.

[Patch]
- Adds alg_suffix-based parameter injection for QMR variants:
  - QL_W0p9   -> omega = 0.9
  - QL_B0p8   -> beta = 0.8
  - QL_EX100  -> ExpireTime = 100
  - QL_LR1p6  -> alpha_scale = 1.6   (learning-rate scale for Q-learning update; used by QMAR/QL only)
  - QL_DM0p3  -> dm_lambda = 0.3     (deadline-miss risk penalty strength; used by QMAR/QL only)
  - QL_URG0p25_G3p0 -> allowed through (handled by QMAR; simulator must not reject it)
- Keeps baseline behavior unchanged:
  - baseline omega defaults to 0.7
  - baseline beta defaults to 0.5
  - baseline ExpireTime defaults to 300
  - baseline alpha_scale defaults to 1.0
  - baseline dm_lambda defaults to 0.0
- Ensures output JSON filename contains the suffix by using routing_algorithm_tag.
"""


class Simulator:

    def __init__(self,
                 n_drones,
                 seed,
                 len_simulation=15000,
                 time_step_duration=config.TS_DURATION,
                 env_width=config.ENV_WIDTH,
                 env_height=config.ENV_HEIGHT,
                 drone_com_range=config.COMMUNICATION_RANGE_DRONE,
                 drone_sen_range=config.SENSING_RANGE_DRONE,
                 drone_speed=config.DRONE_SPEED,
                 drone_max_buffer_size=config.DRONE_MAX_BUFFER_SIZE,
                 drone_max_energy=config.DRONE_MAX_ENERGY,
                 drone_retransmission_delta=config.RETRANSMISSION_DELAY,
                 drone_communication_success=config.COMMUNICATION_P_SUCCESS,
                 depot_com_range=config.DEPOT_COMMUNICATION_RANGE,
                 depot_coordinates=config.DEPOT_COO,
                 event_duration=config.EVENTS_DURATION,
                 event_generation_prob=config.P_FEEL_EVENT,
                 event_generation_delay=config.D_FEEL_EVENT,
                 packets_max_ttl=config.PACKETS_MAX_TTL,
                 show_plot=config.PLOT_SIM,
                 routing_algorithm=config.ROUTING_ALGORITHM,
                 communication_error_type=config.CHANNEL_ERROR_TYPE,
                 prob_size_cell_r=config.CELL_PROB_SIZE_R,
                 simulation_name="",
                 alg_suffix=None):

        self.cur_step = None
        self.drone_com_range = drone_com_range
        self.drone_sen_range = drone_sen_range
        self.drone_speed = drone_speed
        self.drone_max_buffer_size = drone_max_buffer_size
        self.drone_max_energy = drone_max_energy
        self.drone_retransmission_delta = drone_retransmission_delta
        self.drone_communication_success = drone_communication_success
        self.n_drones = n_drones
        self.env_width = env_width
        self.env_height = env_height
        self.depot_com_range = depot_com_range
        self.depot_coordinates = depot_coordinates
        self.len_simulation = len_simulation
        self.time_step_duration = time_step_duration
        self.seed = seed
        self.event_duration = event_duration
        self.event_max_retrasmission = math.ceil(event_duration / drone_retransmission_delta)
        self.event_generation_prob = event_generation_prob
        self.event_generation_delay = event_generation_delay
        self.packets_max_ttl = packets_max_ttl
        self.show_plot = show_plot
        self.routing_algorithm = routing_algorithm
        self.communication_error_type = communication_error_type
        self.seed = seed

        # -----------------------------
        # Default hyperparameters (baseline)
        # -----------------------------
        self.beta = 0.5
        self.omega = 0.7
        self.ExpireTime = 300

        # -----------------------------
        # Learning-rate control (baseline)
        # -----------------------------
        self.alpha_scale = 1.0
        self.alpha_min = 0.05
        self.alpha_max = 0.95

        # -----------------------------
        # Deadline-miss penalty (baseline)
        # Used by QMAR/QL variant that adds miss-risk penalty in relay selection.
        # -----------------------------
        self.dm_lambda = 0.0

        # -----------------------------
        # Variant injection via suffix
        # -----------------------------
        self.alg_suffix = alg_suffix
        self._apply_alg_suffix_overrides()

        # Routing tag used for output filename (supports suffix without requiring Enum extension)
        base_name = self.routing_algorithm.name if hasattr(self.routing_algorithm, "name") else str(self.routing_algorithm)
        self.routing_algorithm_tag = base_name if not self.alg_suffix else f"{base_name}_{self.alg_suffix}"

        # --------------- cell for drones -------------
        self.prob_size_cell_r = prob_size_cell_r
        self.prob_size_cell = int(self.drone_com_range * self.prob_size_cell_r)
        self.cell_prob_map = defaultdict(lambda: [0, 0, 0])

        self.sim_save_file = config.SAVE_PLOT_DIR + self.__sim_name()
        self.path_to_depot = None

        # Setup metrics
        self.metrics = Metrics(self)

        # setup network
        self.__setup_net_dispatcher()

        # Setup the simulation
        self.__set_simulation()
        self.__set_metrics()

        self.simulation_name = "simulation-" + utilities.date() + "_" + str(simulation_name) + "_" + str(self.seed) + "_" + str(self.n_drones) + "_" + str(self.routing_algorithm_tag)
        self.simulation_test_dir = self.simulation_name + "/"

        self.start = time.time()
        self.event_generator = utilities.EventGenerator(self)

    # -----------------------------
    # Suffix parsing helpers
    # -----------------------------
    @staticmethod
    def _parse_float_with_p(s: str):
        """
        Parses strings like '0p9' -> 0.9, '1p0' -> 1.0
        """
        return float(s.replace("p", "."))

    def _apply_alg_suffix_overrides(self):
        """
        Apply algorithm suffix overrides (parameter injections).

        Supported tokens (can be combined with "_"):
          - EX###        -> ExpireTime = ###
          - B0p# / B#p#  -> beta = float
          - W0p# / W#p#  -> omega = float
          - LR#p#        -> alpha_scale = float
          - DM#p#        -> dm_lambda = float
          - URG / URG0p# -> allowed through (handled by QMAR in q_learning_routing.py)
          - G#p#         -> allowed through (handled by QMAR in q_learning_routing.py)
        """
        suf = getattr(self, "alg_suffix", None)
        if not isinstance(suf, str) or len(suf.strip()) == 0:
            return

        tokens = [t for t in suf.split("_") if t]

        for t in tokens:
            if t.startswith("EX"):
                try:
                    self.ExpireTime = int(t[2:])
                except Exception:
                    raise ValueError(f"Invalid EX token in alg_suffix: {t} (expected EX###)")
                continue

            if t.startswith("B"):
                try:
                    self.beta = self._parse_float_with_p(t[1:])
                except Exception:
                    raise ValueError(f"Invalid B token in alg_suffix: {t} (expected B0p#)")
                continue

            if t.startswith("W"):
                try:
                    self.omega = self._parse_float_with_p(t[1:])
                except Exception:
                    raise ValueError(f"Invalid W token in alg_suffix: {t} (expected W0p#)")
                continue

            if t.startswith("LR"):
                try:
                    self.alpha_scale = self._parse_float_with_p(t[2:])
                except Exception:
                    raise ValueError(f"Invalid LR token in alg_suffix: {t} (expected LR#p#)")
                continue

            if t.startswith("DM"):
                try:
                    self.dm_lambda = self._parse_float_with_p(t[2:])
                except Exception:
                    raise ValueError(f"Invalid DM token in alg_suffix: {t} (expected DM#p#)")
                continue

            # NEW: allow urgent-mode tokens (interpreted by QMAR; simulator only needs to pass them through)
            if t == "URG" or t.startswith("URG"):
                continue

            if t.startswith("G"):
                continue

            raise ValueError(
                f"Unknown alg_suffix token: {t}. Supported tokens: "
                f"EX###, B0p#, W0p#, LR#p#, DM#p#, URG(0p#), G#p# (combine with '_')."
            )

    def __setup_net_dispatcher(self):
        self.network_dispatcher = MediumDispatcher(self.metrics)

    def __set_metrics(self):
        """ sets up all the parameters in the metrics class """
        self.metrics.info_mission()

    def __set_random_generators(self):
        if self.seed is not None:
            self.rnd_network = np.random.RandomState(self.seed)
            self.rnd_routing = np.random.RandomState(self.seed)
            self.rnd_env = np.random.RandomState(self.seed)
            self.rnd_event = np.random.RandomState(self.seed)

    def __set_simulation(self):
        """ creates all the uav entities """

        self.__set_random_generators()

        self.path_manager = utilities.PathManager(config.PATH_FROM_JSON, config.JSONS_PATH_PREFIX, self.seed)
        self.environment = Environment(self.env_width, self.env_height, self)

        self.depot = Depot(self.depot_coordinates, self.depot_com_range, self)

        self.drones = []

        # drone 0 is the first
        for i in range(self.n_drones):
            self.drones.append(Drone(i, self.path_manager.path(i, self), self.depot, self))

        self.environment.add_drones(self.drones)
        self.environment.add_depot(self.depot)

        self.max_dist_drone_depot = utilities.euclidean_distance(self.depot.coords, (self.env_width, self.env_height))

        if self.show_plot or config.SAVE_PLOT:
            self.draw_manager = pp_draw.PathPlanningDrawer(self.environment, self, borders=True)

    def __sim_name(self):
        """
        return identification name for current simulation (used for saved frames)
        """
        return "sim_seed" + str(self.seed) + "drones" + str(self.n_drones) + "_step"

    def __plot(self, cur_step):
        """ plot the simulation """
        if cur_step % config.SKIP_SIM_STEP != 0:
            return

        if config.WAIT_SIM_STEP > 0:
            time.sleep(config.WAIT_SIM_STEP)

        for drone in self.drones:
            self.draw_manager.draw_drone(drone, cur_step)

        self.draw_manager.draw_depot(self.depot)

        for event in self.environment.active_events:
            self.draw_manager.draw_event(event)

        self.draw_manager.draw_simulation_info(cur_step=cur_step, max_steps=self.len_simulation)

        self.draw_manager.update(show=self.show_plot, save=config.SAVE_PLOT,
                                 filename=self.sim_save_file + str(cur_step) + ".png")

    def increase_meetings_probs(self, drones, cur_step):
        """ Increases the probabilities of meeting someone. """
        cells = set()
        for drone in drones:
            coords = drone.coords
            cell_index = utilities.TraversedCells.coord_to_cell(size_cell=self.prob_size_cell,
                                                                width_area=self.env_width,
                                                                x_pos=coords[0],
                                                                y_pos=coords[1])
            cells.add(int(cell_index[0]))

        for cell, cell_center in utilities.TraversedCells.all_centers(self.env_width, self.env_height,
                                                                      self.prob_size_cell):

            index_cell = int(cell[0])
            old_vals = self.cell_prob_map[index_cell]

            if index_cell in cells:
                old_vals[0] += 1

            old_vals[1] = cur_step + 1
            old_vals[2] = old_vals[0] / max(1, old_vals[1])
            self.cell_prob_map[index_cell] = old_vals

    def run(self):
        """
        Simulator main function
        """
        for cur_step in range(self.len_simulation):
            self.cur_step = cur_step

            self.network_dispatcher.run_medium(cur_step)

            self.event_generator.handle_events_generation(cur_step, self.drones)

            for drone in self.drones:
                drone.update_packets(cur_step)
                drone.routing(self.drones, self.depot, cur_step)
                drone.move(self.time_step_duration)

            if config.ENABLE_PROBABILITIES:
                self.increase_meetings_probs(self.drones, cur_step)

            if self.show_plot or config.SAVE_PLOT:
                self.__plot(cur_step)

        if config.DEBUG:
            print("End of simulation, sim time: " + str(
                (cur_step + 1) * self.time_step_duration) + " sec, #iteration: " + str(cur_step + 1))

    def close(self):
        """ do some stuff at the end of simulation"""
        print("Closing simulation")

        self.print_metrics(plot_id="final")

        # Use routing_algorithm_tag so suffix is reflected in output JSON
        filename_path = (config.EXPERIMENTS_DIR + f"out__ndrones_{self.n_drones}_seed{self.seed}_alg_{self.routing_algorithm_tag}")
        print("[SAVE] writing:", filename_path + ".json")
        self.save_metrics(filename_path)

    def print_metrics(self, plot_id="final"):
        self.metrics.print_overall_stats()

    def save_metrics(self, filename_path, save_pickle=False):
        self.metrics.save_as_json(filename_path + ".json")
        if save_pickle:
            self.metrics.save(filename_path + ".pickle")

    def score(self):
        score = round(self.metrics.score(), 2)
        return score
