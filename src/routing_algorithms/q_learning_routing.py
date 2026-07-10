import random
import numpy as np
import math

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities.policies import *


class QMAR(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

        self.maxReward = 5
        self.minReward = -5

        # Baseline weight (omega) : allow external injection (e.g., QL_W0p9)
        self.w = getattr(self.simulator, "omega", 0.7)

        # Deadline-miss penalty strength (optional)
        # token: DM0p3 -> dm_lambda = 0.3
        self.dm_lambda = 0.0

        # -----------------------------
        # Deadline-Aware Two-Mode Policy (Urgent Mode)
        # -----------------------------
        # enable flag
        self.enable_urgent_mode = False

        # when ttl_frac <= urgent_ttl_frac -> urgent mode
        # token: URG0p25 -> urgent_ttl_frac = 0.25
        self.urgent_ttl_frac = 0.25

        # urgent bonus gain
        # token: G3p0 -> urgency_gain = 3.0
        self.urgency_gain = 3.0

        # Apply suffix-based injections (DM / URG / G)
        suf = getattr(self.simulator, "alg_suffix", None)
        self._apply_suffix_tokens(suf)

    # ------------------------------------------------------------------
    # Suffix parsing
    # ------------------------------------------------------------------
    
    @staticmethod
    def _token_to_float(token: str) -> float:
        return float(token.replace("p", "."))

    def _apply_suffix_tokens(self, suf):
        if not isinstance(suf, str) or len(suf.strip()) == 0:
            return

        tokens = [t for t in suf.split("_") if t]

        for t in tokens:
            # Deadline miss penalty
            if t.startswith("DM"):
                raw = t[2:]
                try:
                    self.dm_lambda = float(raw.replace("p", "."))
                except Exception:
                    self.dm_lambda = 0.0
                continue

            # Urgent mode enable + threshold
            if t.startswith("URG"):
                self.enable_urgent_mode = True
                raw = t[3:]
                if raw:
                    try:
                        self.urgent_ttl_frac = self._token_to_float(raw)
                    except Exception:
                        pass
                continue

            # Urgency gain
            if t.startswith("G"):
                self.enable_urgent_mode = True
                raw = t[1:]
                if raw:
                    try:
                        self.urgency_gain = self._token_to_float(raw)
                    except Exception:
                        pass
                continue

        # clamps
        self.urgent_ttl_frac = max(0.0, min(1.0, float(self.urgent_ttl_frac)))
        self.urgency_gain = float(self.urgency_gain)
        self.dm_lambda = float(self.dm_lambda)

    # ------------------------------------------------------------------
    # Optional: learning-rate scaling (if your neighbor_table stores alpha)
    # ------------------------------------------------------------------
    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _effective_alpha(self, id_j: int) -> float:
        base_alpha = float(self.drone.neighbor_table[id_j, 10])

        alpha_scale = float(getattr(self.simulator, "alpha_scale", 1.0))
        alpha_min = float(getattr(self.simulator, "alpha_min", 0.0))
        alpha_max = float(getattr(self.simulator, "alpha_max", 1.0))

        eff = base_alpha * alpha_scale
        eff = self._clip(eff, alpha_min, alpha_max)
        return eff

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------
    def feedback(self, outcome, id_j, Q_value_best_action):
        alpha = self._effective_alpha(id_j)
        gamma = self.drone.neighbor_table[id_j, 7]
        Q_value_i_j = self.drone.neighbor_table[id_j, 9]

        if outcome == 1:
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (self.maxReward)

        elif outcome == 0:
            delay = self.drone.neighbor_table[id_j, 8] + self.drone.neighbor_table[id_j, 11]
            reward = self.computeReward(outcome, delay)
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (reward + gamma * Q_value_best_action - Q_value_i_j)

        else:
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (self.minReward + gamma * Q_value_best_action - Q_value_i_j)

    def computeReward(self, outcome, delay):
        return self.w * math.exp(delay) + (1 - self.w) * (self.drone.residual_energy / self.drone.initial_energy)

    # ------------------------------------------------------------------
    # Relay selection (Decision structure changed here)
    # ------------------------------------------------------------------
    @staticmethod
    def deadline_miss_risk(actual_v: float, req_v: float) -> float:
        """
        Risk proxy:
        - 0이면 deadline 만족 가능성이 높음
        - 1에 가까울수록 deadline miss 위험 높음
        """
        if req_v <= 0:
            return 0.0
        if actual_v <= 0:
            return 1.0
        if actual_v >= req_v:
            return 0.0
        # (req_v - actual_v)/req_v : 부족분 비율
        return max(0.0, min(1.0, (req_v - actual_v) / req_v))
    def relay_selection(self, opt_neighbors, data):
        """
        Two-Mode Policy:
          - Normal: score = (Q * k) - dm_lambda * miss_risk
          - Urgent: score = (Q * k) - dm_lambda * miss_risk + urgency_gain * (1-ttl_frac) * progress

        progress = (dist_i - dist_j) / max_dist, positive is good (closer to depot)
        ttl_frac  = remaining_steps / max_ttl_steps
        """
        packet = data[0]
        candidates = []
        candidates2 = []

        # TTL horizon
        max_ttl_steps = int(getattr(self.simulator, "event_duration", 2001))
        if max_ttl_steps <= 0:
            max_ttl_steps = 2001

        remaining_steps = max_ttl_steps - (self.simulator.cur_step - packet.time_step_creation)
        if remaining_steps <= 0:
            return None

        ttl_frac = remaining_steps / max_ttl_steps
        ttl_frac = max(0.0, min(1.0, ttl_frac))

        urgent_mode = self.enable_urgent_mode and (ttl_frac <= self.urgent_ttl_frac)

        dist_i = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)

        max_dist = getattr(self.simulator, "max_dist_drone_depot", None)
        if max_dist is None or max_dist <= 0:
            env_w = getattr(self.simulator, "env_width", None)
            env_h = getattr(self.simulator, "env_height", None)
            if env_w is not None and env_h is not None:
                max_dist = util.euclidean_distance((0, 0), (env_w, env_h))
            else:
                max_dist = max(1.0, dist_i)

        req_v = dist_i / max(1, remaining_steps)

        # iterate neighbors
        for node_j in self.simulator.drones:
            if node_j in opt_neighbors:
                j = node_j.identifier

                actual_v, distance_i_j, distance_j, delay = self.computeActualVel(j, node_j, dist_i)

                # miss risk (try policies.py if exists; fallback otherwise)
                try:
                    miss_risk = self.deadline_miss_risk(actual_v, req_v)
                except Exception:
                    if actual_v <= 0:
                        miss_risk = 1.0
                    elif actual_v < req_v:
                        miss_risk = float((req_v - actual_v) / max(req_v, 1e-9))
                    else:
                        miss_risk = 0.0

                # k as original
                LQ = self.drone.neighbor_table[j, 12]
                R = self.drone.communication_range
                if distance_i_j > R:
                    M = 0
                else:
                    M = 1 - (distance_i_j / R)
                k = M * LQ

                # progress bonus (urgent)
                progress = (dist_i - distance_j) / max_dist  # positive good
                urgent_bonus = 0.0
                if urgent_mode:
                    urgent_bonus = self.urgency_gain * (1.0 - ttl_frac) * progress

                if actual_v >= req_v:
                    candidates.append((node_j, k, miss_risk, urgent_bonus))
                else:
                    candidates2.append((node_j, actual_v, miss_risk, urgent_bonus))

        # routing hole handling
        if len(candidates) == 0:
            if len(candidates2) > 0:
                chosen = None
                best_score = -1e18
                for node_j, actual_v, miss_risk, urgent_bonus in candidates2:
                    score = actual_v - (self.dm_lambda * miss_risk) + urgent_bonus
                    if score > best_score:
                        best_score = score
                        chosen = node_j
                return chosen
            return "RHP"

        # choose best by score
        chosen = None
        best_score = -1e18
        for node_j, k, miss_risk, urgent_bonus in candidates:
            j = node_j.identifier
            Q_value = self.drone.neighbor_table[j, 9]
            score = (Q_value * k) - (self.dm_lambda * miss_risk) + urgent_bonus
            if score > best_score:
                best_score = score
                chosen = node_j

        return chosen
    

    def computeActualVel(self, j, node_j, distance_i):
        x2 = self.drone.neighbor_table[j, 4]
        y2 = self.drone.neighbor_table[j, 5]
        x1 = self.drone.neighbor_table[j, 0]
        y1 = self.drone.neighbor_table[j, 1]

        if (x2 - x1) != 0:
            angle_j = math.atan((y2 - y1) / (x2 - x1))
        else:
            angle_j = 0.0

        delay = self.drone.neighbor_table[j, 8] + self.drone.neighbor_table[j, 11]
        if delay == 0:
            delay = 0.01

        t1 = self.drone.neighbor_table[j, 6]
        t3 = self.simulator.cur_step + delay

        x = x1 + node_j.speed * math.cos(angle_j) * (t3 - t1)
        y = node_j.coords[1] + node_j.speed * math.sin(angle_j) * (t3 - t1)

        distance_j = util.euclidean_distance((x, y), self.simulator.depot_coordinates)
        distance_i_j = util.euclidean_distance(self.drone.coords, (x, y))

        actual_v = (distance_i - distance_j) / delay
        return actual_v, distance_i_j, distance_j, delay
