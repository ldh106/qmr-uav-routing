import src.utilities.utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing


class RandomRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
    def feedback(self, *args, **kwargs):
        return
    def relay_selection(self, opt_neighbors, pkd=None):
        """
        This function returns a random relay for packets.

        @param opt_neighbors: a list of tuples (hello_packet, drone) OR a list of drones
        @param pkd: packet descriptor / packet (unused in random routing, kept for BASE compatibility)
        @return: a random drone as relay
        """

        if opt_neighbors is None or len(opt_neighbors) == 0:
            return None

        first = opt_neighbors[0]
        if isinstance(first, tuple):
            candidates = [v[1] for v in opt_neighbors]
        else:
            candidates = opt_neighbors

        return self.simulator.rnd_routing.choice(candidates)
