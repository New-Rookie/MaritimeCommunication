"""Quick smoke test to verify all modules import and the simulation runs."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    # Test env imports
    from env.config import SATELLITE_CONFIG, NODE_CONFIGS, SINR_THRESHOLD_DB
    print("[OK] env.config")

    from env.nodes import SatelliteNode, UAVNode, ShipNode, BuoyNode, BaseStationNode
    print("[OK] env.nodes")

    from env.channel.path_loss import fspl_db, two_ray_maritime_db, air_to_ground_db
    print("[OK] env.channel.path_loss")

    from env.channel.interference import calculate_sinr_db
    print("[OK] env.channel.interference")

    from env.network.topology import TopologyManager
    from env.network.mec import MECManager, Task
    print("[OK] env.network")

    from env.ocean_env import OceanEnv
    print("[OK] env.ocean_env")

    # Quick simulation
    env = OceanEnv(render_mode="none")
    obs, info = env.reset(seed=42)
    n_nodes = info["num_nodes"]
    n_edges = info["num_edges"]
    print(f"[OK] reset: {n_nodes} nodes, {n_edges} edges")

    for i in range(5):
        obs, r, done, trunc, info = env.step()
    n_edges = info["num_edges"]
    print(f"[OK] 5 steps: {n_edges} edges, {info['pending_tasks']} tasks")
    env.close()

    # Path loss formulas
    pl1 = fspl_db(100000, 2.4e9)
    print(f"[OK] FSPL(100km, 2.4GHz) = {pl1:.1f} dB")
    pl2 = two_ray_maritime_db(30000, 15, 2, 2.4e9)
    print(f"[OK] TwoRay(30km) = {pl2:.1f} dB")
    pl3 = air_to_ground_db(10000, 200, 2.4e9)
    print(f"[OK] A2G(10km, 200m) = {pl3:.1f} dB")

    # Test P1 imports
    from P1.mechanisms.indp import INDPMechanism
    from P1.mechanisms.disco import DiscoMechanism
    from P1.mechanisms.aloha import ALOHAMechanism
    print("[OK] P1 mechanisms")

    from P1.nd_env import NeighborDiscoveryEnv
    print("[OK] P1 nd_env")

    # Quick INDP test
    indp = INDPMechanism()
    env2 = OceanEnv(render_mode="none")
    env2.reset(seed=42)
    discovered, energy = indp.discover(env2.nodes, noise_factor=2.0)
    n_disc = sum(len(v) for v in discovered.values())
    print(f"[OK] INDP discover: {n_disc} links found, energy={energy:.4f}J")
    env2.close()

    # Test P2 imports
    from P2.ls_env import LinkSelectionEnv
    from P2.models.gcn import GCNEncoder
    print("[OK] P2 models + env")

    # Test P3 imports
    from P3.rm_env import ResourceManagementEnv
    from P3.models.imatd3 import IMATD3Agent
    from P3.models.matd3 import MATD3Agent
    print("[OK] P3 models + env")

    # Test torch
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] PyTorch device: {device}")

    print("\n=== ALL SMOKE TESTS PASSED ===")

if __name__ == "__main__":
    main()
