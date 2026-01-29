"""
simulator_interface.py
======================

Interface to the digital-twin power-grid simulator (pandapower) with transparent caching.

What this module provides
-------------------------
- Robust relative path resolution to the JSON grid model (digital twin).
- `query_simulator(sample)`:
    Runs base-case + N-1 contingency analysis and returns "secure" / "insecure".
- `query_simulator_cached(sample)`:
    Same as above, but with an LRU cache to avoid recomputing identical inputs.

Key notes
---------
- Inputs are expected as a flat dict with keys like:
    "load_{i}_p_mw", "gen_{i}_p_mw", "sgen_{i}_p_mw".
- Constraints are considered violated if:
    * Any line loading (except index 45) exceeds 100%, or
    * Any bus voltage < 0.9 pu, or > 1.1 pu.
- N-1 contingencies:
    * Each line (except 45) is switched out, one at a time.
    * Each generator is switched out, one at a time.
- All simulator calls are pure functions of `sample` and thus safely cacheable.
"""

import os
import copy
from functools import lru_cache

import pandapower as pp

def _resolve_net_path() -> str:
    """
    Build an absolute path to the JSON grid model (relative to this file).
    Raises:
        FileNotFoundError if the JSON file is missing.

    Returns:
        Absolute filesystem path to the digital-twin JSON.
    """

    # Start from current file location → ../data/digital_twin_ext_grid.json
    here = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(here, "..", "data", "digital_twin_ext_grid.json")
    net_path = os.path.abspath(net_path)

    if not os.path.isfile(net_path):
        raise FileNotFoundError(f"Net JSON not found at {net_path}")

    return net_path

# Load a template network once. Each simulation run deep-copies this template.
net_template = pp.from_json(_resolve_net_path())

# Precompute contingency sets for N-1 analysis
# Lines: exclude index 45 (domain-specific exception, e.g., a special tie-line)
contingency_lines = [idx for idx in net_template.line.index if idx != 45]
# Generators: all indices
contingency_gens = net_template.gen.index.tolist()


def violates_constraints(net) -> bool:
    """"
    Check whether the solved network violates operational constraints.

    Constraints considered:
      - Line thermal loading must be ≤ 100% (line 45 excluded from the check).
      - Bus voltage must be within [0.9, 1.1] pu.

    Args:
        net: pandapower network after a successful power flow (runpp).

    Returns:
        True if any constraint is violated; otherwise False.
    """

    # Max loading excluding line 45
    max_line_loading = net.res_line.loc[net.res_line.index != 45, "loading_percent"].max()
    # Voltage limits across all buses
    min_voltage = net.res_bus.vm_pu.min()
    max_voltage = net.res_bus.vm_pu.max()
    return (max_line_loading > 100) or (min_voltage < 0.9) or (max_voltage > 1.1)


def query_simulator(sample: dict) -> str:
    """
    Run base-case + N-1 simulations for a single operating point.

    Steps:
      1) Deep-copy the static network template.
      2) Populate loads, gens, sgens from `sample`.
      3) Solve base-case. If it fails or violates constraints → "insecure".
      4) For each line outage (except 45): solve. Any failure/violation → "insecure".
      5) For each generator outage: solve. Any failure/violation → "insecure".
      6) If all pass → "secure".

    Args:
        sample: dict with keys like "load_{i}_p_mw", "gen_{i}_p_mw", "sgen_{i}_p_mw".

    Returns:
        "secure" or "insecure".
    """

    # Start from a clean copy so successive calls are independent
    net = copy.deepcopy(net_template)

    # Set operating point from the sample dict (default 0.0 if missing)
    for i in net.load.index:
        net.load.at[i, "p_mw"] = sample.get(f"load_{i}_p_mw", 0.0)

    for i in net.gen.index:
        net.gen.at[i, "p_mw"] = sample.get(f"gen_{i}_p_mw", 0.0)

    for i in net.sgen.index:
        net.sgen.at[i, "p_mw"] = sample.get(f"sgen_{i}_p_mw", 0.0)

    # --- Base-case ---
    try:
        pp.runpp(net) # <-- this is the simulator call (pandapower power flow)
    except Exception:
        # Power flow did not converge → insecure
        return "insecure"

    if violates_constraints(net):
        return "insecure"

    # --- N-1: line outages ---
    for line_idx in contingency_lines:
        net_copy = copy.deepcopy(net)
        net_copy.line.at[line_idx, "in_service"] = False
        try:
            pp.runpp(net_copy) # <-- simulator call for a single line outage
            if violates_constraints(net_copy):
                return "insecure"
        except Exception:
            return "insecure"

    # --- N-1: generator outages ---
    for gen_idx in contingency_gens:
        net_copy = copy.deepcopy(net)
        net_copy.gen.at[gen_idx, "in_service"] = False
        try:
            pp.runpp(net_copy) # <-- simulator call for a single generator outage
            if violates_constraints(net_copy):
                return "insecure"
        except Exception:
            return "insecure"

    return "secure"


# ---------- Caching layer ----------

# This layer remembers past simulator results (caching) so repeated inputs don't need to be recomputed.
# NOTE: lru_cache keeps results only in memory (RAM) during program execution.
# The cache is cleared once the program ends or the computer restarts.

# (TODO future): If persistent caching is needed across program runs,
# implement disk-based storage (e.g., pickle, joblib.Memory, or SQLite).

# ---------------------------------------------------------------------------
# How the caching flow works (example)
#
# 1) User calls: query_simulator_cached({"load_0_p_mw": 50.0000001, "gen_0_p_mw": 20.0})
# 2) _cache_key turns the dict into a normalized tuple:
#       (('gen_0_p_mw', 20.0), ('load_0_p_mw', 50.0))
#    - values are rounded to 6 decimals (so 50.0000001 → 50.0)
#    - keys are sorted so order does not matter
# 3) _cached_query_simulator_by_key is called with that tuple key
#    - On the first call: runs query_simulator(sample) → e.g. "secure"
#      and stores it in the LRU cache under that key.
#    - On later calls with the same inputs: returns the cached "secure"
#      instantly, without running the simulator again.
#
# In short: the simulator is only executed once per unique input.
# ---------------------------------------------------------------------------

def _cache_key(sample: dict) -> tuple:
    """
    Normalize a sample dict into a hashable, order-independent key for caching.

    Rules:
      - Convert numeric values to float and round to 6 decimals (reduces duplicates
        caused by tiny float noise).
      - Keep non-numeric values as-is.
      - Sort by key name to make the tuple order-independent.

    Args:
        sample: input dict passed to the simulator.

    Returns:
        A tuple of (key, value) pairs suitable for use as an LRU cache key.
    """

    items = []
    for k, v in sample.items():
        try:
            vf = float(v)
            items.append((k, round(vf, 6)))
        except Exception:
            items.append((k, v))
    return tuple(sorted(items))


# Use LRU cache: remember up to 50,000 past results and reuse them instead of re-running the simulator.
@lru_cache(maxsize=50000)
def _cached_query_simulator_by_key(key: tuple) -> str:
    """
    Cached backend: accepts a normalized tuple key, reconstructs the dict,
    runs the simulator, and caches the result.
    """

    sample = {k: v for k, v in key}
    return query_simulator(sample)


def query_simulator_cached(sample: dict) -> str:
    """
    Public cached entry point. Use this in your code instead of `query_simulator`
    when repeated inputs are expected.

    Args:
        sample: input dict for the operating point.

    Returns:
        "secure" or "insecure" (possibly served from cache).
    """

    return _cached_query_simulator_by_key(_cache_key(sample))


if __name__ == "__main__":
    # Minimal sanity check
    sample = {"load_0_p_mw": 50.0, "gen_0_p_mw": 20.0, "sgen_0_p_mw": 5.0}
    print("Simulation result:", query_simulator_cached(sample))
