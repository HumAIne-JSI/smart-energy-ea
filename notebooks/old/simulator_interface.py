# simulator_interface.py

import copy
from functools import lru_cache

import pandapower as pp
import numpy as np
import pandas as pd

# Naloži statično topologijo (relativno na projekt)
net_template = pp.from_json("data/digital_twin_ext_grid.json")

# Določi komponente za N-1 (po analogiji)
contingency_lines = [idx for idx in net_template.line.index if idx != 45]
contingency_gens = net_template.gen.index.tolist()


def violates_constraints(net):
    """Pomožna funkcija: preveri, če pride do kršitev napetosti ali obremenitev."""
    max_line_loading = net.res_line.loc[net.res_line.index != 45, "loading_percent"].max()
    min_voltage = net.res_bus.vm_pu.min()
    max_voltage = net.res_bus.vm_pu.max()
    return (max_line_loading > 100) or (min_voltage < 0.9) or (max_voltage > 1.1)


def query_simulator(sample: dict) -> str:
    """
    Izvede simulacijo za en sam vzorec (en časovni korak).
    Vrne 'secure' ali 'insecure' glede na N-1 preverjanje.

    sample: dict z značilkami (ključ: feature_name, vrednost: float)
    """
    # 1. Ustvari kopijo osnovnega omrežja
    net = copy.deepcopy(net_template)

    # 2. Nastavi vhodne vrednosti
    for i in net.load.index:
        net.load.at[i, "p_mw"] = sample.get(f"load_{i}_p_mw", 0.0)

    for i in net.gen.index:
        net.gen.at[i, "p_mw"] = sample.get(f"gen_{i}_p_mw", 0.0)

    for i in net.sgen.index:
        net.sgen.at[i, "p_mw"] = sample.get(f"sgen_{i}_p_mw", 0.0)

    # 3. Base-case
    try:
        pp.runpp(net)
    except Exception:
        return "insecure"

    if violates_constraints(net):
        return "insecure"

    # 4. N-1: lines
    for line_idx in contingency_lines:
        net_copy = copy.deepcopy(net)
        net_copy.line.at[line_idx, "in_service"] = False
        try:
            pp.runpp(net_copy)
            if violates_constraints(net_copy):
                return "insecure"
        except Exception:
            return "insecure"

    # 5. N-1: generators
    for gen_idx in contingency_gens:
        net_copy = copy.deepcopy(net)
        net_copy.gen.at[gen_idx, "in_service"] = False
        try:
            pp.runpp(net_copy)
            if violates_constraints(net_copy):
                return "insecure"
        except Exception:
            return "insecure"

    return "secure"


# ---------- Caching plast ----------

def _cache_key(sample: dict) -> tuple:
    """Deterministični ključ za cache: (k, zaokrožena_vrednost) po abecedi."""
    items = []
    for k, v in sample.items():
        try:
            vf = float(v)
            items.append((k, round(vf, 6)))
        except Exception:
            items.append((k, v))
    return tuple(sorted(items))


@lru_cache(maxsize=50000)
def _cached_query_simulator_by_key(key: tuple) -> str:
    sample = {k: v for k, v in key}
    return query_simulator(sample)


def query_simulator_cached(sample: dict) -> str:
    """Cache-wrapper nad query_simulator()."""
    return _cached_query_simulator_by_key(_cache_key(sample))