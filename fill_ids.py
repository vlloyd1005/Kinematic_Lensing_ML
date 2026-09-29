
# -----------------------------------------------------------
# How to fill in real subhalo_ids
# -----------------------------------------------------------
# For each (sim, snap) in your plan, query the TNG API for
# star-forming disc galaxies in the KL mass range.
#
# The key filters (matching Xu+2022 §3.2 KL sample):
#   - Stellar mass  1e9 < M* < 1e12  Msun  (log M* ≈ 9–12)
#   - SubhaloFlag == 1  (genuine galaxy, not spurious)
#   - Sufficient star particles (proxy for disc size)
#
# Example (Python, using the TNG API):

import requests, pandas as pd

API_KEY = "16a29db7f934e4d33640dcd47e7f80be"
headers = {"api-key": API_KEY}

def get_subhalos(sim, snap, mass_min_log=9.5, mass_max_log=11.0,
                 sfr_min=0.1, limit=500):
    
    # Query star-forming disc galaxies for the KL sample.

    # Filters (matching Xu+2022 §3.2 KL selection):
    #   mass range   : 10^9.5 – 10^11.0 Msun. Upper limit is 11.0 not 11.5
    #                  because very massive galaxies at z~1 are predominantly
    #                  quenched and would not be Hα emitters in the Roman KL sample.
    #   sfr__gt=0.1  : FIX 3 — exclude quenched galaxies. Without this, ordering
    #                  by mass picks up passive ellipticals first. The SFR cut
    #                  ensures every selected galaxy has detectable Hα emission,
    #                  matching the Roman grism detection criterion of the paper.
    #   order_by=-sfr: prioritise the most actively star-forming discs so the
    #                  first N results have the best-resolved velocity fields.
   
    h = 0.6774
    url = f"https://www.tng-project.org/api/{sim}/snapshots/{snap}/subhalos/"
    params = {
        "limit":          limit,
        "mass_stars__gt": 10**(mass_min_log - 10) / h,
        "mass_stars__lt": 10**(mass_max_log - 10) / h,
        "sfr__gt":        sfr_min,      # exclude quenched / passive galaxies
        "order_by":       "-sfr",       # most actively star-forming first
    }
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    return [entry["id"] for entry in r.json().get("results", [])]

# Fill IDs into the plan CSV:
plan = pd.read_csv("dataset_plan.csv")
for (sim, snap), grp in plan.groupby(["sim", "snap"]):
    ids = get_subhalos(sim, snap, limit=len(grp))
    if len(ids) < len(grp):
        ids += ids * (len(grp) // len(ids) + 1)   # cycle if not enough
    plan.loc[grp.index, "subhalo_id"] = ids[:len(grp)]
plan.to_csv("dataset_plan_with_ids.csv", index=False)

