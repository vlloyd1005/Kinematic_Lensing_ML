
# -----------------------------------------------------------
# How to fill in real subhalo_ids
# -----------------------------------------------------------
# For each (sim, snap) in your plan, query the TNG API for
# star-forming disc galaxies in the KL mass range.
#
# Use --offset to start from a different position in the ranked list,
# e.g. to add galaxies 500.
# The TNG API supports the `offset` parameter natively.
#
# Example (Python, using the TNG API):

import requests, pandas as pd

API_KEY = "16a29db7f934e4d33640dcd47e7f80be"
headers = {"api-key": API_KEY}

def get_subhalos(sim, snap, mass_min_log=9.5, mass_max_log=11.0,
                 sfr_min=0.1, limit=100, offset=0):
    # Filters matching Xu+2022 §3.2 KL selection:
    #   mass range  : 10^9.5 – 10^11.0 Msun.  Upper limit 11.0 not 11.5
    #                 because very massive galaxies at z~1 are mostly quenched.
    #   sfr__gt=0.1 : exclude quenched galaxies; ensures detectable Hα emission
    #                 for Roman grism, matching the paper's detection criterion.
    #   order_by=-sfr: most actively star-forming discs first so the first N
    #                  results have the best-resolved velocity fields.
    #   offset      : skip the first N results — use this to fetch galaxies
    #                 200-300 without re-downloading the first 200.
    h = 0.6774
    url = f"https://www.tng-project.org/api/{sim}/snapshots/{snap}/subhalos/"
    params = {
        "limit":          limit,
        "offset":         offset,
        "mass_stars__gt": 10**(mass_min_log - 10) / h,
        "mass_stars__lt": 10**(mass_max_log - 10) / h,
        "sfr__gt":        sfr_min,
        "order_by":       "-sfr",
    }
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    return [entry["id"] for entry in r.json().get("results", [])]

# Fill IDs into the plan CSV.
# Set OFFSET to start from a different position in the ranked list,
# e.g. OFFSET=200 to get galaxies 200-300 after already having 0-199.
OFFSET = 0   # ← change this when adding more galaxies

plan = pd.read_csv("dataset_plan_to500.csv")
for (sim, snap), grp in plan.groupby(["sim", "snap"]):
    n_needed = len(grp)
    ids = get_subhalos(sim, snap, limit=n_needed, offset=OFFSET)
    if len(ids) < n_needed:
        print(f"WARNING: only {len(ids)} galaxies returned for {sim} snap {snap} "
              f"at offset {OFFSET}, needed {n_needed}. "
              f"Try a smaller offset or lower mass cut.")
        ids = ids + [-1] * (n_needed - len(ids))
    plan.loc[grp.index, "subhalo_id"] = ids[:n_needed]

# Drop any rows where the API returned nothing
plan = plan[plan["subhalo_id"] != -1].reset_index(drop=True)
print(f"Final plan: {len(plan)} rows")
plan.to_csv("dataset_plan_to500.csv", index=False)

