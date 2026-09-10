import json
import numpy as np
from scipy import stats
d = json.load(open("c:/code/UAV/results/n100/eval.json"))
sca = np.array(d["by_method"]["sca"]["per_seed_Mbps"])
for m in ["random","kmeans","pso"]:
    other = np.array(d["by_method"][m]["per_seed_Mbps"])
    delta = sca - other
    w, p = stats.wilcoxon(delta)
    print(m, "mean_delta", delta.mean(), "std", delta.std(ddof=1), "win%", (delta>0).mean()*100, "wilcoxon_p", p)
