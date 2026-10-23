import pandas as pd, matplotlib.pyplot as plt
from data import load_speed, load_graph, ROOT
from closure_exp import neighbor_sets

df = load_speed(); _, _, A = load_graph(); up, dn, _ = neighbor_sets(A)
ev = pd.read_csv(ROOT + 'results/events_test.csv').sample(30, random_state=0)
for k, e in enumerate(ev.itertuples()):
    s, c = e.onset_row, e.sensor
    w = df.iloc[s - 12:s + 12]
    plt.figure(figsize=(8, 3))
    for i in up[c]: plt.plot(w.index, w[i], 'b', alpha=.4)
    for j in dn[c]: plt.plot(w.index, w[j], 'r', alpha=.4)
    plt.plot(w.index, w[c], 'k', lw=2)
    plt.axvline(df.index[s], ls='--')
    plt.title(f'event {k}, sensor {c}: black = sensor, blue = upstream, red = downstream')
    plt.savefig(ROOT + f'results/check_{k}.png'); plt.close()
print('saved 30 plots in results/')
