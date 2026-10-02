import h5py, numpy as np, sys
f = h5py.File(sys.argv[1], "r")
print("keys", list(f.keys()))
print("X", f["X"].attrs.get("encoding-type"), {k: f["X"][k].shape for k in f["X"].keys()} if isinstance(f["X"], h5py.Group) else f["X"].shape)
print("obsm", {k: (f["obsm"][k].shape if isinstance(f["obsm"][k], h5py.Dataset) else list(f["obsm"][k].keys())) for k in f["obsm"].keys()})
print("obs cols", list(f["obs"].keys()))
for c in f["obs"].keys():
    g = f["obs"][c]
    if isinstance(g, h5py.Group) and "categories" in g:
        cats = g["categories"][:]; codes = g["codes"][:]
        cats = [x.decode() if isinstance(x, bytes) else x for x in cats]
        u, n = np.unique(codes, return_counts=True)
        print(c, "ncat", len(cats), "first", cats[:6], "top", [(cats[i], int(k)) for i, k in sorted(zip(u, n), key=lambda t: -t[1])[:4]])
var = f["var"]; print("var cols", list(var.keys()))
idx = var[var.attrs["_index"]][:10]; print("var index", idx)
print("uns", list(f["uns"].keys()) if "uns" in f else None)
