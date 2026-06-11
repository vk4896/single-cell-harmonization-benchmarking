#!/usr/bin/env python3
"""
Run MNN correction on all cells using an existing embedding (default: obsm['X_pca']).

Integration uses **only** Scanpy's external API: ``scanpy.external.pp.mnn_correct``
(mnnpy backend). Gene-space MNN on millions of cells is not practical; PCs are
used as features (reduced-space correction).

After correction, optionally runs ``sc.pp.neighbors`` + ``sc.tl.umap`` on a
**random subset** and stores coordinates in a **new** obsm matrix of shape
(n_obs, 2) with NaN for cells outside the subset, so existing obsm keys (e.g.
X_umap, X_scvi) are never overwritten. The subset neighbor graph lives only in
the optional sidecar .h5ad (see --save-umap-subset-h5ad), not in the parent
obsp, because obsp must be n_obs × n_obs.

Requires: scanpy, anndata, numpy, pandas, mnnpy (via sc.external.pp.mnn_correct).

**Python version:** ``mnnpy`` often fails to compile on Python 3.10+ (including 3.11
in ``disenv``). Use the project venv (Python 3.8 + working mnnpy)::

    source fastMNN/.venv_mnn38/bin/activate
    python fastmnn_full_obsm.py ...

**Newer .h5ad files** (e.g. ``encoding_type='null'`` under ``uns/log1p/base``) cannot
be read by the old anndata bundled with Python 3.8. If ``read_h5ad`` fails for that
reason, the script uses a **bridge**: a modern Python (``--bridge-python``, or
``FASTMNN_BRIDGE_PYTHON``, defaulting to ``~/disenv/bin/python``) loads the file in
**backed** mode, writes ``embed.npy`` + ``obs_batch.json``, you run MNN in this env,
then the bridge Python **copies** the input .h5ad to the output path and appends new
``obsm`` arrays with ``write_elem`` (so the rest of the file is unchanged).

**SciPy:** ``mnnpy`` calls ``cKDTree.query(..., n_jobs=...)``. Recent SciPy removed
that keyword; use ``scipy<1.10`` (e.g. ``pip install 'numpy==1.22.4' 'scipy==1.7.3'``)
in ``.venv_mnn38`` if you see ``Unexpected keyword argument {'n_jobs': ...}``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore


def _batch_categories(obs_batch: pd.Series) -> list:
    if isinstance(obs_batch.dtype, pd.CategoricalDtype):
        return list(obs_batch.cat.categories)
    return sorted(obs_batch.unique(), key=lambda x: str(x))


def _require_mnnpy() -> None:
    try:
        import mnnpy  # noqa: F401
    except ImportError:
        here = Path(__file__).resolve().parent
        py38 = here / ".venv_mnn38" / "bin" / "python"
        print(
            "mnnpy is not installed (required by scanpy.external.pp.mnn_correct).\n"
            "pip install mnnpy usually fails on Python 3.11+ because the package ships an\n"
            "outdated C extension.\n\n"
            "Use the Python 3.8 environment in this folder:\n"
            f"  source {here / '.venv_mnn38' / 'bin' / 'activate'}\n"
            f"  python {Path(__file__).name}  # same arguments as before\n\n"
            "Or run in one line:\n"
            f"  {py38} {Path(__file__).resolve()} ...\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _is_h5ad_null_read_error(exc: BaseException) -> bool:
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(str(cur).lower())
        cur = cur.__cause__
    blob = " ".join(parts)
    return ("encoding_type" in blob and "null" in blob) or (
        "iospec" in blob and "null" in blob
    )


def _resolve_bridge_python(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("FASTMNN_BRIDGE_PYTHON")
    if env:
        return Path(env)
    candidates = [
        Path.home() / "disenv" / "bin" / "python",
        Path("/home/krupavardhan4869@gmail.com/disenv/bin/python"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find a modern Python for --bridge-python (need scanpy + anndata "
        "that can read your .h5ad). Set FASTMNN_BRIDGE_PYTHON to e.g. disenv's python."
    )


def _h5ad_obsm_keys(path: Path) -> set[str]:
    if h5py is None:
        raise ImportError("h5py is required for bridge mode; pip install h5py")
    with h5py.File(path, "r") as f:
        g = f.get("obsm")
        if g is None:
            return set()
        return set(g.keys())


def _bridge_extract_to_workdir(
    bridge_py: Path,
    h5ad_path: Path,
    workdir: Path,
    batch_key: str,
    use_rep: str,
) -> None:
    """Modern Python: backed read, write embed + batch + obs names as .npy (no uns)."""
    script = r"""
import json
import sys
from pathlib import Path
import numpy as np
import scanpy as sc

path, wdir, batch_key, rep = sys.argv[1:5]
wdir = Path(wdir)
wdir.mkdir(parents=True, exist_ok=True)
a = sc.read_h5ad(path, backed="r")
if batch_key not in a.obs.columns:
    raise SystemExit("obs column %r not found" % (batch_key,))
if rep not in a.obsm:
    raise SystemExit("obsm[%r] not found" % (rep,))
emb = np.asarray(a.obsm[rep])
if emb.shape[0] != a.n_obs:
    raise SystemExit("embedding rows != n_obs")
b = a.obs[batch_key].astype(str).to_numpy()
names = a.obs_names.astype(str).to_numpy()
np.save(wdir / "embed.npy", emb.astype(np.float64, copy=False))
# JSON sidecar: avoids object .npy pickle incompatibility (NumPy 2 writer vs NumPy 1 reader)
(wdir / "obs_batch.json").write_text(
    json.dumps({"obs_names": names.tolist(), "batch": b.tolist()}), encoding="utf-8"
)
print("bridge_extract", emb.shape, flush=True)
"""
    r = subprocess.run(
        [str(bridge_py), "-c", script, str(h5ad_path), str(workdir), batch_key, use_rep],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        raise RuntimeError("bridge extract failed")


def _load_minimal_adata_from_bridge(
    workdir: Path, batch_key: str, use_rep: str
) -> AnnData:
    import json
    import scipy.sparse as sp

    emb = np.load(workdir / "embed.npy")
    meta = json.loads((workdir / "obs_batch.json").read_text(encoding="utf-8"))
    names = meta["obs_names"]
    b = meta["batch"]
    n_obs = emb.shape[0]
    if len(b) != n_obs or len(names) != n_obs:
        raise RuntimeError(
            f"bridge files: length mismatch n_obs={n_obs} names={len(names)} batch={len(b)}"
        )
    obs = pd.DataFrame({batch_key: b}, index=names)
    adata = AnnData(
        X=sp.csr_matrix((n_obs, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["_"]),
    )
    adata.obsm[use_rep] = emb
    return adata


def _merge_obsm_into_h5ad_copy(
    bridge_py: Path,
    src_h5ad: Path,
    dst_h5ad: Path,
    obsm_arrays: dict[str, np.ndarray],
) -> None:
    """Copy src to dst, then append/replace obsm keys using modern anndata IO."""
    if not obsm_arrays:
        shutil.copy2(src_h5ad, dst_h5ad)
        return
    np_dir = dst_h5ad.parent / (dst_h5ad.stem + "_merge_arrays")
    np_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for k, arr in obsm_arrays.items():
        p = np_dir / f"{k.replace('/', '_')}.npy"
        np.save(p, arr)
        paths.extend([k, str(p)])
    script = r"""
import shutil, sys, h5py, numpy as np
from pathlib import Path
try:
    from anndata.io import write_elem
except ImportError:
    from anndata.experimental import write_elem

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
rest = sys.argv[3:]
pairs = list(zip(rest[0::2], rest[1::2]))
shutil.copy2(src, dst)
with h5py.File(dst, "r+") as f:
    g = f.require_group("obsm")
    for key, npy in pairs:
        if key in g:
            del g[key]
        arr = np.load(npy)
        write_elem(g, key, arr)
print("merge_obsm_ok", len(pairs), flush=True)
"""
    cmd = [str(bridge_py), "-c", script, str(src_h5ad), str(dst_h5ad), *paths]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        raise RuntimeError("merge obsm into h5ad copy failed")
    try:
        shutil.rmtree(np_dir)
    except OSError:
        pass


def _batch_pca_blocks(
    adata: AnnData, batch_key: str, use_rep: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (per-batch PCA matrices, per-batch original obs indices) in stable batch order."""
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm['{use_rep}'] not found; available: {list(adata.obsm.keys())}")

    x = adata.obsm[use_rep]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x, dtype=np.float64)
    if x.shape[0] != adata.n_obs:
        raise ValueError(f"{use_rep} has {x.shape[0]} rows but adata has {adata.n_obs} obs")

    bc = adata.obs[batch_key]
    order = _batch_categories(bc)
    blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []

    for b in order:
        idx = np.flatnonzero(np.asarray(bc == b))
        if idx.size == 0:
            continue
        blocks.append(np.ascontiguousarray(x[idx]))
        index_blocks.append(idx)

    return blocks, index_blocks


def _scatter_concatenated(
    corrected_concat: np.ndarray,
    index_blocks: list[np.ndarray],
    n_obs: int,
    n_cols: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Map MNN output (batches concatenated in the same order as index_blocks) back to obs order."""
    out = np.empty((n_obs, n_cols), dtype=dtype)
    offset = 0
    for idx in index_blocks:
        n = idx.size
        out[idx] = corrected_concat[offset : offset + n].astype(dtype, copy=False)
        offset += n
    if offset != corrected_concat.shape[0]:
        raise RuntimeError("MNN output length does not match sum of batch sizes")
    return out


def _sample_obs_indices(
    adata: AnnData,
    batch_key: str,
    n_take: int,
    seed: int,
    stratify: bool,
) -> np.ndarray:
    """Row indices into adata (sorted ascending)."""
    n_take = min(int(n_take), adata.n_obs)
    rng = np.random.default_rng(seed)
    if not stratify or batch_key not in adata.obs.columns:
        idx = rng.choice(adata.n_obs, size=n_take, replace=False)
        return np.sort(idx)

    bc = adata.obs[batch_key]
    cats = pd.Series(bc).astype(str)
    vc = cats.value_counts()
    n_batch = vc.shape[0]
    base = n_take // n_batch
    rem = n_take - base * n_batch
    picked: list[int] = []
    for i, (b, _) in enumerate(vc.items()):
        mask = np.flatnonzero(np.asarray(cats == b))
        n_want = base + (1 if i < rem else 0)
        n_want = min(n_want, mask.size)
        if n_want <= 0:
            continue
        sub = rng.choice(mask, size=n_want, replace=False)
        picked.extend(sub.tolist())
    idx = np.array(picked, dtype=np.int64)
    if idx.size < n_take:
        pool = np.setdiff1d(np.arange(adata.n_obs), idx, assume_unique=False)
        extra = min(n_take - idx.size, pool.size)
        if extra > 0:
            add = rng.choice(pool, size=extra, replace=False)
            idx = np.unique(np.concatenate([idx, add]))
    return np.sort(idx[:n_take])


def _umap_subset_on_corrected(
    adata: AnnData,
    rep_key: str,
    subset_idx: np.ndarray,
    n_neighbors: int,
    n_pcs_cap: int,
    min_dist: float,
    random_state: int,
) -> AnnData:
    """Neighbors + UMAP on adata[subset_idx]; returns the subset AnnData with graph + UMAP."""
    sub = adata[subset_idx].copy()
    emb = sub.obsm[rep_key]
    n_use = min(n_pcs_cap, emb.shape[1])
    sc.pp.neighbors(
        sub,
        use_rep=rep_key,
        n_neighbors=n_neighbors,
        n_pcs=n_use,
        random_state=random_state,
    )
    sc.tl.umap(sub, min_dist=min_dist, random_state=random_state)
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="MNN on PCA for full combined_data; add obsm only.")
    here = Path(__file__).resolve().parent
    default_in = here.parent / "combined_data.h5ad"
    default_out_dir = here / "results"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_in,
        help=f"Input .h5ad (default: {default_in})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .h5ad path (default: results/<input_stem>_with_mnn_obsm.h5ad)",
    )
    parser.add_argument("--batch-key", type=str, default="batch")
    parser.add_argument(
        "--use-rep",
        type=str,
        default="X_pca",
        help="obsm key to correct (must already exist, e.g. X_pca)",
    )
    parser.add_argument(
        "--obsm-out",
        type=str,
        default="X_mnn",
        help="New obsm key for corrected embedding (must not already exist)",
    )
    parser.set_defaults(var_adj=False, cos_norm_in=True, cos_norm_out=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--var-adj",
        dest="var_adj",
        action="store_true",
        help="Variance adjustment (very slow on millions of cells; default: off)",
    )
    parser.add_argument(
        "--no-cos-norm-in",
        dest="cos_norm_in",
        action="store_false",
        help="Disable cosine normalization before distances",
    )
    parser.add_argument(
        "--no-cos-norm-out",
        dest="cos_norm_out",
        action="store_false",
        help="Disable cosine normalization on corrected output",
    )
    parser.add_argument("--svd-dim", type=int, default=None, help="Optional biological subspace dims")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional random subsample (seed fixed) for debugging only",
    )
    parser.add_argument(
        "--umap-subset-n",
        type=int,
        default=None,
        help="If set, neighbors+UMAP on this many cells after MNN; stored as new obsm (NaN-padded)",
    )
    parser.add_argument("--umap-subset-seed", type=int, default=42)
    parser.add_argument(
        "--umap-stratify-batch",
        action="store_true",
        help="Stratify UMAP subset across batch categories (approx. equal per batch)",
    )
    parser.add_argument(
        "--umap-key",
        type=str,
        default="X_umap_mnn",
        help="New obsm key for subset UMAP (n_obs × 2, NaN outside subset); must not exist",
    )
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument(
        "--umap-n-pcs-cap",
        type=int,
        default=50,
        help="Max dimensions from corrected embedding passed to neighbors",
    )
    parser.add_argument("--umap-min-dist", type=float, default=0.5)
    parser.add_argument("--umap-random-state", type=int, default=0)
    parser.add_argument(
        "--save-umap-subset-h5ad",
        type=Path,
        default=None,
        help="Write subset AnnData with real neighbors obsp + X_umap (default: results/<stem>_mnn_umap_subset.h5ad when UMAP runs)",
    )
    parser.add_argument(
        "--bridge-python",
        type=Path,
        default=None,
        help="Python 3.10+ env with scanpy/anndata that can read the .h5ad (bridge mode; default: disenv)",
    )
    parser.add_argument(
        "--keep-bridge-workdir",
        action="store_true",
        help="Do not delete the temporary bridge directory after a successful run",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        alt = Path("/home/krupavardhan4869@gmail.com/combined_data.h5ad")
        if alt.is_file():
            print(f"Input not found at {args.input}, using {alt}", file=sys.stderr)
            args.input = alt
        else:
            print(f"Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

    out_dir = default_out_dir if args.output is None else args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output
        if args.output is not None
        else out_dir / f"{args.input.stem}_with_mnn_obsm.h5ad"
    )

    need_umap = args.umap_subset_n is not None and args.umap_subset_n > 0
    workdir: Path | None = None
    bridge_py: Path | None = None
    bridged = False

    def _check_obsm_collisions(existing: set[str]) -> None:
        if args.obsm_out in existing:
            print(
                f"obsm['{args.obsm_out}'] already exists in the input file; "
                "use another --obsm-out.",
                file=sys.stderr,
            )
            sys.exit(1)
        if need_umap and args.umap_key in existing:
            print(
                f"obsm['{args.umap_key}'] already exists; use another --umap-key.",
                file=sys.stderr,
            )
            sys.exit(1)

    if h5py is not None:
        _check_obsm_collisions(_h5ad_obsm_keys(args.input))
    else:
        print(
            "Warning: h5py not installed; cannot pre-check obsm keys or use bridge merge.",
            file=sys.stderr,
        )

    print(f"Loading {args.input} ...")
    try:
        adata = sc.read_h5ad(args.input)
    except Exception as e:
        if not _is_h5ad_null_read_error(e):
            raise
        if h5py is None:
            print(
                "Install h5py in this environment, then retry (required for bridge mode).",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            bridge_py = _resolve_bridge_python(args.bridge_python)
        except FileNotFoundError as fnf:
            print(str(fnf), file=sys.stderr)
            sys.exit(1)
        workdir = Path(tempfile.mkdtemp(prefix="fastmnn_bridge_", dir=str(out_dir)))
        print(
            f"Direct read failed (newer .h5ad / null encoding in uns). "
            f"Bridge extract via {bridge_py} -> {workdir}"
        )
        _bridge_extract_to_workdir(
            bridge_py, args.input, workdir, args.batch_key, args.use_rep
        )
        adata = _load_minimal_adata_from_bridge(workdir, args.batch_key, args.use_rep)
        bridged = True
    else:
        if h5py is None:
            _check_obsm_collisions(set(adata.obsm.keys()))

    if args.max_cells is not None and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(42)
        ix = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        adata = adata[ix].copy()
        print(f"DEBUG: subsampled to {adata.n_obs} cells")

    if args.batch_key not in adata.obs.columns:
        print(f"obs column '{args.batch_key}' not found.", file=sys.stderr)
        sys.exit(1)

    blocks, index_blocks = _batch_pca_blocks(adata, args.batch_key, args.use_rep)
    if len(blocks) < 2:
        print("Need at least two batches for MNN; copying use_rep to output key.")
        src = adata.obsm[args.use_rep]
        src = src.toarray() if hasattr(src, "toarray") else np.asarray(src)
        adata.obsm[args.obsm_out] = np.asarray(src, dtype=np.float32)
    else:
        n_pcs = blocks[0].shape[1]
        var_index = [f"PC{i + 1}" for i in range(n_pcs)]

        print(
            f"MNN on {args.use_rep}: {adata.n_obs} cells, {len(blocks)} batches, "
            f"{n_pcs} dimensions, k={args.k}, var_adj={args.var_adj}"
        )
        print("This can take a long time and use a lot of memory for very large batches.")

        _require_mnnpy()
        try:
            corrected_concat, _, _ = sc.external.pp.mnn_correct(
                *blocks,
                var_index=var_index,
                k=args.k,
                sigma=args.sigma,
                cos_norm_in=args.cos_norm_in,
                cos_norm_out=args.cos_norm_out,
                svd_dim=args.svd_dim,
                var_adj=args.var_adj,
                compute_angle=False,
                do_concatenate=True,
                n_jobs=args.n_jobs,
            )
        except ImportError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        corrected_concat = np.asarray(corrected_concat)
        out_dtype = np.float32
        adata.obsm[args.obsm_out] = _scatter_concatenated(
            corrected_concat, index_blocks, adata.n_obs, n_pcs, out_dtype
        )

    if args.umap_subset_n is not None and args.umap_subset_n > 0:
        print(
            f"\nSubset neighbors + UMAP (Scanpy) on {args.umap_subset_n} cells, "
            f"rep={args.obsm_out!r}, new obsm key={args.umap_key!r} ..."
        )
        u_idx = _sample_obs_indices(
            adata,
            args.batch_key,
            args.umap_subset_n,
            args.umap_subset_seed,
            args.umap_stratify_batch,
        )
        sub = _umap_subset_on_corrected(
            adata,
            rep_key=args.obsm_out,
            subset_idx=u_idx,
            n_neighbors=args.umap_n_neighbors,
            n_pcs_cap=args.umap_n_pcs_cap,
            min_dist=args.umap_min_dist,
            random_state=args.umap_random_state,
        )
        umap_sub = np.asarray(sub.obsm["X_umap"])
        n_dim = umap_sub.shape[1]
        full_umap = np.full((adata.n_obs, n_dim), np.nan, dtype=np.float32)
        full_umap[u_idx] = umap_sub.astype(np.float32, copy=False)
        adata.obsm[args.umap_key] = full_umap

        meta = {
            "rep_key": args.obsm_out,
            "subset_indices": u_idx.astype(np.int64),
            "n_neighbors": args.umap_n_neighbors,
            "n_pcs_used": min(args.umap_n_pcs_cap, adata.obsm[args.obsm_out].shape[1]),
            "min_dist": args.umap_min_dist,
            "random_state": args.umap_random_state,
            "stratify_batch": bool(args.umap_stratify_batch),
        }
        if bridged:
            print(
                "Note: bridge mode does not write uns['mnn_umap_subset'] into the output .h5ad "
                "(subset indices are only in the subset sidecar if saved).",
                file=sys.stderr,
            )
        elif "mnn_umap_subset" in adata.uns:
            print("Warning: uns['mnn_umap_subset'] exists; not overwriting.", file=sys.stderr)
        else:
            adata.uns["mnn_umap_subset"] = meta

        subset_path = args.save_umap_subset_h5ad
        if subset_path is None:
            subset_path = out_dir / f"{args.input.stem}_mnn_umap_subset.h5ad"
        print(f"Writing subset AnnData (neighbors + UMAP graph) to {subset_path} ...")
        sub.write(subset_path)

    print(f"Writing {out_path} ...")
    merged_keys: list[str] = []
    if bridged:
        assert bridge_py is not None
        to_merge = {args.obsm_out: np.asarray(adata.obsm[args.obsm_out])}
        if need_umap and args.umap_key in adata.obsm:
            to_merge[args.umap_key] = np.asarray(adata.obsm[args.umap_key])
        merged_keys = list(to_merge.keys())
        _merge_obsm_into_h5ad_copy(bridge_py, args.input, out_path, to_merge)
    else:
        adata.write(out_path)

    if workdir is not None and not args.keep_bridge_workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    print("Done.")
    print(f"obsm keys in memory: {list(adata.obsm.keys())}")
    if bridged:
        print(f"(Merged into copy of input; new obsm keys: {merged_keys})")


if __name__ == "__main__":
    main()
