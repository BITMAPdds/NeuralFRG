import os
import numpy as np
from time import time
import h5py
import argparse


def main(path: str, pct: float, verbose: bool):

    if verbose:
        start = time()

    folder, filename = os.path.split(path)

    if filename.endswith(".h5"):
        extension = ".h5"
    elif filename.endswith(".hdf5"):
        extension = ".hdf5"
    else:
        raise RuntimeError(f"Cannot recognize file extension: {filename}")

    with h5py.File(path, "r") as f:
        vertices = np.array(f["vertices"])
        times = np.array(f["times"])
        couplings = np.array(f["couplings"])
        masks = np.array(f["masks"], dtype=bool)

    if verbose:
        print("Data read in from", path, flush=True)

    n_trajectories = vertices.shape[0]
    n_train = int(pct * n_trajectories)
    i = np.random.choice(np.arange(n_trajectories), size=n_train, replace=False)
    i.sort()
    i_ = np.setdiff1d(np.arange(n_trajectories), i)

    train_path = os.path.join(folder, filename.replace(extension, "_train" + extension))
    test_path = os.path.join(folder, filename.replace(extension, "_test" + extension))

    if verbose:
        print("Train data points:", *i)
        print("Test data points:", *i_, flush=True)

    with h5py.File(train_path, "w") as f:
        f["vertices"] = vertices[i]
        f["times"] = times
        f["couplings"] = couplings[i]
        f["masks"] = masks[i]

    if verbose:
        print("Training set written to:", train_path, flush=True)

    with h5py.File(test_path, "w") as f:
        f["vertices"] = vertices[i_]
        f["times"] = times
        f["couplings"] = couplings[i_]
        f["masks"] = masks[i_]

    if verbose:
        print("Test set written to:", test_path)
        print(f"Time elapsed: {(time()-start):.2f} s", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="Data file path")
    parser.add_argument(
        "--pct", type=float, default=0.7, help="Percentage of data to use for training"
    )
    parser.add_argument("--verbose", help="Increase output verbosity", action="store_true")
    args = parser.parse_args()

    main(args.path, args.pct, args.verbose)
