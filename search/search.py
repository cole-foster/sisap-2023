'''
    Cole Foster
    July 11th, 2023

    SISAP Indexing Challenge
'''
import argparse
import hnswlib_modified
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(kind, size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }
    
    for version, url in task.items():
        download(url, os.path.join("data", kind, size, f"{version}.h5"))

def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()



'''
    MAIN SCRIPT: 
        - download/load dataset
        - construct [modified] hnsw index
        - knn search, save results
'''
method = "HNSW"
def run(kind, key, size, M, ef_construction):
    print(f"Running {method} on {kind}-{size}")
    index_identifier = f"{method}-M-{M}-EFC-{ef_construction}"
    
    #> Load Dataset- download if necessary
    prepare(kind, size)
    data = np.ascontiguousarray(h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key],dtype=np.float32) 
    queries = np.array(h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key],dtype=np.float32)
    n, d = data.shape
    ids = np.arange(n) # element ids
    print("Printing Array Information: Ensure Contiguous Array")
    print(data.flags)
    if not data.flags.c_contiguous:
        print("Error: Not a contiguous array- this is needed for C++")
        return

    #> Initialize Index, Normalize Data
    if kind.startswith("clip768"):
        print(f"Initializing Index with Cosine Distance, D={d}")
        index = hnswlib_modified.Index(space='ip', dim=d) # possible options are l2, cosine or ip
        print("Normalizing the Vectors")
        for i in range(n):
            data[i,:] /= np.linalg.norm(data[i,:]) # prevent making a whole copy
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    else:
        raise Exception(f"unsupported input type {kind}")
    
    # Train the index
    print(f"Training index on {data.shape}")
    start_time = time.time()
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M, random_seed=10)
    index.add_items(data, ids)
    elapsed_build = time.time() - start_time
    print(f"Done training in {elapsed_build:.6} (s)")

    #> Searching on the index
    ef_vec = [10, 20, 30, 50, 70, 100, 140, 190, 250, 320, 400, 500, 650, 800, 1000]
    for ef in ef_vec:
        print(f"Searching with ef={ef}")
        start = time.time()
        index.set_ef(ef)  # ef should always be > k
        labels, distances = index.knn_query(queries, k=10)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({index_identifier}),query=(ef={ef})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), index_identifier, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        default="300K"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--EF",
        type=int,
        default=800,
    )
    args = parser.parse_args()
    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    print("Running Script With:")
    print(f"  * Dataset=clip768v2")
    print(f"  * Size={args.size}")
    print(f"  * M={args.M}                  | HNSW Parameter M")
    print(f"  * EFC={args.EF}               | HNSW Parameter ef_construction")
    run("clip768v2", "emb", args.size, args.M, args.EF)
    print(f"Done! Have a good day!")
