'''
    Cole Foster
    July 11th, 2023

    SISAP Indexing Challenge
'''
import argparse
import hnswlib
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time

data_dir =  "/users/cfoste18/scratch/LAION/sisap"; # HARD-CODED

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
        download(url, os.path.join(data_dir,"data", kind, size, f"{version}.h5"))

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
    Main Script:    
        - Runs HNSW to build bottom layer graph


'''
method = "HNSW-MOD"
def run(kind, key, size, M, ef_construction):
    print(f"Running {method} on {kind}-{size}")
    index_identifier = f"{method}-M-{M}-EFC-{ef_construction}"
    
    #> Load Dataset- download if necessary
    prepare(kind, size)
    data = np.ascontiguousarray(h5py.File(os.path.join(data_dir,"data", kind, size, "dataset.h5"), "r")[key],dtype=np.float32) 
    queries = np.array(h5py.File(os.path.join(data_dir,"data", kind, size, "query.h5"), "r")[key],dtype=np.float32)
    n, d = data.shape
    ids = np.arange(n) # element ids
    print("Printing Array Information: Ensure Contiguous Array")
    print(data.flags)
    if not data.flags.c_contiguous:
        print("Error: Not a contiguous array- this is needed for C++")
        return

    #> Initialize Index/Data
    if kind.startswith("pca"):
        print(f"Initializing Index with Euclidean Distance, D={d}")
        index = hnswlib.Index(space='l2', dim=d)  # possible options are l2, cosine or ip
    elif kind.startswith("clip768"):
        print(f"Initializing Index with Cosine Distance, D={d}")
        index = hnswlib.Index(space='ip', dim=d) # possible options are l2, cosine or ip

        print("Normalizing the Vectors")
        for i in range(n):
            data[i,:] /= np.linalg.norm(data[i,:]) # prevent making a whole copy
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    else:
        raise Exception(f"unsupported input type {kind}")

    #> Get the buffer (contains pointer to dataset in memory)
    data_buffer = data.data

    #> Build or Load the Normal HNSW Index
    index_path_name = os.path.join(data_dir,"index", kind, size, f"{index_identifier}.bin")
    if os.path.exists(index_path_name):
        print("Loading Normal Index from: ",index_path_name)
        start = time.time()
        index.load_index(index_path_name, max_elements = n)
        elapsed_build = time.time() - start
        print(f"Loaded Normal Index in {elapsed_build}s.")
        index.load_dataset(data_buffer)  # always load dataset on load

    else:
        # if not, let's create and save the index
        os.makedirs(Path(index_path_name).parent, exist_ok=True)
        
        # train the index
        print(f"Training index on {data.shape}")
        start = time.time()
        index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        index.add_items(data_buffer, ids)
        elapsed_build = time.time() - start
        print(f"Done training in {elapsed_build}s.")

        # save index to file 
        index.save_index(index_path_name)
        print(f"Saved Index To: {index_path_name}")
    index.print_hierarchy()
    
    # approach 1 with normal hnsw index: normal hnsw search
    ef_vec = [10, 20, 30, 50, 70, 100, 140, 190, 250, 320, 400, 500, 650, 800, 1000]
    for ef in ef_vec:
        print(f"Starting Approach 1 Search onwith ef={ef}")
        start = time.time()
        index.set_ef(ef)  # ef should always be > k
        labels, distances = index.knn_hnsw(queries, k=10)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        identifier_modified = f"{index_identifier}-approach1"
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({identifier_modified}),query=(ef={ef})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), identifier_modified, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)

    # approach 3 with normal hnsw index: greedy traversal by hsp on each layer
    m_vec = [4, 8, 12, 16, 20, 24, 28, 32, 10, 20, 30, 40, 60, 80, 100, 120, 150] 
    for m in m_vec:
        print(f"Starting Approach 3 Search on with m={m}")
        start = time.time()
        labels, distances = index.knn_approach3(queries, k=10, m=m)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        identifier_modified = f"{index_identifier}-approach3"
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({identifier_modified}),query=(m={m})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), identifier_modified, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)

    # approach 4 with normal hnsw index
    b_vec = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48] 
    for b in b_vec:
        print(f"Starting Approach 4 Search with b={b}")
        #index.set_ef(ef)  # ef should always be > k
        start = time.time()
        labels, distances = index.knn_approach4(queries, k=10, m=b)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        identifier_modified = f"{index_identifier}-approach4"
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({identifier_modified}),query=(b={b})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), identifier_modified, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)

    # Create an approximately monotonic hierarchy
    # Build an approximate hsp on each layer (besides bottom)
    index.monotonic_hierarchy()

    # approach 1 with normal hnsw index: normal hnsw search
    for ef in ef_vec:
        print(f"Starting Approach 1 Search with ef={ef}")
        start = time.time()
        index.set_ef(ef)  # ef should always be > k
        labels, distances = index.knn_hnsw(queries, k=10)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        identifier_modified = f"{index_identifier}-hsp-approach1"
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({identifier_modified}),query=(ef={ef})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), identifier_modified, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)

    # approach 3 with normal hnsw index: greedy traversal by hsp on each layer
    for m in m_vec:
        print(f"Starting Approach 3 Search with m={m}")
        start = time.time()
        labels, distances = index.knn_approach3(queries, k=10, m=m)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        identifier_modified = f"{index_identifier}-hsp-approach3"
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({identifier_modified}),query=(m={m})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), identifier_modified, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)

    # approach 4 with normal hnsw index
    for b in b_vec:
        print(f"Starting Approach 4 Search with b={b}")
        #index.set_ef(ef)  # ef should always be > k
        start = time.time()
        labels, distances = index.knn_approach4(queries, k=10, m=b)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        # save the results
        identifier_modified = f"{index_identifier}-hsp-approach4"
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({identifier_modified}),query=(b={b})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), identifier_modified, kind, distances, labels, elapsed_build, elapsed_search, identifier, size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        default="100K"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--EF",
        type=int,
        default=400,
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
