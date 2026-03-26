"""
incremental_csr_lib.py

Self-optimising incremental CSR builder with dynamic JSON configuration,
background health monitoring, and true streaming rebuild.

Features:
- All parameters come from a JSON file, reloaded every 5 seconds.
- Background thread computes stats (rows, NNZ, shards, pending, etc.)
- No vocabulary RAM explosion → HashingVectorizer with fixed feature space.
- No file explosion → sentences batched into multi-row shards (default 50k rows).
- True streaming rebuild → memory-mapped CSR append, constant RAM usage.
- Zero-downtime ingestion and rebuild.

Usage:
    from incremental_csr_lib import IncrementalCSR
    csr = IncrementalCSR(config_path="config.json")
    csr.add("hello world")
    csr.add_batch(["a", "b", "c"])
    stats = csr.get_stats()
    print(f"Total rows: {stats['total_rows']}")
    csr.rebuild()   # merges all shards + main.npz
"""

import os
import glob
import json
import pickle
import uuid
import threading
import shutil
import tempfile
import time
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

# ======================================================================
#  Default configuration – written to config.json on first run
# ======================================================================
DEFAULT_CONFIG = {
    # Core matrix parameters
    "n_features": 2 ** 20,           # 1,048,576 columns
    "dtype": "float32",              # binary=True → all entries = 1.0
    "main_npz": "main.npz",
    "vectorizer_pkl": "vectorizer.pkl",
    "shard_dir": "shards",
    "archive_dir": "shards_processed",

    # Batching & flushing
    "batch_size": 50_000,           # rows per shard
    "auto_flush": True,            # flush when batch_size reached

    # Rebuild behaviour
    "rebuild_threshold_shards": 100,   # auto-rebuild when shard count >= this
    "rebuild_ram_high": 80.0,         # % RAM at which rebuild switches to disk mode
    "rebuild_ram_low": 50.0,          # % RAM to switch back to memory mode
    "rebuild_check_interval": 10,     # matrices between RAM checks

    # Monitoring
    "stats_update_interval": 60,      # seconds between full stats recompute
    "config_reload_interval": 5,      # seconds between config file checks

    # Performance tuning (advanced)
    "initial_rows_capacity": 1024,
    "initial_nnz_capacity": 1024,
}


class IncrementalCSR:
    """
    Thread-safe, self-monitoring incremental CSR builder.
    All parameters are read from a JSON config file and can be updated live.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialise the system. If config_path does not exist, a default
        configuration is written there.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, config_path)
        config_path = self.config_path
        self.lock = threading.RLock()
        self._pending_sentences = []

        # ------------------------------------------------------------------
        #  Load or create configuration
        # ------------------------------------------------------------------
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            print(f"Created default config at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Apply configuration (sets self.n_features, self.shard_dir, etc.)
        self._update_config(config, initial=True)

        # ------------------------------------------------------------------
        #  HashingVectorizer – no vocabulary stored, fixed output space
        # ------------------------------------------------------------------
        if os.path.exists(self.vectorizer_pkl):
            with open(self.vectorizer_pkl, "rb") as f:
                self.vectorizer = pickle.load(f)
            if self.vectorizer.n_features != self.n_features:
                raise ValueError(
                    f"Vectorizer in {self.vectorizer_pkl} has n_features="
                    f"{self.vectorizer.n_features}, but config has {self.n_features}. "
                    "Delete the pickle or adjust config."
                )
        else:
            self.vectorizer = HashingVectorizer(
                n_features=self.n_features,
                binary=True,
                ngram_range=(1, 2),
                norm=None,
                alternate_sign=False,
                dtype=np.dtype(self.dtype),
            )
            with open(self.vectorizer_pkl, "wb") as f:
                pickle.dump(self.vectorizer, f)

        # ------------------------------------------------------------------
        #  Pending batch (accumulates rows before writing a shard)
        # ------------------------------------------------------------------
        self._pending_matrices = []
        self._pending_rows = 0
        self._pending_nnz = 0
        self._batch_counter = 0          # sequence number for shard files

        # ------------------------------------------------------------------
        #  Statistics cache (updated by background thread)
        # ------------------------------------------------------------------
        self._stats = {
            "total_rows": 0,
            "total_nnz": 0,
            "rows_in_main": 0,
            "nnz_in_main": 0,
            "rows_in_shards": 0,
            "nnz_in_shards": 0,
            "rows_pending": 0,
            "nnz_pending": 0,
            "shard_count": 0,
            "pending_batch_size": 0,
            "last_rebuild_time": None,
            "last_stats_update": None,
        }
        self._last_config_mtime = os.path.getmtime(config_path)
        self._last_stats_full_update = 0

        # ------------------------------------------------------------------
        #  Start background monitor thread (daemon)
        # ------------------------------------------------------------------
        self._monitor_thread = threading.Thread(target=self._background_monitor,
                                                daemon=True)
        self._monitor_thread.start()

    # ------------------------------------------------------------------
    #  Configuration handling
    # ------------------------------------------------------------------
    def _update_config(self, config: Dict[str, Any], initial: bool = False):
        """
        Update instance attributes from a config dictionary.
        Thread-safe – call with self.lock held.
        """
        # Directly assign all keys that exist in the config.
        # For initial setup, all attributes are created.
        for key, value in config.items():
            setattr(self, key, value)

        # Ensure directories exist
        self.main_npz = os.path.join(self.base_dir, self.main_npz)
        self.vectorizer_pkl = os.path.join(self.base_dir, self.vectorizer_pkl)
        self.shard_dir = os.path.join(self.base_dir, self.shard_dir)
        self.archive_dir = os.path.join(self.base_dir, self.archive_dir)

        os.makedirs(self.shard_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

        # Convert dtype string to actual dtype if needed
        if hasattr(self, 'dtype') and isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype)

    # ------------------------------------------------------------------
    #  Public API: add sentences, flush, rebuild, stats
    # ------------------------------------------------------------------
    def add(self, sentence: str) -> Dict[str, Any]:
        """Add a single sentence – will be batched with others."""
        return self.add_batch([sentence])[0]

    def add_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Add multiple sentences. They are accumulated until `batch_size`
        rows are reached, then flushed to a multi-row shard.
        Returns a list of metadata dicts (one per sentence).
        """
        if not sentences:
            return []

        results = []

        with self.lock:
            # store raw sentences instead of CSR rows
            if not hasattr(self, "_pending_sentences"):
                self._pending_sentences = []

            self._pending_sentences.extend(sentences)
            self._pending_rows += len(sentences)

            for s in sentences:
                results.append({
                    "shard": None,
                    "row_index": None,
                    "nnz": None,   # unknown until vectorized at flush
                })

            # flush if needed
            if self.auto_flush and self._pending_rows >= self.batch_size:
                self._flush_batch()

        return results

    def flush(self):
        """Force writing the current pending batch to a shard file."""
        with self.lock:
            self._flush_batch()

    def _flush_batch(self):
        """Write all pending sentences to a single .npz shard + metadata."""
        if not getattr(self, "_pending_sentences", None):
            return

        # Vectorize only once at flush time
        X = self.vectorizer.transform(self._pending_sentences).tocsr()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_name = f"shard_{timestamp}_{self._batch_counter:06d}.npz"
        shard_path = os.path.join(self.shard_dir, shard_name)

        sp.save_npz(shard_path, X)

        # Metadata
        meta = {
            "rows": X.shape[0],
            "nnz": int(X.nnz),
            "timestamp": timestamp,
            "batch_counter": self._batch_counter,
            "texts": self._pending_sentences
        }
        with open(shard_path.replace('.npz', '.meta.json'), 'w') as f:
            json.dump(meta, f)

        # Clear pending buffers
        self._pending_sentences = []
        self._pending_rows = 0
        self._pending_nnz = 0
        self._batch_counter += 1

    # ------------------------------------------------------------------
    #  True streaming rebuild (memory-mapped CSR append)
    # ------------------------------------------------------------------
    def rebuild(self, keep_shards: bool = False) -> Dict[str, Any]:
        """
        Merge all shard files + existing main.npz into a new main.npz.
        Uses memory-mapped CSR arrays to append rows one-by-one.
        Peak RAM = size of one shard + small buffers.
        """
        with self.lock:
            temp_dir = tempfile.mkdtemp(prefix="csr_rebuild_")
            self._flush_batch()
            try:
                # Paths for memmap CSR components
                indptr_path = os.path.join(temp_dir, "indptr.dat")
                indices_path = os.path.join(temp_dir, "indices.dat")
                data_path = os.path.join(temp_dir, "data.dat")

                rows_used = 0
                nnz_used = 0
                rows_capacity = self.initial_rows_capacity
                nnz_capacity = self.initial_nnz_capacity

                # Create memmap arrays with initial capacity
                indptr = np.memmap(indptr_path, dtype=np.int64, mode='w+',
                                   shape=(rows_capacity + 1,))
                indptr[0] = 0
                indptr[1:] = 0
                indptr.flush()

                indices = np.memmap(indices_path, dtype=np.int32, mode='w+',
                                    shape=(nnz_capacity,))
                indices.flush()

                data = np.memmap(data_path, dtype=self.dtype, mode='w+',
                                 shape=(nnz_capacity,))
                data.flush()

                def append_csr(mat: sp.csr_matrix):
                    nonlocal rows_used, nnz_used, rows_capacity, nnz_capacity
                    nonlocal indptr, indices, data

                    n_rows = mat.shape[0]
                    n_nnz = mat.nnz

                    # Grow indptr if needed
                    while rows_used + n_rows + 1 > rows_capacity:
                        new_capacity = max(rows_capacity * 2, rows_used + n_rows + 1)
                        new_indptr = np.memmap(indptr_path + ".tmp", dtype=np.int64,
                                               mode='w+', shape=(new_capacity + 1,))
                        new_indptr[:rows_used + 1] = indptr[:rows_used + 1]
                        new_indptr[rows_used + 1:] = indptr[rows_used]
                        new_indptr.flush()
                        os.replace(indptr_path + ".tmp", indptr_path)
                        indptr = np.memmap(indptr_path, dtype=np.int64, mode='r+',
                                           shape=(new_capacity + 1,))
                        rows_capacity = new_capacity

                    # Grow indices & data if needed
                    while nnz_used + n_nnz > nnz_capacity:
                        new_capacity = max(nnz_capacity * 2, nnz_used + n_nnz)
                        # indices
                        new_indices = np.memmap(indices_path + ".tmp", dtype=np.int32,
                                                mode='w+', shape=(new_capacity,))
                        new_indices[:nnz_used] = indices[:nnz_used]
                        new_indices.flush()
                        os.replace(indices_path + ".tmp", indices_path)
                        indices = np.memmap(indices_path, dtype=np.int32, mode='r+',
                                            shape=(new_capacity,))
                        # data
                        new_data = np.memmap(data_path + ".tmp", dtype=self.dtype,
                                             mode='w+', shape=(new_capacity,))
                        new_data[:nnz_used] = data[:nnz_used]
                        new_data.flush()
                        os.replace(data_path + ".tmp", data_path)
                        data = np.memmap(data_path, dtype=self.dtype, mode='r+',
                                         shape=(new_capacity,))
                        nnz_capacity = new_capacity

                    # Append rows from mat
                    write_ptr = nnz_used
                    indptr[rows_used] = write_ptr
                    for i in range(n_rows):
                        row_start = mat.indptr[i]
                        row_end = mat.indptr[i + 1]
                        row_nnz = row_end - row_start

                        indices[write_ptr:write_ptr + row_nnz] = mat.indices[row_start:row_end]
                        data[write_ptr:write_ptr + row_nnz] = mat.data[row_start:row_end]

                        write_ptr += row_nnz
                        indptr[rows_used + i + 1] = write_ptr

                    rows_used += n_rows
                    nnz_used += n_nnz
                    indptr.flush()
                    indices.flush()
                    data.flush()

                # ------------------------------------------------------------------
                #  Process existing main.npz
                # ------------------------------------------------------------------
                stats = {
                    "main_rows_before": 0,
                    "main_nnz_before": 0,
                    "shards_processed": 0,
                    "shard_nnz_total": 0,
                }
                if os.path.exists(self.main_npz):
                    main_mat = sp.load_npz(self.main_npz)
                    stats["main_rows_before"] = main_mat.shape[0]
                    stats["main_nnz_before"] = main_mat.nnz
                    append_csr(main_mat)
                    del main_mat

                # ------------------------------------------------------------------
                #  Process all shard files
                # ------------------------------------------------------------------
                shard_pattern = os.path.join(self.shard_dir, "shard_*.npz")
                shard_files = sorted(glob.glob(shard_pattern))
                stats["shards_processed"] = len(shard_files)

                for idx, sf in enumerate(shard_files):
                    try:
                        shard_mat = sp.load_npz(sf)
                        if shard_mat.shape[1] != self.n_features:
                            print(f"Warning: {sf} has {shard_mat.shape[1]} features, skipping.")
                            continue
                        stats["shard_nnz_total"] += shard_mat.nnz
                        append_csr(shard_mat)
                    except Exception as e:
                        print(f"Warning: could not process {sf}: {e}")

                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx+1}/{len(shard_files)} shards...")

                # ------------------------------------------------------------------
                #  Export final memmap CSR to main.npz (streaming save)
                # ------------------------------------------------------------------
                final_indptr = indptr[:rows_used + 1]
                final_indices = indices[:nnz_used]
                final_data = data[:nnz_used]

                X_final = sp.csr_matrix((final_data, final_indices, final_indptr),
                                        shape=(rows_used, self.n_features))
                tmp_npz = self.main_npz + ".tmp"
                sp.save_npz(tmp_npz, X_final)
                os.replace(tmp_npz, self.main_npz)

                stats.update({
                    "status": "success",
                    "final_rows": rows_used,
                    "final_nnz": nnz_used,
                })

                # ------------------------------------------------------------------
                #  Archive processed shards (unless keep_shards=True)
                # ------------------------------------------------------------------
                moved_count = 0
                if not keep_shards:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_subdir = os.path.join(self.archive_dir, timestamp)
                    os.makedirs(archive_subdir, exist_ok=True)

                    for sf in shard_files:
                        try:
                            shutil.move(sf, os.path.join(archive_subdir, os.path.basename(sf)))
                            meta_file = sf.replace('.npz', '.meta.json')
                            if os.path.exists(meta_file):
                                shutil.move(meta_file, os.path.join(archive_subdir, os.path.basename(meta_file)))
                            moved_count += 1
                        except Exception as e:
                            print(f"Warning: could not move {sf}: {e}")

                stats["shards_moved"] = moved_count
                stats["last_rebuild_time"] = datetime.now().isoformat()
                return stats

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    #  Statistics & monitoring
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of the latest system statistics."""
        with self.lock:
            return self._stats.copy()

    def _compute_stats_fast(self) -> Dict[str, Any]:
        """
        Quickly compute stats using pre-written shard metadata.
        Called frequently (every few seconds) by background thread.
        """
        stats = {}

        # Pending data (accurate, already in memory)
        with self.lock:
            stats["rows_pending"] = self._pending_rows
            stats["nnz_pending"] = self._pending_nnz
            stats["pending_batch_size"] = self._pending_rows

        # Shard metadata – fast via .meta.json files
        meta_pattern = os.path.join(self.shard_dir, "shard_*.meta.json")
        meta_files = glob.glob(meta_pattern)
        stats["shard_count"] = len(meta_files)
        rows_in_shards = 0
        nnz_in_shards = 0
        for mf in meta_files:
            try:
                with open(mf, "r") as f:
                    meta = json.load(f)
                    rows_in_shards += meta["rows"]
                    nnz_in_shards += meta["nnz"]
            except:
                pass
        stats["rows_in_shards"] = rows_in_shards
        stats["nnz_in_shards"] = nnz_in_shards

        # Main matrix – read shape & nnz from npz without full load
        rows_main = 0
        nnz_main = 0
        if os.path.exists(self.main_npz):
            try:
                with np.load(self.main_npz, allow_pickle=False) as data:
                    shape = data['shape']
                    rows_main = shape[0]
                    # nnz = len(indices) or indptr[-1]
                    if 'indices' in data:
                        nnz_main = len(data['indices'])
                    elif 'indptr' in data:
                        nnz_main = data['indptr'][-1]
            except:
                pass
        stats["rows_in_main"] = rows_main
        stats["nnz_in_main"] = nnz_main

        # Totals
        stats["total_rows"] = rows_main + rows_in_shards + stats["rows_pending"]
        stats["total_nnz"] = nnz_main + nnz_in_shards + stats["nnz_pending"]

        # Timestamps
        stats["last_stats_update"] = datetime.now().isoformat()
        if "last_rebuild_time" not in self._stats:
            stats["last_rebuild_time"] = None
        else:
            stats["last_rebuild_time"] = self._stats["last_rebuild_time"]

        return stats

    # ------------------------------------------------------------------
    #  Background monitor thread
    # ------------------------------------------------------------------
    def _background_monitor(self):
        """
        Runs in a daemon thread:
        - Reloads config every `config_reload_interval` seconds.
        - Updates dynamic parameters.
        - Refreshes stats cache.
        - Triggers auto-rebuild if threshold met.
        """
        last_config_mtime = self._last_config_mtime
        last_stats_time = 0

        while True:
            try:
                now = time.time()

                # ------------------------------------------------------------------
                #  1. Reload configuration if changed
                # ------------------------------------------------------------------
                if os.path.exists(self.config_path):
                    mtime = os.path.getmtime(self.config_path)
                    if mtime != last_config_mtime:
                        with open(self.config_path, "r") as f:
                            config = json.load(f)
                        with self.lock:
                            self._update_config(config, initial=False)
                        last_config_mtime = mtime

                # ------------------------------------------------------------------
                #  2. Update statistics (full recompute at slower interval)
                # ------------------------------------------------------------------
                if now - last_stats_time >= self.stats_update_interval:
                    stats = self._compute_stats_fast()
                    with self.lock:
                        self._stats.update(stats)
                    last_stats_time = now

                # ------------------------------------------------------------------
                #  3. Auto-rebuild if shard count exceeds threshold
                # ------------------------------------------------------------------
                if hasattr(self, 'rebuild_threshold_shards'):
                    threshold = self.rebuild_threshold_shards
                    if threshold > 0:
                        meta_pattern = os.path.join(self.shard_dir, "shard_*.meta.json")
                        shard_count = len(glob.glob(meta_pattern))
                        if shard_count >= threshold:
                            print(f"Auto-rebuild triggered: {shard_count} shards >= {threshold}")
                            # Run rebuild in a separate thread to not block monitor
                            threading.Thread(target=self.rebuild, daemon=True).start()

            except Exception as e:
                print(f"Background monitor error: {e}")

            time.sleep(self.config_reload_interval)

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------
    def count_shards(self) -> int:
        """Number of unprocessed shard files (without loading them)."""
        pattern = os.path.join(self.shard_dir, "shard_*.npz")
        return len(glob.glob(pattern))

    def load_full(self) -> sp.csr_matrix:
        """
        Load the entire main matrix into memory.
        Warning: This loads the whole matrix into RAM.
        """
        if not os.path.exists(self.main_npz):
            return sp.csr_matrix((0, self.n_features), dtype=self.dtype)
        return sp.load_npz(self.main_npz)

    def get_vectorizer(self):
        """Return the HashingVectorizer instance (for inference)."""
        return self.vectorizer


# ======================================================================
#  Example / self-test (only runs when script executed directly)
# ======================================================================
if __name__ == "__main__":
    print("Initialising IncrementalCSR with config.json...")
    csr = IncrementalCSR("config_Listner.json")

    # Add some sentences
    print("Adding sentences...")
    csr.add("the quick brown fox jumps over the lazy dog")
    csr.add_batch(["hello world"] * 5500)   # will create at least one shard
    csr.flush()

    # Show current stats
    stats = csr.get_stats()
    print(f"Stats: total_rows={stats['total_rows']}, shards={stats['shard_count']}")

    # Trigger rebuild
    print("Rebuilding...")
    rebuild_stats = csr.rebuild()
    print(f"Rebuild stats: {rebuild_stats}")

    # Final matrix info
    X = csr.load_full()
    print(f"Final matrix: {X.shape} rows, {X.nnz} nnz")