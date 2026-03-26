import json
import time
from collections import OrderedDict
import numpy as np
# import duckdb   # DISABLED (DB OFF)
import sys, os
sys.path.insert(0, os.getcwd())
import SystemStats as system_stats
from incremental_csr_lib import IncrementalCSR


class AdaptiveSearchBackend:
    """
    Optimized semantic search backend (SHARD-ONLY MODE).

    DuckDB Tier-1 disabled — uses IncrementalCSR shards only.
    """

    def __init__(self,
                 config_path="search_config.json",
                 csr_config="config_listner.json",
                 tier1_db="tier1.duckdb"):

        BASE_DIR = os.path.dirname(__file__)
        self.config_path = os.path.join(BASE_DIR, config_path)

        self._load_full_config()

        # --------------------------------------------------------------
        # EXPERIENCE MEMORY (live incremental CSR)
        # --------------------------------------------------------------
        self.live_csr = IncrementalCSR(csr_config)
        self.vectorizer = self.live_csr.get_vectorizer()
        self._live_cache = None
        self._live_cache_time = 0
        self.live_cache_interval = getattr(self, 'live_cache_interval', 5.0)

        # --------------------------------------------------------------
        # TIER-1 MEANING MEMORY (DISABLED)
        # --------------------------------------------------------------
        self.tier1_db = tier1_db
        self.con = None
        # self.con = duckdb.connect(self.tier1_db)
        # try:
        #     self.con.execute("LOAD fts;")
        # except Exception:
        #     pass

        # --------------------------------------------------------------
        # REFERENCE MEMORY (disabled)
        # --------------------------------------------------------------
        self.total_rows = 0
        self.X_ref = None

        # --------------------------------------------------------------
        # SYSTEM STATS CACHE
        # --------------------------------------------------------------
        self._stats_cache = None
        self._stats_cache_time = 0
        self.stats_cache_interval = getattr(self, 'stats_cache_interval', 0.5)

        # --------------------------------------------------------------
        # CACHE STRUCTURES (kept for compatibility)
        # --------------------------------------------------------------
        self._tier1_cache = OrderedDict()
        self.tier1_cache_ttl = getattr(self, 'tier1_cache_ttl', 60)
        self.tier1_cache_maxsize = getattr(self, 'tier1_cache_maxsize', 1000)

        # --------------------------------------------------------------
        # PERFORMANCE COUNTERS
        # --------------------------------------------------------------
        self.search_count = 0
        self.live_hits = 0
        self.tier1_hits = 0
        self.cache_hits_tier1 = 0
        self.cache_misses_tier1 = 0
        self.stats_calls = 0

    # ------------------------------------------------------------------
    # CONFIG LOADING
    # ------------------------------------------------------------------
    def _load_full_config(self):
        with open(self.config_path) as f:
            cfg = json.load(f)

        for key, value in cfg.items():
            setattr(self, key, value)

        self.n_features = getattr(self, 'n_features', 10000)
        self.cpu_threshold = getattr(self, 'cpu_threshold', 80)
        self.ram_threshold = getattr(self, 'ram_threshold', 85)
        self.degraded_search_ratio = getattr(self, 'degraded_search_ratio', 0.5)
        self.confidence_threshold = getattr(self, 'confidence_threshold', 0.4)

    # ------------------------------------------------------------------
    def _get_system_stats(self):
        now = time.time()
        if (self._stats_cache is None or
                now - self._stats_cache_time > self.stats_cache_interval):
            self._stats_cache = system_stats.get_all_stats()
            self._stats_cache_time = now
            self.stats_calls += 1
        return self._stats_cache

    # ------------------------------------------------------------------
    def _get_live_matrix(self):
        now = time.time()
        if (self._live_cache is None or
                now - self._live_cache_time > self.live_cache_interval):
            self._live_cache = self.live_csr.load_full()
            self._live_cache_time = now
        return self._live_cache

    # ------------------------------------------------------------------
    # TIER-1 SEARCH (DISABLED)
    # ------------------------------------------------------------------
    def _search_tier1_cached(self, text, top_k):
        # DuckDB disabled
        return []

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    def search(self, text, top_k=5):
        self.search_count += 1

        q_vec = self.vectorizer.transform([text]).tocsr()
        X_live = self._get_live_matrix()

        if X_live.shape[0] > 0:
            scores_live = X_live.dot(q_vec.T).toarray().ravel()
            best_live = scores_live.max() if scores_live.size > 0 else 0.0

            if scores_live.size >= top_k:
                top_idx = np.argpartition(scores_live, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores_live[top_idx])[::-1]]
            else:
                top_idx = np.argsort(scores_live)[::-1]

            if best_live >= self.confidence_threshold:
                self.live_hits += 1
                return [
                    ("experience", int(i), float(scores_live[i]))
                    for i in top_idx
                ]

        return []

    # ------------------------------------------------------------------
    def cleanup(self, force=False):
        if force:
            self._live_cache = None
            self._tier1_cache.clear()
            self._stats_cache = None

    def close(self):
        # if self.con:
        #     self.con.close()
        #     self.con = None
        self._live_cache = None
        self._tier1_cache.clear()
        self._stats_cache = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    def get_performance_stats(self):
        return {
            "search_count": self.search_count,
            "live_hits": self.live_hits,
            "tier1_hits": self.tier1_hits,
            "tier1_cache_hits": self.cache_hits_tier1,
            "tier1_cache_misses": self.cache_misses_tier1,
            "tier1_cache_size": len(self._tier1_cache),
            "system_stats_calls": self.stats_calls,
            "live_cache_age": (
                time.time() - self._live_cache_time
                if self._live_cache is not None
                else -1
            ),
        }

    def _search_tier1(self, text, top_k):
        return []
