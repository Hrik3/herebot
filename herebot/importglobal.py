import sys
import time
import threading
import math
import pickle
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import beta as beta_dist, dirichlet, norm, gamma, multivariate_normal
from scipy.special import digamma, gammaln, rel_entr
from scipy.spatial.distance import cdist, cosine
import heapq
from scipy.cluster.vq import kmeans2
import psutil
import json
from scipy.special import logsumexp
from scipy.optimize import minimize
from typing import Callable
import copy
import importlib.util
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
import re
import itertools
# ------------------------------------------------------------
# Real CI, Phonotactic, Diagnostic engines from backend.ci
# ------------------------------------------------------------
from backend.ci import (
    CIEngine,
    PhonotacticEngine,
    DiagnosticEngine,
    add_sentence_to_memory,
)