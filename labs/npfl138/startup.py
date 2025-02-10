# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import random

import numpy as np
import torch


def startup(seed: int | None = None, threads: int | None = None, forkserver_instead_of_fork: bool = True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if threads is not None and threads > 0:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)

    if "fork" in torch.multiprocessing.get_all_start_methods():
        if os.environ.get("FORCE_FORK_METHOD") == "1":
            torch.multiprocessing.set_start_method("fork")
        elif forkserver_instead_of_fork:
            torch.multiprocessing.set_start_method("forkserver")
