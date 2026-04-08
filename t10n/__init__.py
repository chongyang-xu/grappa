#  Copyright (c) 2024 by Data Systems Group, MPI-SWS
#  All rights reserved.
#
#  Author: Chongyang Xu <cxu@mpi-sws.org>

__version__ = "0.0.0"

import sys

if 'torch' in sys.modules and 't10' not in sys.modules:
    assert False, "Import t10n first, then torch to avoid openmp version conflicts"