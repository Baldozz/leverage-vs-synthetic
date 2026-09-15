"""Cash / money-market account (SPEC §4.3 cash flexibility): simple interest on the start-of-step balance.

Cash yield = short rate − ``rates.cash.spread``; negative balances (only transiently, before the
liquidity waterfall of §5.5 step 8) accrue at the loan rate passed by the caller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fosim.engine.conventions import DayCount, accrual_factor

F64 = NDArray[np.float64]


def cash_interest(cash: F64, r_short: F64, spread: float, days: float, day_count: DayCount, overdraft_rate: F64 | None = None) -> F64:
    af = accrual_factor(days, day_count)
    pos = np.maximum(cash, 0.0) * (r_short - spread) * af
    if overdraft_rate is None:
        return np.asarray(pos, dtype=np.float64)
    neg = np.minimum(cash, 0.0) * overdraft_rate * af
    return np.asarray(pos + neg, dtype=np.float64)
