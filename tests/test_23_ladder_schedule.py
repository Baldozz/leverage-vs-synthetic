"""Spec test 23 (schedule part): rolling-ladder tranche count, maturities, strikes and roll dates vs a
hand-built schedule. The engine-level check (strikes = spot at purchase, residual maturities) is in
tests/test_engine_strategy_b.py.
"""

from fosim.config.schema import OptionsConfig
from fosim.instruments.option_ladder import LadderSchedule


def test_23_rolling_ladder_hand_schedule() -> None:
    opt = OptionsConfig(ladder_mode="rolling_ladder", buildup_months=12, hold_months_before_roll=12, tenor_years=5)
    sch = LadderSchedule.from_config(opt)
    assert sch.n_slots == 12
    # hand-built: slot j purchased at months j, j+12, j+24, ...; expiry = purchase + 60; sold at bid before every repurchase after the first
    expected = []
    for m in range(0, 37):
        for j in range(12):
            if m >= j and (m - j) % 12 == 0:
                expected.append({"month": m, "slot": j, "expiry_month": m + 60, "sold_before": int(m > j)})
    assert sch.hand_built_table(36) == expected
    # after build-up, exactly 12 tranches with residual maturities between 48 and 60 months
    for m in range(12, 60):
        residuals = []
        for j in range(12):
            last_purchase = max(mm for mm in range(0, m + 1) if sch.is_purchase_month(j, mm))
            residuals.append(last_purchase + 60 - m)
        assert len(residuals) == 12
        assert min(residuals) >= 48 and max(residuals) <= 60


def test_23_other_modes() -> None:
    bullet = LadderSchedule.from_config(OptionsConfig(ladder_mode="bullet", tenor_years=5, bullet_roll_residual_years=1))
    assert bullet.n_slots == 1
    # purchase at 0, roll (sell + buy) every 48 months when residual hits 12 months
    assert [m for m in range(0, 120) if bullet.is_purchase_month(0, m)] == [0, 48, 96]
    assert [m for m in range(0, 120) if bullet.is_roll_sale_month(0, m)] == [48, 96]
    hold = LadderSchedule.from_config(OptionsConfig(ladder_mode="monthly_buildup_hold_to_expiry", buildup_months=6, tenor_years=2))
    assert hold.n_slots == 6
    assert [m for m in range(0, 60) if hold.is_purchase_month(3, m)] == [3, 27, 51]
    assert not any(hold.is_roll_sale_month(3, m) for m in range(60))
