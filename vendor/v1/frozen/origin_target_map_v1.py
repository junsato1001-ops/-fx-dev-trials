"""
ORIGIN_TARGET_MAP_V1
Freeze-candidate reference implementation for ECB/TARGET-based FX forecasting.

This module intentionally does NOT read ECB raw data and does NOT call the
frozen M0 walk_forward() helper.  It only maps formal forecast opportunities
to Last Available Fix / Target Start / Target End / H according to the
agreed calendar rules.

Core assumptions:
- Formal cutoff: 08:30 JST.
- At 08:30 JST, the ECB fix carrying the same publication date is not yet
  available. Therefore no historical intraday publication-time estimate is
  needed for this schedule.
- FX 1D origins: TARGET publication days at 08:30 JST.
- FX 1W/4W origins: every Sunday at 08:30 JST.
- Target Start: first TARGET publication day on/after origin date.
- Last Available Fix: immediately preceding TARGET publication day.
- Target End: Start + {1,7,28} calendar days, rolled forward to the next
  TARGET publication day.
- H: number of canonical TARGET intervals from Start to End.
- Historical TARGET closing rules are era-specific for 1999, 2000, 2001,
  and 2002+.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

CUTOFF_JST = "08:30"
FORMAL_PI_COVERAGE = 0.80

HORIZON_CALENDAR_DAYS = {
    "1D": 1,
    "1W": 7,
    "4W": 28,
}

class MappingError(ValueError):
    pass

def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def target_closing_days(year: int) -> set[date]:
    """
    TARGET closing days excluding weekends.

    1999:
      New Year's Day, Christmas Day, and exceptional 31 Dec.
      (Good Friday / Easter Monday / 1 May / 26 Dec were not common
       TARGET closing days in 1999.)
    2000:
      Standard six holidays.
    2001:
      Standard six + exceptional 31 Dec.
    2002+:
      Standard six-holiday long-term calendar.
    """
    if year < 1999:
        raise MappingError("TARGET calendar before 1999 is outside V1 scope")

    e = easter_sunday(year)
    standard_six = {
        date(year, 1, 1),
        e - timedelta(days=2),   # Good Friday
        e + timedelta(days=1),   # Easter Monday
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }

    if year == 1999:
        return {
            date(1999, 1, 1),
            date(1999, 12, 25),
            date(1999, 12, 31),
        }
    if year == 2001:
        return standard_six | {date(2001, 12, 31)}
    return standard_six

def is_target_publication_day(d: date) -> bool:
    if d.year < 1999:
        raise MappingError("TARGET calendar before 1999 is outside V1 scope")
    if d.weekday() >= 5:
        return False
    return d not in target_closing_days(d.year)

def next_target_day_on_or_after(d: date) -> date:
    x = d
    while not is_target_publication_day(x):
        x += timedelta(days=1)
    return x

def previous_target_day(d: date) -> date:
    """Strictly previous TARGET publication day."""
    x = d - timedelta(days=1)
    while not is_target_publication_day(x):
        x -= timedelta(days=1)
    return x

def count_target_intervals(start: date, end: date) -> int:
    """
    Number of canonical TARGET intervals from start to end.
    Equivalent to the number of publication days in (start, end].
    Both start/end must be TARGET publication days.
    """
    if not is_target_publication_day(start):
        raise MappingError("start must be a TARGET publication day")
    if not is_target_publication_day(end):
        raise MappingError("end must be a TARGET publication day")
    if end < start:
        raise MappingError("end before start")
    n = 0
    x = start
    while x < end:
        x += timedelta(days=1)
        if is_target_publication_day(x):
            n += 1
    return n

def is_formal_origin(origin_date: date, horizon: str) -> bool:
    if horizon not in HORIZON_CALENDAR_DAYS:
        raise MappingError(f"unsupported horizon: {horizon}")
    if horizon == "1D":
        return is_target_publication_day(origin_date)
    # 1W and 4W formal issuance schedule is every Sunday.
    return origin_date.weekday() == 6

@dataclass(frozen=True)
class OriginTargetMap:
    origin_date: date
    cutoff_jst: str
    horizon: str
    last_available_fix: date
    target_start: date
    ideal_target_end: date
    target_end: date
    h_canonical_intervals: int

def map_origin(origin_date: date, horizon: Literal["1D","1W","4W"]) -> OriginTargetMap:
    if not is_formal_origin(origin_date, horizon):
        raise MappingError(
            f"{origin_date.isoformat()} is not a formal {horizon} origin"
        )

    # Because cutoff is fixed at 08:30 JST, a same-date ECB fix is still
    # unavailable. Thus the first future/same-date canonical fix is Start.
    target_start = next_target_day_on_or_after(origin_date)
    last_fix = previous_target_day(target_start)

    ideal_end = target_start + timedelta(days=HORIZON_CALENDAR_DAYS[horizon])
    target_end = next_target_day_on_or_after(ideal_end)
    h = count_target_intervals(target_start, target_end)

    # Required invariant: Last Fix and Start are adjacent canonical observations.
    if next_target_day_on_or_after(last_fix + timedelta(days=1)) != target_start:
        raise MappingError("ORIGIN_START_ADJACENCY_FAIL")

    return OriginTargetMap(
        origin_date=origin_date,
        cutoff_jst=CUTOFF_JST,
        horizon=horizon,
        last_available_fix=last_fix,
        target_start=target_start,
        ideal_target_end=ideal_end,
        target_end=target_end,
        h_canonical_intervals=h,
    )

def validate_formal_pi_coverage(coverage: float | None) -> None:
    """Formal pipeline must explicitly pass 0.80; omission/other values fail."""
    if coverage is None:
        raise MappingError("FORMAL_PI_COVERAGE_MISSING")
    if abs(float(coverage) - FORMAL_PI_COVERAGE) > 1e-15:
        raise MappingError("FORMAL_PI_COVERAGE_INVALID")
