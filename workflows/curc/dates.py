"""Date helpers for CURC workflow planning."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def parse_iso_date(value: str) -> date:
    """Parse an ISO date string."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def water_year_bounds(water_year: int) -> tuple[date, date]:
    """Return the start and end dates for a western-US water year."""
    start = date(water_year - 1, 10, 1)
    end = date(water_year, 9, 30)
    return start, end


def water_year_calendar_years(water_year: int) -> tuple[int, int]:
    """Return the two calendar years spanned by a water year."""
    start, end = water_year_bounds(water_year)
    return start.year, end.year


def iter_dates(start: date, end: date) -> list[str]:
    """Return all ISO dates between two endpoints, inclusive."""
    current = start
    result: list[str] = []
    while current <= end:
        result.append(current.isoformat())
        current += timedelta(days=1)
    return result

