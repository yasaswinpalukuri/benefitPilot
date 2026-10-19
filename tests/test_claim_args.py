from __future__ import annotations

from decimal import Decimal

import pytest

from benefitpilot.routing.claim_args import find_amounts


@pytest.mark.parametrize(
    "text, amounts",
    [
        ("I've used $420 of physio. Will a $120 session be covered?", ["420", "120"]),
        ("it cost around 450 dollars", ["450"]),
        ("a 1 dollar co-pay", ["1"]),
        ("I paid 85$ for it", ["85"]),
        ("it was 60 bucks", ["60"]),
        ("a CAD 1,200.50 crown", ["1200.50"]),
        ("$ 75 visit", ["75"]),
        ("$420 dollars", ["420"]),  # one amount, not two
    ],
)
def test_find_amounts_reads_common_ways_of_writing_money(text, amounts):
    assert [c.amount for c in find_amounts(text)] == [Decimal(a) for a in amounts]


@pytest.mark.parametrize(
    "text", ["2 sessions in 24 months", "my 3 kids", "section 4.2", "since 2022"]
)
def test_find_amounts_ignores_numbers_that_are_not_money(text):
    assert find_amounts(text) == []
