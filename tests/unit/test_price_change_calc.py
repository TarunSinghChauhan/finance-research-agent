from src.tools.financial import calculate_price_change


def test_positive_change():
    change, pct = calculate_price_change(price=110.0, previous_close=100.0)
    assert change == 10.0
    assert pct == 10.0


def test_negative_change():
    change, pct = calculate_price_change(price=90.0, previous_close=100.0)
    assert change == -10.0
    assert pct == -10.0


def test_zero_previous_close_avoids_division_by_zero():
    change, pct = calculate_price_change(price=50.0, previous_close=0)
    assert change == 50.0
    assert pct == 0


def test_no_change():
    change, pct = calculate_price_change(price=100.0, previous_close=100.0)
    assert change == 0.0
    assert pct == 0.0


def test_rounds_to_two_decimals():
    change, pct = calculate_price_change(price=100.333, previous_close=100.0)
    assert change == 0.33
