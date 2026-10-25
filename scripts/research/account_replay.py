"""Local research accounts: freeze recommendations, obey cash and halted inventory."""

import numpy as np


def drawdown(path):
    values = np.r_[1.0, path]
    return float(np.min(values / np.maximum.accumulate(values) - 1))


def replay(window, schedule, arrays, symbols, day_index, cost_bps, style, position_limit=25):
    fee = cost_bps / 10000
    cash = 1.0
    positions, marks = {}, {}
    path, trades, events = [], [], []
    entries = {day: i for i, day in enumerate(window["entry_dates"])}
    cycle_exits = set(window["exit_dates"])
    is_cycle = style == "cycle_close_open"

    def threshold(row, stock):
        return (
            0.045
            if arrays["is_st"][row, stock]
            else (0.195 if arrays["growth_board"][stock] else 0.095)
        )

    def sellable(row, stock, when):
        price = arrays[when][row, stock]
        valid = (
            np.isfinite(price)
            and price > 0
            and arrays["volume"][row, stock] > 0
            and arrays["tradable"][row, stock]
        )
        if not valid:
            return False
        if when == "open":
            # Continuous accounts use the observed adjusted prior close. The cycle
            # account reproduces the preceding experiment's delayed-exit rule.
            reference = marks[stock] if is_cycle else arrays["prior_close"][row, stock]
            return (
                np.isfinite(reference)
                and reference > 0
                and price / reference - 1 > -threshold(row, stock)
            )
        high, low = arrays["high"][row, stock], arrays["low"][row, stock]
        previous = arrays["prior_close"][row, stock]
        one_price = np.isfinite(high) and np.isfinite(low) and abs(high - low) < 1e-9
        return not (
            one_price and np.isfinite(previous) and price / previous - 1 <= -threshold(row, stock)
        )

    def sell(day, row, stock, quantity, when):
        nonlocal cash
        amount = quantity * arrays[when][row, stock]
        cash += amount * (1 - fee)
        trades.append(
            {
                "day": day,
                "stock": stock,
                "side": "sell",
                "notional": amount,
                "fee": amount * fee,
                "price": arrays[when][row, stock],
            }
        )
        positions[stock] -= quantity
        if positions[stock] < 1e-13:
            del positions[stock]

    def buy(day, row, stock, amount, legacy=False):
        nonlocal cash
        # Legacy reproduction charged cost to allocated cash. Continuous controls
        # both charge fee on traded notional; their comparison uses identical fees.
        charged = amount * fee / (1 - fee) if legacy else amount * fee
        cash -= amount + charged
        price = arrays["open"][row, stock]
        positions[stock] = positions.get(stock, 0.0) + amount / price
        marks[stock] = price
        trades.append(
            {
                "day": day,
                "stock": stock,
                "side": "buy",
                "notional": amount,
                "fee": charged,
                "price": price,
            }
        )

    for day in window["days"]:
        row = day_index[day]
        if day in entries:
            wanted = [symbols[s] for s in schedule[entries[day]]]
            if is_cycle:
                for stock, quantity in list(positions.items()):
                    if sellable(row, stock, "open"):
                        sell(day, row, stock, quantity, "open")
                wanted = [s for s in wanted if s not in positions]
                if position_limit is not None:
                    wanted = wanted[: max(0, position_limit - len(positions))]
                if wanted and cash > 0:
                    allocation = cash / len(wanted)
                    for stock in wanted:
                        buy(day, row, stock, allocation * (1 - fee), legacy=True)
                    if abs(cash) < 1e-12:
                        cash = 0.0
            else:
                fixed = {stock for stock in positions if not sellable(row, stock, "open")}
                active = [stock for stock in wanted if stock not in fixed]
                if position_limit is not None:
                    active = active[: max(0, position_limit - len(fixed))]
                active_set = set(active)
                for stock, quantity in list(positions.items()):
                    if stock not in fixed and (
                        style == "continuous_full" or stock not in active_set
                    ):
                        sell(day, row, stock, quantity, "open")
                for stock in fixed:
                    events.append({"day": day, "stock": stock, "action": "rebalance_retained"})
                if active:
                    values = {
                        stock: positions.get(stock, 0.0) * arrays["open"][row, stock]
                        for stock in active
                    }
                    if style == "continuous_full":
                        target = cash / (len(active) * (1 + fee))
                    else:
                        # Solve a cash-conserving equal-weight portfolio after fees.
                        low, high = 0.0, (cash + sum(values.values())) / len(active)
                        for _ in range(60):
                            target = (low + high) / 2
                            remaining = cash + sum(
                                max(value - target, 0) * (1 - fee)
                                - max(target - value, 0) * (1 + fee)
                                for value in values.values()
                            )
                            if remaining > 0:
                                low = target
                            else:
                                high = target
                        target = (low + high) / 2
                        for stock, value in values.items():
                            if value > target + 1e-13:
                                sell(
                                    day,
                                    row,
                                    stock,
                                    (value - target) / arrays["open"][row, stock],
                                    "open",
                                )
                    for stock, value in values.items():
                        if target > value + 1e-13:
                            buy(day, row, stock, target - value)
                    if abs(cash) < 1e-12:
                        cash = 0.0
        assert cash >= -1e-10, (day, cash)
        assert position_limit is None or len(positions) <= position_limit, (day, len(positions))
        for stock in positions:
            price = arrays["close"][row, stock]
            if np.isfinite(price) and price > 0:
                marks[stock] = price
        if (is_cycle and day in cycle_exits) or day == window["days"][-1]:
            for stock, quantity in list(positions.items()):
                if sellable(row, stock, "close"):
                    sell(day, row, stock, quantity, "close")
                else:
                    events.append({"day": day, "stock": stock, "action": "unfilled_exit_retained"})
        path.append(cash + sum(quantity * marks[stock] for stock, quantity in positions.items()))
    return np.array(path), trades, events, len(positions)


def self_checks():
    window = {"days": ["d1", "d2"], "entry_dates": ["d1", "d2"], "exit_dates": ["d1", "d2"]}
    arrays = {
        key: np.array([[10.0, 10.0], [10.0, 10.0]])
        for key in ("open", "close", "high", "low", "prior_close")
    }
    arrays["volume"] = np.ones((2, 2))
    arrays["tradable"] = np.ones((2, 2), dtype=bool)
    arrays["is_st"] = np.zeros((2, 2), dtype=bool)
    arrays["growth_board"] = np.zeros(2, dtype=bool)
    full = replay(
        window, [["a"], ["a"]], arrays, {"a": 0, "b": 1}, {"d1": 0, "d2": 1}, 10, "continuous_full"
    )
    net = replay(
        window, [["a"], ["a"]], arrays, {"a": 0, "b": 1}, {"d1": 0, "d2": 1}, 10, "continuous_net"
    )
    assert len(full[1]) == 4 and len(net[1]) == 2
    assert abs(net[0][-1] - (1 - 0.001) / (1 + 0.001)) < 1e-12
    assert net[0][-1] > full[0][-1]
    for result in (full, net):
        assert abs(result[0][-1] + sum(t["fee"] for t in result[1]) - 1) < 1e-12
    arrays["tradable"][:, 0] = False
    arrays["open"][1, 0] = np.nan
    for style in ("continuous_full", "continuous_net", "cycle_close_open"):
        result = replay(
            window, [["a"], ["b"]], arrays, {"a": 0, "b": 1}, {"d1": 0, "d2": 1}, 10, style
        )
        assert result[3] == 1 and len(result[1]) == 1
        assert result[1][0]["stock"] == 0
        assert result[0][-1] <= 1
    return {
        "unchanged_holdings_skip_roundtrip": True,
        "cash_and_fees_conserved": True,
        "halted_inventory_retained": True,
        "long_only_and_position_limit": True,
    }
