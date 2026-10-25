"""Validation replay with dated ST limits and unchanged cash accounting."""

from __future__ import annotations

import numpy as np


def capped_weights(raw, cap):
    values = np.asarray(raw, dtype=float)
    assert len(values) and np.isfinite(values).all() and (values > 0).all()
    cap = max(float(cap), 1 / len(values))
    weights, remaining = np.zeros(len(values)), np.ones(len(values), dtype=bool)
    budget = 1.0
    while remaining.any():
        scaled = values[remaining] / values[remaining].sum() * budget
        indices = np.flatnonzero(remaining)
        clipped = scaled > cap + 1e-13
        if not clipped.any():
            weights[indices] = scaled
            break
        weights[indices[clipped]] = cap
        remaining[indices[clipped]] = False
        budget -= cap * clipped.sum()
    assert abs(weights.sum() - 1) < 1e-10 and weights.max() <= cap + 1e-10
    return weights


def replay_rules(
    window,
    contexts,
    arrays,
    symbol_index,
    day_index,
    cost_bps,
    candidate,
    reference=False,
    full_exposure=False,
    reference_weighting="equal",
):
    fee = cost_bps / 10000
    positions, marks, cash = {}, {}, 1.0
    path, trades, events, decisions = [], [], [], []
    rebalance = {
        window["days"][offset]: window["signal_dates"][offset // 5]
        for offset in range(0, len(window["days"]), candidate["rebalance_days"])
    }
    limit = None if reference else candidate["positions"]

    def sellable(row, stock, when):
        price = arrays[when][row, stock]
        if not (
            np.isfinite(price)
            and price > 0
            and arrays["volume"][row, stock] > 0
            and arrays["tradable"][row, stock]
        ):
            return False
        threshold = (
            0.195
            if arrays["growth_board"][stock]
            else (0.045 if arrays["is_st"][row, stock] and day < "2026-07-06" else 0.095)
        )
        prior = arrays["prior_close"][row, stock]
        if when == "open":
            return np.isfinite(prior) and prior > 0 and price / prior - 1 > -threshold
        high, low = arrays["high"][row, stock], arrays["low"][row, stock]
        locked = np.isfinite(high) and np.isfinite(low) and abs(high - low) < 1e-9
        return not (locked and np.isfinite(prior) and price / prior - 1 <= -threshold)

    def sell(day, row, stock, amount, when):
        nonlocal cash
        price = arrays[when][row, stock]
        quantity = amount / price
        cash += amount * (1 - fee)
        trades.append(
            {
                "day": day,
                "symbol": stock,
                "side": "sell",
                "notional": amount,
                "fee": amount * fee,
                "price": price,
            }
        )
        positions[stock] -= quantity
        if positions[stock] < 1e-13:
            del positions[stock]

    for day in window["days"]:
        row = day_index[day]
        if day in rebalance:
            signal = rebalance[day]
            context = contexts[signal]
            ranking = context["pool"] if reference else context["ranking"]
            # Orders use quotes/status observed at this scheduled open; scores use prior close.
            executable = []
            previous = day_index[signal]
            for symbol in ranking:
                stock = symbol_index[symbol]
                opening, prior = arrays["open"][row, stock], arrays["close"][previous, stock]
                threshold = 0.195 if arrays["growth_board"][stock] else 0.095
                if (
                    np.isfinite(opening)
                    and opening > 0
                    and arrays["volume"][row, stock] > 0
                    and arrays["tradable"][row, stock]
                    and not arrays["is_st"][row, stock]
                    and not (np.isfinite(prior) and opening / prior - 1 >= threshold)
                ):
                    executable.append(symbol)
            if reference:
                wanted = executable
            else:
                retained = set(ranking[: candidate["retain_rank"]])
                held = set(positions)
                wanted = [s for s in executable if symbol_index[s] in held and s in retained]
                wanted = wanted[:limit]
                used = set(wanted)
                wanted += [s for s in executable if s not in used][: limit - len(wanted)]
            fixed = {s for s in positions if not sellable(row, s, "open")}
            active = [symbol_index[s] for s in wanted if symbol_index[s] not in fixed]
            if limit is not None:
                active = active[: max(0, limit - len(fixed))]
            active_set = set(active)
            for stock, quantity in list(positions.items()):
                if stock not in fixed and stock not in active_set:
                    sell(day, row, stock, quantity * arrays["open"][row, stock], "open")
            for stock in fixed:
                events.append({"day": day, "stock": stock, "action": "rebalance_retained"})
            exposure = 1.0 if full_exposure else context["exposure"]
            if active:
                weighting = reference_weighting if reference else candidate["weighting"]
                if weighting == "inverse_vol":
                    raw = [1 / max(context["volatility"][s], 0.001) for s in active]
                    weights = capped_weights(raw, 0.04)
                else:
                    weights = np.full(len(active), 1 / len(active))
                values = np.array([positions.get(s, 0.0) * arrays["open"][row, s] for s in active])
                fixed_value = sum(
                    positions[s]
                    * (
                        arrays["open"][row, s]
                        if np.isfinite(arrays["open"][row, s]) and arrays["open"][row, s] > 0
                        else marks[s]
                    )
                    for s in fixed
                )
                free = cash + float(values.sum())
                low, high = 0.0, max(0.0, exposure * (free + fixed_value) - fixed_value)
                for _ in range(60):
                    target = (low + high) / 2
                    fees = fee * float(np.abs(values - target * weights).sum())
                    residual = target - max(
                        0.0, exposure * (free + fixed_value - fees) - fixed_value
                    )
                    if residual > 0:
                        high = target
                    else:
                        low = target
                targets = (low + high) / 2 * weights
                for stock, value, target in zip(active, values, targets, strict=True):
                    if value > target + 1e-13:
                        sell(day, row, stock, float(value - target), "open")
                for stock, value, target in zip(active, values, targets, strict=True):
                    if target > value + 1e-13:
                        amount = float(target - value)
                        price = arrays["open"][row, stock]
                        cash -= amount * (1 + fee)
                        positions[stock] = positions.get(stock, 0.0) + amount / price
                        marks[stock] = price
                        trades.append(
                            {
                                "day": day,
                                "symbol": stock,
                                "side": "buy",
                                "notional": amount,
                                "fee": amount * fee,
                                "price": price,
                            }
                        )
            if abs(cash) < 1e-12:
                cash = 0.0
            decisions.append(
                {
                    "day": day,
                    "signal": signal,
                    "exposure_target": exposure,
                    "positions": len(positions),
                    "symbols": [s for s in symbol_index if symbol_index[s] in positions],
                }
            )
        assert cash >= -1e-10, (day, cash)
        assert limit is None or len(positions) <= limit
        for stock in positions:
            price = arrays["close"][row, stock]
            if np.isfinite(price) and price > 0:
                marks[stock] = price
        if day == window["days"][-1]:
            for stock, quantity in list(positions.items()):
                if sellable(row, stock, "close"):
                    sell(day, row, stock, quantity * arrays["close"][row, stock], "close")
                else:
                    events.append({"day": day, "stock": stock, "action": "unfilled_exit_retained"})
        path.append(cash + sum(quantity * marks[s] for s, quantity in positions.items()))
    return np.array(path), trades, events, len(positions), decisions


def self_checks():
    arrays = {key: np.full((3, 2), 10.0) for key in ["open", "close", "high", "low", "prior_close"]}
    arrays.update(
        volume=np.ones((3, 2)),
        tradable=np.ones((3, 2), dtype=bool),
        is_st=np.zeros((3, 2), dtype=bool),
        growth_board=np.zeros(2, dtype=bool),
    )
    window = {
        "days": ["a", "b"],
        "signal_dates": ["prev"],
        "entry_dates": ["a"],
        "exit_dates": ["b"],
    }
    context = {
        "prev": {
            "pool": ["X", "Y"],
            "ranking": ["X", "Y"],
            "exposure": 0.5,
            "volatility": {0: 0.01, 1: 0.02},
        }
    }
    candidate = {"rebalance_days": 10, "positions": 2, "retain_rank": 2, "weighting": "equal"}
    result = replay_rules(
        window, context, arrays, {"X": 0, "Y": 1}, {"prev": 0, "a": 1, "b": 2}, 10, candidate
    )
    assert abs(result[0][-1] + sum(t["fee"] for t in result[1]) - 1) < 1e-12
    buys = sum(t["notional"] for t in result[1] if t["side"] == "buy")
    assert abs(buys - 0.5 / (1 + 0.5 * 0.001)) < 1e-12
    assert capped_weights([100, 1, 1, 1], 0.3).max() <= 0.3 + 1e-12
    arrays["tradable"][2, 0] = False
    retained = replay_rules(
        window, context, arrays, {"X": 0, "Y": 1}, {"prev": 0, "a": 1, "b": 2}, 10, candidate
    )
    assert retained[3] == 1 and retained[0][-1] > 0
    return {
        "cash_and_fees_conserved": True,
        "fractional_exposure_verified": True,
        "weight_cap_verified": True,
        "blocked_exit_retained": True,
    }
