import csv
from datetime import date
from pathlib import Path

from bot.execution.manual_executor import load_orders_from_csv, write_manual_order_sheet


def test_manual_orders_round_trip(tmp_path: Path) -> None:
    input_path = tmp_path / "orders.csv"
    input_path.write_text(
        "\n".join(
            [
                "symbol,side,quantity,order_type,limit_price,stop_price,time_in_force,strategy_name,thesis",
                "nvda,buy,10,market,,,day,breakout_momentum,breakout entry",
                "msft,sell,5,limit,420.0,,day,rebalance,trim winner",
            ]
        ),
        encoding="utf-8",
    )

    orders = load_orders_from_csv(input_path)
    output_path = write_manual_order_sheet(
        orders,
        tmp_path / "out" / "manual_orders.csv",
        as_of_date=date(2026, 3, 17),
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [order.symbol for order in orders] == ["NVDA", "MSFT"]
    assert rows[0]["order_type"] == "MARKET"
    assert rows[1]["limit_price"] == "420"
    assert rows[0]["as_of_date"] == "2026-03-17"
