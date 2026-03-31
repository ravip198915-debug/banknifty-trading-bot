"""
BankNifty 5-minute opening-range breakout strategy using Zerodha Kite Connect.

Strategy rules implemented from user requirements:
- Entry trigger uses 09:30-09:35 candle breakout (high => CE, low => PE).
- 3-point spot buffer on both breakout sides.
- Trade only if CPR is NOT narrow and price side matches daily 20MA filter.
- One trade per day maximum (either CE or PE).
- ATM option selection (nearest strike).
- 1 lot = 30 quantity (configurable lot multiplier).
- Spot SL/Target: 80 / 140 points.
- Premium SL/Target: 40 / 70 points.
- Session start at 09:15, force exit at 15:20, shutdown at 15:30.
- Persists strategy state in JSON and writes detailed logs.

IMPORTANT:
- This script is educational. Validate on paper trade before live deployment.
- You are responsible for compliance, risk checks, and broker/exchange rules.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

from kiteconnect import KiteConnect

T = TypeVar("T")


@dataclass
class StrategyConfig:
    api_key: str
    access_token: str
    state_file: str = "banknifty_strategy_state.json"
    log_file: str = "banknifty_strategy.log"

    # Instruments
    index_exchange: str = "NSE"
    index_symbol: str = "NIFTY BANK"
    option_exchange: str = "NFO"
    option_root: str = "BANKNIFTY"

    # Time windows (IST)
    session_start: dt.time = dt.time(9, 15)
    breakout_candle_time: dt.time = dt.time(9, 30)
    force_exit_time: dt.time = dt.time(15, 20)
    session_close_time: dt.time = dt.time(15, 30)

    # Trading parameters
    timeframe: str = "5minute"
    breakout_buffer_points: float = 3.0
    qty_per_lot: int = 30
    lots: int = 1

    # Spot based risk
    spot_sl_points: float = 80.0
    spot_target_points: float = 140.0

    # Premium based risk
    premium_sl_points: float = 40.0
    premium_target_points: float = 70.0

    # CPR filter tuning
    narrow_cpr_threshold_pct: float = 0.30  # width % of pivot (excel-aligned)

    # polling
    poll_interval_seconds: int = 5
    exit_poll_interval_seconds: int = 1

    # reliability controls
    api_retry_attempts: int = 3
    api_retry_delay_seconds: float = 0.6
    order_fill_timeout_seconds: int = 12
    order_poll_interval_seconds: float = 0.5


@dataclass
class TradeState:
    trade_date: str
    traded: bool = False
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    option_type: Optional[str] = None
    entry_spot: Optional[float] = None
    entry_premium: Optional[float] = None
    sl_spot: Optional[float] = None
    target_spot: Optional[float] = None
    sl_premium: Optional[float] = None
    target_premium: Optional[float] = None
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None


class BankNiftyBreakoutAlgo:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.kite = KiteConnect(api_key=cfg.api_key)
        self.kite.set_access_token(cfg.access_token)
        self.state = self._load_state()
        self._instrument_cache: Optional[List[Dict[str, Any]]] = None
        self._exit_in_progress = False

    # ---------- utility ----------
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.cfg.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _today_str(self) -> str:
        return dt.date.today().isoformat()

    def _load_state(self) -> TradeState:
        if os.path.exists(self.cfg.state_file):
            with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("trade_date") == self._today_str():
                return TradeState(**raw)
        return TradeState(trade_date=self._today_str())

    def _save_state(self) -> None:
        with open(self.cfg.state_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)

    def _now(self) -> dt.datetime:
        return dt.datetime.now()

    @staticmethod
    def _pivot(h: float, l: float, c: float) -> float:
        return (h + l + c) / 3.0

    def _calc_cpr(self, prev_h: float, prev_l: float, prev_c: float) -> Dict[str, float]:
        pivot = self._pivot(prev_h, prev_l, prev_c)
        bc = (prev_h + prev_l) / 2.0
        tc = 2 * pivot - bc
        width = abs(tc - bc)
        pivot_for_width = max(abs(pivot), 0.01)
        width_pct = (width / pivot_for_width) * 100
        return {
            "pivot": pivot,
            "bc": min(bc, tc),
            "tc": max(bc, tc),
            "width": width,
            "width_pct": width_pct,
        }

    def _cpr_type(self, width_pct: float) -> str:
        narrow_threshold = self.cfg.narrow_cpr_threshold_pct
        normal_threshold = getattr(self.cfg, "normal_cpr_threshold_pct", 0.50)

        if width_pct <= narrow_threshold:
            return "NARROW"
        if width_pct <= normal_threshold:
            return "NORMAL"
        return "WIDE"

    def _lot_qty(self) -> int:
        return self.cfg.qty_per_lot * self.cfg.lots

    def _with_retry(self, fn: Callable[[], T], action: str) -> T:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.api_retry_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == self.cfg.api_retry_attempts:
                    break
                sleep_s = self.cfg.api_retry_delay_seconds * attempt
                logging.warning(
                    "API retry %s/%s failed for %s: %s. Sleeping %.2fs before retry.",
                    attempt,
                    self.cfg.api_retry_attempts,
                    action,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"{action} failed after retries: {last_exc}") from last_exc

    # ---------- market data ----------
    def _instruments(self) -> List[Dict[str, Any]]:
        if self._instrument_cache is None:
            self._instrument_cache = self._with_retry(
                lambda: self.kite.instruments(),
                "fetch instruments",
            )
        return self._instrument_cache

    def _instrument_by_exchange_symbol(self, exchange: str, tradingsymbol: str) -> Dict[str, Any]:
        for ins in self._instruments():
            if ins["exchange"] == exchange and ins["tradingsymbol"] == tradingsymbol:
                return ins
        raise RuntimeError(f"Instrument not found: {exchange}:{tradingsymbol}")

    def _index_token(self) -> int:
        index = self._instrument_by_exchange_symbol(self.cfg.index_exchange, self.cfg.index_symbol)
        return index["instrument_token"]

    def _get_historical(
        self,
        token: int,
        from_dt: dt.datetime,
        to_dt: dt.datetime,
        interval: str,
    ) -> List[Dict[str, Any]]:
        return self._with_retry(
            lambda: self.kite.historical_data(token, from_dt, to_dt, interval, continuous=False, oi=False),
            f"fetch historical_data token={token} interval={interval}",
        )

    def _get_previous_day_ohlc(self) -> Dict[str, float]:
        token = self._index_token()
        end = self._now()
        start = end - dt.timedelta(days=10)
        candles = self._get_historical(token, start, end, "day")
        if len(candles) < 2:
            raise RuntimeError("Not enough day candles for previous day CPR calculation")
        prev = candles[-2]
        return {"high": prev["high"], "low": prev["low"], "close": prev["close"]}

    def _get_day_20ma(self) -> float:
        token = self._index_token()
        end = self._now()
        start = end - dt.timedelta(days=60)
        candles = self._get_historical(token, start, end, "day")
        closes = [c["close"] for c in candles if c["close"] is not None]
        if len(closes) < 21:
            raise RuntimeError("Not enough day candles for 20MA")
        return sum(closes[-21:-1]) / 20.0

    def _get_930_candle(self) -> Dict[str, Any]:
        token = self._index_token()
        today = dt.date.today()
        from_dt = dt.datetime.combine(today, dt.time(9, 15))
        to_dt = dt.datetime.combine(today, dt.time(9, 40))
        candles = self._get_historical(token, from_dt, to_dt, self.cfg.timeframe)
        for c in candles:
            if c["date"].time().replace(tzinfo=None) == self.cfg.breakout_candle_time:
                return c
        raise RuntimeError("09:30 candle not available yet")

    def _ltp(self, exchange: str, symbol: str) -> float:
        key = f"{exchange}:{symbol}"
        return self._with_retry(
            lambda: self.kite.ltp([key])[key]["last_price"],
            f"fetch ltp {key}",
        )

    def _index_ltp(self) -> float:
        return self._ltp(self.cfg.index_exchange, self.cfg.index_symbol)

    def _nearest_strike(self, spot: float, step: int = 100) -> int:
        return int(round(spot / step) * step)

    def _nearest_expiry_option(self, strike: int, option_type: str) -> Dict[str, Any]:
        candidates = []
        today = dt.date.today()
        for ins in self._instruments():
            if ins["exchange"] != self.cfg.option_exchange:
                continue
            if ins.get("name") != self.cfg.option_root:
                continue
            if ins.get("strike") != float(strike):
                continue
            if ins.get("instrument_type") != option_type:
                continue
            expiry = ins.get("expiry")
            if expiry and expiry >= today:
                candidates.append(ins)

        if not candidates:
            raise RuntimeError(f"No option found for strike={strike}, type={option_type}")

        candidates.sort(key=lambda x: x["expiry"])
        return candidates[0]

    # ---------- trading ----------
    def _place_market_buy(self, tradingsymbol: str, qty: int) -> str:
        return self._with_retry(
            lambda: self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.cfg.option_exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET,
                validity=self.kite.VALIDITY_DAY,
            ),
            f"place market buy {tradingsymbol}",
        )

    def _place_market_sell(self, tradingsymbol: str, qty: int) -> str:
        return self._with_retry(
            lambda: self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.cfg.option_exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=qty,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET,
                validity=self.kite.VALIDITY_DAY,
            ),
            f"place market sell {tradingsymbol}",
        )

    def _order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        orders = self._with_retry(lambda: self.kite.orders(), "fetch orderbook")
        for order in reversed(orders):
            if order.get("order_id") == order_id:
                return order
        return None

    def _wait_for_order_terminal(self, order_id: str, side: str) -> Dict[str, Any]:
        deadline = time.time() + self.cfg.order_fill_timeout_seconds
        last_status = "UNKNOWN"
        while time.time() <= deadline:
            order = self._order_by_id(order_id)
            if not order:
                time.sleep(self.cfg.order_poll_interval_seconds)
                continue
            status = (order.get("status") or "").upper()
            if status:
                last_status = status
            if status == "COMPLETE":
                logging.info(
                    "%s order complete | order_id=%s avg_price=%s filled=%s pending=%s",
                    side,
                    order_id,
                    order.get("average_price"),
                    order.get("filled_quantity"),
                    order.get("pending_quantity"),
                )
                return order
            if status in {"REJECTED", "CANCELLED"}:
                raise RuntimeError(
                    f"{side} order {order_id} {status}: {order.get('status_message') or order.get('status_message_raw')}"
                )
            time.sleep(self.cfg.order_poll_interval_seconds)
        raise RuntimeError(f"{side} order {order_id} not complete before timeout (last_status={last_status})")

    def _entry_signal(self, candle930: Dict[str, Any], day_20ma: float, spot_ltp: float) -> Optional[str]:
        high_trigger = candle930["high"] + self.cfg.breakout_buffer_points
        low_trigger = candle930["low"] - self.cfg.breakout_buffer_points

        if spot_ltp > high_trigger and spot_ltp > day_20ma:
            return "CE"
        if spot_ltp < low_trigger and spot_ltp < day_20ma:
            return "PE"
        return None

    def _enter_trade(self, option_type: str, spot_ltp: float) -> None:
        strike = self._nearest_strike(spot_ltp)
        instrument = self._nearest_expiry_option(strike, option_type)
        symbol = instrument["tradingsymbol"]

        qty = self._lot_qty()
        order_id = self._place_market_buy(symbol, qty)
        entry_order = self._wait_for_order_terminal(order_id, "ENTRY")
        premium_entry = float(entry_order.get("average_price") or 0.0)
        if premium_entry <= 0:
            premium_entry = self._ltp(self.cfg.option_exchange, symbol)
            logging.warning("ENTRY avg_price missing; fallback to LTP %.2f", premium_entry)

        if option_type == "CE":
            sl_spot = spot_ltp - self.cfg.spot_sl_points
            target_spot = spot_ltp + self.cfg.spot_target_points
        else:
            sl_spot = spot_ltp + self.cfg.spot_sl_points
            target_spot = spot_ltp - self.cfg.spot_target_points

        self.state.traded = True
        self.state.order_id = order_id
        self.state.symbol = symbol
        self.state.option_type = option_type
        self.state.entry_spot = spot_ltp
        self.state.entry_premium = premium_entry
        self.state.sl_spot = sl_spot
        self.state.target_spot = target_spot
        self.state.sl_premium = premium_entry - self.cfg.premium_sl_points
        self.state.target_premium = premium_entry + self.cfg.premium_target_points
        self.state.entry_time = self._now().isoformat(timespec="seconds")
        self._save_state()

        logging.info(
            "ENTRY %s | %s | qty=%s | spot=%.2f | prem=%.2f | sl_spot=%.2f | tgt_spot=%.2f | sl_prem=%.2f | tgt_prem=%.2f",
            option_type,
            symbol,
            qty,
            spot_ltp,
            premium_entry,
            self.state.sl_spot,
            self.state.target_spot,
            self.state.sl_premium,
            self.state.target_premium,
        )

    def _exit_trade(self, reason: str) -> None:
        if not self.state.symbol or self.state.exit_time or self._exit_in_progress:
            return

        self._exit_in_progress = True
        qty = self._lot_qty()
        try:
            exit_order_id = self._place_market_sell(self.state.symbol, qty)
            exit_order = self._wait_for_order_terminal(exit_order_id, "EXIT")
            avg_exit = float(exit_order.get("average_price") or 0.0)
            self.state.exit_time = self._now().isoformat(timespec="seconds")
            self.state.exit_reason = (
                f"{reason} | exit_order={exit_order_id} | avg_exit={avg_exit:.2f}"
                if avg_exit > 0
                else f"{reason} | exit_order={exit_order_id}"
            )
            self._save_state()
            logging.info(
                "EXIT %s | %s | order_id=%s | avg_exit=%.2f",
                reason,
                self.state.symbol,
                exit_order_id,
                avg_exit,
            )
        finally:
            self._exit_in_progress = False

    def _should_exit_on_rule(self) -> Optional[str]:
        if not self.state.symbol:
            return None

        try:
            spot = self._index_ltp()
            premium = self._ltp(self.cfg.option_exchange, self.state.symbol)
        except Exception as exc:
            logging.error("Exit check data fetch failed: %s", exc)
            return None

        # Spot SL/Target
        if self.state.option_type == "CE":
            if spot <= (self.state.sl_spot or -math.inf):
                return f"SPOT_SL_HIT @ {spot:.2f}"
            if spot >= (self.state.target_spot or math.inf):
                return f"SPOT_TARGET_HIT @ {spot:.2f}"
        else:
            if spot >= (self.state.sl_spot or math.inf):
                return f"SPOT_SL_HIT @ {spot:.2f}"
            if spot <= (self.state.target_spot or -math.inf):
                return f"SPOT_TARGET_HIT @ {spot:.2f}"

        # Premium SL/Target (long option)
        if premium <= (self.state.sl_premium or -math.inf):
            return f"PREMIUM_SL_HIT @ {premium:.2f}"
        if premium >= (self.state.target_premium or math.inf):
            return f"PREMIUM_TARGET_HIT @ {premium:.2f}"

        return None

    # ---------- main loop ----------
    def run(self) -> None:
        self._setup_logging()
        logging.info("Starting BankNifty CPR+20MA breakout strategy")

        prev = self._get_previous_day_ohlc()
        cpr = self._calc_cpr(prev["high"], prev["low"], prev["close"])
        cpr_type = self._cpr_type(cpr["width_pct"])

        day_20ma = self._get_day_20ma()
        logging.info(
            "CPR type=%s (width_pct=%.4f) | pivot=%.2f bc=%.2f tc=%.2f | day20ma=%.2f",
            cpr_type,
            cpr["width_pct"],
            cpr["pivot"],
            cpr["bc"],
            cpr["tc"],
            day_20ma,
        )

        if cpr_type == "NARROW":
            logging.info("Narrow CPR day. No trade as per rule.")
            return

        def _shutdown(signum: int, _frame: Any) -> None:
            logging.warning("Received signal=%s. Graceful shutdown.", signum)
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        candle930: Optional[Dict[str, Any]] = None
        one_trade_done_logged = False

        while True:
            now = self._now().time()

            if now < self.cfg.session_start:
                time.sleep(self.cfg.poll_interval_seconds)
                continue

            # hard stop window
            if now >= self.cfg.force_exit_time and self.state.traded and not self.state.exit_time:
                try:
                    self._exit_trade("FORCE_EXIT_15_20")
                except Exception as exc:
                    logging.error("Force exit failed: %s", exc)

            if now >= self.cfg.session_close_time:
                logging.info("Session close 15:30 reached. Exiting script.")
                break

            # one trade/day restriction
            if self.state.traded and self.state.exit_time:
                if not one_trade_done_logged:
                    logging.info("One trade per day completed. No further entries for today.")
                    one_trade_done_logged = True
                time.sleep(self.cfg.poll_interval_seconds)
                continue

            # wait until 09:30 candle is available
            if candle930 is None:
                try:
                    candle930 = self._get_930_candle()
                    logging.info(
                        "09:30 candle found O=%.2f H=%.2f L=%.2f C=%.2f",
                        candle930["open"],
                        candle930["high"],
                        candle930["low"],
                        candle930["close"],
                    )
                except RuntimeError:
                    time.sleep(self.cfg.poll_interval_seconds)
                    continue

            if not self.state.traded:
                try:
                    spot = self._index_ltp()
                except Exception as exc:
                    logging.error("Spot LTP fetch failed during entry scan: %s", exc)
                    time.sleep(self.cfg.poll_interval_seconds)
                    continue
                signal_type = self._entry_signal(candle930, day_20ma, spot)
                if signal_type:
                    try:
                        self._enter_trade(signal_type, spot)
                    except Exception as exc:
                        logging.error("Entry failed for %s: %s", signal_type, exc)

            else:
                reason = self._should_exit_on_rule()
                if reason and not self.state.exit_time:
                    try:
                        self._exit_trade(reason)
                    except Exception as exc:
                        logging.critical("Rule exit failed: %s | reason=%s", exc, reason)

            sleep_seconds = (
                self.cfg.exit_poll_interval_seconds
                if self.state.traded and not self.state.exit_time
                else self.cfg.poll_interval_seconds
            )
            time.sleep(sleep_seconds)


def load_config_from_env() -> StrategyConfig:
    api_key = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")

    if not api_key or not access_token:
        raise RuntimeError("Set KITE_API_KEY and KITE_ACCESS_TOKEN in environment.")

    return StrategyConfig(
        api_key=api_key,
        access_token=access_token,
        state_file=os.getenv("STRATEGY_STATE_FILE", "banknifty_strategy_state.json"),
        log_file=os.getenv("STRATEGY_LOG_FILE", "banknifty_strategy.log"),
    )


if __name__ == "__main__":
    config = load_config_from_env()
    algo = BankNiftyBreakoutAlgo(config)
    algo.run()
