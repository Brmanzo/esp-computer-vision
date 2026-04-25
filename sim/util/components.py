import queue
import cocotb
import random
from util.utilities import assert_resolvable
from cocotb.triggers import RisingEdge, with_timeout

class ReadyValidInterface():
    def __init__(self, clk, reset, valid, ready):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        # Handle case where reset might be a constant Logic(0)
        if not hasattr(self._rst_i, "value"):
            return bool(int(self._rst_i))
            
        if((not self._rst_i.value.is_resolvable) or self._rst_i.value == 1):
            return True
        return False

    def _get_pin_value(self, pin):
        """Helper to handle both simulator handles and constant Logic/ints."""
        if hasattr(pin, "value"):
            return int(pin.value)
        return int(pin)
        
    def assert_resolvable(self):
        """Only call the external utility if the pins are actual simulator handles."""
        if(not self.is_in_reset()):
            if hasattr(self._valid, "value"):
                assert_resolvable(self._valid)
            if hasattr(self._ready, "value"):
                assert_resolvable(self._ready)

    def is_handshake(self):
        v = self._get_pin_value(self._valid)
        r = self._get_pin_value(self._ready)
        return (v == 1) and (r == 1)

    async def _handshake(self):
        while True:
            await RisingEdge(self._clk_i)
            if (not self.is_in_reset()):
                self.assert_resolvable()
                if(self.is_handshake()):
                    break

    async def handshake(self, ns):
        if(ns):
            await with_timeout(self._handshake(), ns, 'ns')
        else:
            await self._handshake()

class ModelRunner:
    def __init__(self, dut, model):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i

        self._rv_in = ReadyValidInterface(
            self._clk_i, self._rst_i,
            dut.valid_i, dut.ready_o
        )

        self._rv_out = ReadyValidInterface(
            self._clk_i, self._rst_i,
            dut.valid_o, dut.ready_i
        )

        self._model = model
        self._events = queue.SimpleQueue()

        self._coro_run_input = None
        self._coro_run_output = None
        
        # Flag to track when the pipeline is officially primed
        self._seen_expected = False

    def start(self):
        if self._coro_run_input is not None or self._coro_run_output is not None:
            raise RuntimeError("Model already started")

        self._coro_run_input = cocotb.start_soon(self._run_input())
        self._coro_run_output = cocotb.start_soon(self._run_output())

    async def _run_input(self):
        while True:
            await self._rv_in.handshake(None)

            expected = self._model.consume()

            if expected is not None:
                if isinstance(expected, list):
                    expected = tuple(expected)
                    for item in expected:
                        self._events.put(item)
                else:
                    self._events.put(expected)

    async def _run_output(self):
        from cocotb.triggers import Timer
        from decimal import Decimal
        
        while True:
            await self._rv_out.handshake(None)
            
            # 1. Resolve same-cycle Cocotb scheduling races
            if self._events.qsize() == 0:
                await Timer(Decimal(0), units="ns")
                
            # 2. Ignore warmup garbage from deep pipelines
            if self._events.qsize() > 0:
                self._seen_expected = True
            elif not self._seen_expected:
                continue # Ignore valid outputs until the model is primed

            assert self._events.qsize() > 0, (
                "Error! Module produced output without expected input"
            )

            expected = self._events.get()
            self._model.produce(expected)

    def stop(self):
        if self._coro_run_input is None and self._coro_run_output is None:
            raise RuntimeError("Model never started")

        if self._coro_run_input is not None:
            self._coro_run_input.kill()
            self._coro_run_input = None

        if self._coro_run_output is not None:
            self._coro_run_output.kill()
            self._coro_run_output = None

class StreamDriver:
    def __init__(self, clk, rst, data_pins, valid_pin, ready_pin):
        self.rv = ReadyValidInterface(clk, rst, valid_pin, ready_pin)
        # Accept either a single pin or a list of pins
        if isinstance(data_pins, list):
            self.data_pins = data_pins
        else:
            self.data_pins = [data_pins]

    async def drive(self, generator, rate_gen, length, on_handshake=None):
        count = 0
        self.rv._valid.value = 0
        
        while self.rv.is_in_reset():
            await RisingEdge(self.rv._clk_i)

        while count < length:
            produce = rate_gen.generate()
            self.rv._valid.value = int(produce)

            if produce:
                packed_vals, raw_vals = generator.generate()

                if not isinstance(packed_vals, (list, tuple)):
                    packed_vals = [packed_vals]
                # Drive all associated data pins
                for pin, val in zip(self.data_pins, packed_vals):
                    pin.value = val

                await self.rv.handshake(None)
                if on_handshake:
                    on_handshake(raw_vals)
                count += 1
            else:
                await RisingEdge(self.rv._clk_i)
        
        self.rv._valid.value = 0

class RateGenerator():
    def __init__(self, dut, r):
        self._rate = r

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            return (random.randint(1,int(1/self._rate)) == 1)
        
class InputModel():
    def __init__(self, dut, data_gen, rate_gen, length, 
                 data_pins=None, valid_pin=None, ready_pin=None, 
                 on_handshake=None):
        
        self._dut = dut
        
        # 1. Resolve Pins (Use provided override OR fallback to standard names)
        # Fallback to dut.data_i, dut.valid_i, dut.ready_o if not specified
        d_pins = data_pins if data_pins is not None else getattr(dut, "data_i", None)
        v_pin  = valid_pin  if valid_pin  is not None else getattr(dut, "valid_i", None)
        r_pin  = ready_pin  if ready_pin  is not None else getattr(dut, "ready_o", None)

        # 2. Initialize the internal StreamDriver with the resolved pins
        self._driver = StreamDriver(
            dut.clk_i, dut.rst_i, 
            d_pins, v_pin, r_pin
        )

        self._data_gen = data_gen
        self._rate_gen = rate_gen
        self._length = length
        self._on_handshake = on_handshake 
        self._coro = None
        self._nin = 0

    def _handshake_wrapper(self, raw_val):
        """Internal helper to increment counters and trigger external callbacks."""
        self._nin += 1
        if self._on_handshake:
            self._on_handshake(raw_val)

    def start(self):
        if self._coro is not None:
            raise RuntimeError("Input Model already started")
        # All cycle-by-cycle logic happens inside the Driver now
        self._coro = cocotb.start_soon(
            self._driver.drive(
                self._data_gen, 
                self._rate_gen, 
                self._length, 
                on_handshake=self._handshake_wrapper
            )
        )

    def stop(self) -> None:
        if self._coro is not None:
            self._coro.kill()
            self._coro = None

    async def wait(self, t):
        if self._coro is not None:
            await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin