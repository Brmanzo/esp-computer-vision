import queue
import cocotb
import random
from util.utilities import assert_resolvable
from cocotb.triggers import RisingEdge, with_timeout
from cocotb.triggers import RisingEdge, FallingEdge, Timer

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
    def __init__(self, dut, model, valid_i_pin=None, ready_o_pin=None, valid_o_pin=None, ready_i_pin=None):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        self._model = model
        
        # AUTO-DETECT: Fall back to standard ready/valid pins if not explicitly overridden
        self._valid_i = valid_i_pin if valid_i_pin is not None else getattr(dut, 'valid_i', None)
        self._ready_o = ready_o_pin if ready_o_pin is not None else getattr(dut, 'ready_o', None)
        self._valid_o = valid_o_pin if valid_o_pin is not None else getattr(dut, 'valid_o', None)
        self._ready_i = ready_i_pin if ready_i_pin is not None else getattr(dut, 'ready_i', None)

        self._events = queue.SimpleQueue()
        self._seen_expected = False
        self._nout = 0

        self._coro_run_input = None
        self._coro_run_output = None

    def start(self):
        if self._coro_run_input is not None or self._coro_run_output is not None:
            raise RuntimeError("Model already started")
        self._coro_run_input = cocotb.start_soon(self._run_input())
        self._coro_run_output = cocotb.start_soon(self._run_output())

    def stop(self):
        if self._coro_run_input is None and self._coro_run_output is None:
            raise RuntimeError("Model never started")
        if self._coro_run_input is not None and not self._coro_run_input.done():
            self._coro_run_input.kill()
            self._coro_run_input = None
        if self._coro_run_output is not None and not self._coro_run_output.done():
            self._coro_run_output.kill()
            self._coro_run_output = None

    def nproduced(self):
        return self._nout

    async def _run_input(self):
        while True:
            await RisingEdge(self._clk_i)
            if self._rst_i.value.is_resolvable and int(self._rst_i.value) == 1:
                continue

            # Evaluate Input Handshake
            v = self._valid_i.value == 1 if self._valid_i is not None else True
            r = self._ready_o.value == 1 if self._ready_o is not None else True
            
            if not (v and r):
                continue

            expected = self._model.consume()

            # Restore backward compatibility for single items vs lists
            if expected is not None:
                if isinstance(expected, (list, tuple)):
                    for item in expected:
                        self._events.put(item)
                else:
                    self._events.put(expected)

    async def _run_output(self):
        from decimal import Decimal
        while True:
            await RisingEdge(self._clk_i)
            if self._rst_i.value.is_resolvable and int(self._rst_i.value) == 1:
                continue

            # Evaluate Output Handshake
            if self._valid_o is not None or self._ready_i is not None:
                v = self._valid_o.value == 1 if self._valid_o is not None else True
                r = self._ready_i.value == 1 if self._ready_i is not None else True
                if not (v and r):
                    continue
            else:
                # Fixed-latency Mode: Wait until we actually expect something
                if self._events.empty():
                    continue

            # 1. Resolve same-cycle Cocotb scheduling races
            if self._events.qsize() == 0:
                await Timer(Decimal(0), units="ns")

            # 2. Ignore warmup garbage from deep pipelines
            if self._events.qsize() > 0:
                self._seen_expected = True
            elif not self._seen_expected:
                continue 

            assert self._events.qsize() > 0, "Error! Module produced output without expected input"

            expected = self._events.get()
            self._model.produce(expected)
            self._nout += 1

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
    from cocotb.triggers import FallingEdge, with_timeout
from cocotb.triggers import FallingEdge, RisingEdge, with_timeout
from cocotb.triggers import FallingEdge, RisingEdge, with_timeout

class OutputModel():
    def __init__(self, dut, generator=None, length=None, valid_pin=None, ready_pin=None, runner=None):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut

        self._length = length

        # Duck-typing: Handle legacy positional calls like OutputModel(dut, runner, length)
        # If 'generator' was populated positionally but is actually a runner (no generate method)
        if generator is not None and not hasattr(generator, 'generate'):
            self._runner = generator
            self._generator = None
        else:
            # Otherwise, respect the explicit keyword arguments
            self._generator = generator
            self._runner = runner

        # Auto-detect standard pins for backward compatibility
        self._valid_pin = valid_pin if valid_pin is not None else getattr(dut, 'valid_o', None)
        self._ready_pin = ready_pin if ready_pin is not None else getattr(dut, 'ready_i', None)

        self._coro = None
        self._nout = 0

    def start(self):
        if self._coro is not None:
            raise RuntimeError("Output Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self):
        if self._coro is None:
            raise RuntimeError("Output Model never started")
        self._coro.kill()
        self._coro = None

    async def wait(self, t):
        if self._coro is None:
            raise RuntimeError("Output Model never started")
        await with_timeout(self._coro, t, 'ns')

    def nproduced(self):
        return self._nout

    async def _run(self):
        self._nout = 0
        length = self._length
        if length is None:
            raise RuntimeError("Output Model length not set")
            
        await FallingEdge(self._clk_i)

        if not (self._rst_i.value.is_resolvable and self._rst_i.value == 0):
            await FallingEdge(self._rst_i)

        while self._nout < length:
            # 1. ADD THIS AWAIT HERE!
            # Always yield to the simulator to advance time 
            # and prevent zero-cycle infinite loops.
            await FallingEdge(self._clk_i) 

            consume = self._generator.generate() if self._generator else 1
            
            if self._ready_pin is not None:
                self._ready_pin.value = consume

            success = False
            while consume and not success:
                await RisingEdge(self._clk_i)
                
                if self._valid_pin is not None:
                    # Streaming Handshake Mode
                    valid_val = self._valid_pin.value if hasattr(self._valid_pin, "value") else self._valid_pin
                    if hasattr(valid_val, "is_resolvable"):
                        assert valid_val.is_resolvable, "Unresolvable value in valid_o"
                    success = True if (int(valid_val) == 1) else False
                else:
                    # Fixed-Latency Mode
                    if self._runner is not None and self._runner.nproduced() > self._nout:
                        success = True

            if success:
                self._nout += 1

        return self._nout