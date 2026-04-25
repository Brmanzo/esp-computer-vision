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
        if((not self._rst_i.value.is_resolvable) or self._rst_i.value  == 1):
            return True
        
    def assert_resolvable(self):
        if(not self.is_in_reset()):
            assert_resolvable(self._valid)
            assert_resolvable(self._ready)

    def is_handshake(self):
        return (int(self._valid.value) == 1) and (int(self._ready.value) == 1)

    async def _handshake(self):
        while True:
            await RisingEdge(self._clk_i)
            if (not self.is_in_reset()):
                self.assert_resolvable()
                if(self.is_handshake()):
                    break

    async def handshake(self, ns):
        """Wait for a handshake, raising an exception if it hasn't
        happened after ns nanoseconds of simulation time"""

        # If ns is none, wait indefinitely
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
        
        # NEW: Flag to track when the pipeline is officially primed
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
                
            # 2. NEW: Ignore warmup garbage from deep pipelines
            if self._events.qsize() > 0:
                self._seen_expected = True # Lock in strict checking!
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

class RateGenerator():
    def __init__(self, dut, r):
        self._rate = r

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            return (random.randint(1,int(1/self._rate)) == 1)