# test_deframer.py
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
from util.components import ReadyValidInterface, ModelRunner, RateGenerator
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError

import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test'
        ,'inout_fuzz_repeat_test'
        ,'in_fuzz_repeat_test'
        ,'out_fuzz_repeat_test'
        ,'full_bw_repeat_test']

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum, PacketLenElems", [("2", "4", "10"), ("1", "8", "10")])
def test_each(test_name, simulator, UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_deframer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum, PacketLenElems", [("2", "4", "10"), ("1", "8", "10")])
def test_all(simulator, UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_deframer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_deframer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_deframer")

class DeframerModel():
    HEADER0 = 0
    HEADER1 = 1
    FORWARD = 2

    def __init__(self, dut):
        self._dut = dut
        self._unpacked_o = dut.unpacked_o
        self._data_i = dut.data_i

        self._UnpackedWidth  = int(dut.UnpackedWidth.value)
        self._PackedNum      = int(dut.PackedNum.value)
        self._PackedWidth    = int(dut.PackedWidth.value)
        self._PacketLenElems = int(dut.PacketLenElems.value)
        self._HeaderByte0    = int(dut.HeaderByte0.value)
        self._HeaderByte1    = int(dut.HeaderByte1.value)

        self._mask = (1 << self._UnpackedWidth) - 1

        self._state = self.HEADER0
        self._in_remaining = 0
        
        self._deqs = 0
        self._enqs = 0
    
    def consume(self):
        assert_resolvable(self._data_i)
        b = int(self._data_i.value) & ((1 << (self._PackedWidth)) - 1)
        # Detecting first byte
        if self._state == self.HEADER0:
            if b == self._HeaderByte0:
                self._state = self.HEADER1
            return None
        
        # Detecting second byte after the first byte
        elif self._state == self.HEADER1:
            if b == self._HeaderByte1:
                self._state = self.FORWARD
                self._in_remaining = self._PacketLenElems * self._PackedNum
            elif b == self._HeaderByte0:
                # Stay in HEADER1 if we see another HEADER0
                self._state = self.HEADER1
            else:
                # Reset if unexpected byte
                self._state = self.HEADER0
            return None
        # Once both bytes detected, enqueue the rest of the packet
        elif self._state == self.FORWARD:
            if self._in_remaining > 0:
                expected_outputs = []
                for step in range(self._PackedNum):
                    val = ((b >> (self._UnpackedWidth * step)) & self._mask)
                    expected_outputs.append(val)

                self._in_remaining -= self._PackedNum
                self._enqs += 1

                if self._in_remaining <= 0:
                    self._state = self.HEADER0
                return expected_outputs
            else:
                # Packet complete: ignore until next header
                self._state = self.HEADER0
                return False

    def produce(self, expected):
        assert_resolvable(self._unpacked_o)
        got = int(self._unpacked_o.value) & self._mask
       
        self._deqs += 1
        print(f'Output #{self._deqs}: Got unpacked: {got}, Expected unpacked: {expected}')

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected {expected}, got {got}"
        )

class RandomHeaderGenerator():
    '''After a predefined delay, outputs the deframer bytes, then random data'''
    def __init__(self, dut, initial_delay, repetitions, period):
        self._dut = dut
        self._width_p = dut.PackedWidth.value
        self._initial_delay = initial_delay
        self._header_delay = initial_delay # Cycles until header appears
        self._repetitions = repetitions
        self._period = period
        self._period_remaining = period
        
    def generate(self):
        mask = (1 << self._width_p) - 1

        # Countdown before header (random output)
        if self._header_delay > 0:
            self._header_delay -= 1
            return random.randint(0, mask)

        # Header byte 0
        if self._header_delay == 0:
            self._header_delay = -1
            return int(self._dut.HeaderByte0.value) & mask

        # Header byte 1
        if self._header_delay == -1:
            self._header_delay = -2
            self._period_remaining = self._period
            return int(self._dut.HeaderByte1.value) & mask

        # Payload period (random output)
        x_i = random.randint(0, mask)

        if self._period_remaining > 0:
            self._period_remaining -= 1

        if self._period_remaining == 0:
            self._repetitions -= 1
            if self._repetitions > 0:
                # wait initial_delay random bytes before next header sequence
                self._header_delay = self._initial_delay
                self._period_remaining = self._period  # pre-init; will be reset at -1->-2 anyway

        return x_i
    
class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._width_p = dut.PackedWidth.value

    def generate(self):
        return gen_random_unsigned(self._width_p, random)

class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)

        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)
        self._generator = g
        self._length = l

        self._coro = None

        self._nout = 0

    def start(self):
        """ Start Output Model """
        if self._coro is not None:
            raise RuntimeError("Output Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
        """ Stop Output Model """
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
        """ Output Model Coroutine"""

        self._nout = 0
        clk_i = self._clk_i
        ready_i = self._dut.ready_i
        rst_i = self._dut.rst_i
        valid_o = self._dut.valid_o

        await FallingEdge(clk_i)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        # Precondition: Falling Edge of Clock
        while self._nout < self._length:
            consume = self._generator.generate()
            success = 0
            ready_i.value = consume

            # Wait until valid
            while(consume and not success):
                await RisingEdge(clk_i)
                assert_resolvable(valid_o)
                #assert valid_o.value.is_resolvable, f"Unresolvable value in valid_o (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

                fire_out = (int(valid_o.value) == 1) and (int(ready_i.value) == 1)
                if fire_out:
                    self._nout += 1
                    success = 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)

        self._rate = rate
        self._data = data
        self._length = l

        self._coro = None

        self._nin = 0

    def start(self):
        """ Start Input Model """
        if self._coro is not None:
            raise RuntimeError("Input Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
        """ Stop Input Model """
        if self._coro is None:
            raise RuntimeError("Input Model never started")
        self._coro.kill()
        self._coro = None

    async def wait(self, t):
        if self._coro is None:
            raise RuntimeError("Input Model never started")
        await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin

    async def _run(self):
        """ Input Model Coroutine"""

        self._nin = 0
        clk_i = self._clk_i
        rst_i = self._dut.rst_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        data_i = self._dut.data_i

        await delay_cycles(self._dut, 1, False)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        data = self._data.generate()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            valid_i.value = produce
            data_i.value = data

            
            await RisingEdge(clk_i)
            assert_resolvable(ready_o)

            fire_in = (int(valid_i.value) == 1) and (int(ready_o.value) == 1)
            if(fire_in):
                self._nin += 1
                data = self._data.generate()

            await FallingEdge(clk_i)
            
        return self._nin

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    delay = 10
    header_cycles = 2
    l = 1
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)
    rate = 1

    timeout = 20000
   
    model = DeframerModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), n_in)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    im.start()
    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    timed_out = False
    try:
        await om.wait(timeout + 10)
    except:
        timed_out = True
    assert not timed_out, "Error! Maximum latency expected for this fifo is two cycles."

    dut.valid_i.value = 0
    dut.ready_i.value = 0


async def rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""
    delay = 10
    header_cycles = 2
    l = 10
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), n_in)

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 
    timeout_ns        = int((n_in + 500) / slow)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout_ns} ns, with output rate {out_rate}"

async def repeat_rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""
    delay = 10
    header_cycles = 2
    repetitions = 3
    packet_len = 10

    eg = RandomHeaderGenerator(dut, delay, repetitions, period=packet_len)
    l = packet_len*repetitions
    n_in  = repetitions * (delay + header_cycles + packet_len)
    n_out = l * int(dut.PackedNum.value)

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), n_in)

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 
    timeout_ns        = int((n_in + 500) / slow)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout_ns} ns, with output rate {out_rate}"

# Standard rate tests
@cocotb.test
async def out_fuzz_test(dut):
    await rate_tests(dut, in_rate=1.0, out_rate=0.5)

@cocotb.test
async def in_fuzz_test(dut):
    await rate_tests(dut, in_rate=0.5, out_rate=1.0)

@cocotb.test
async def inout_fuzz_test(dut):
    await rate_tests(dut, in_rate=0.5, out_rate=0.5)

@cocotb.test
async def full_bw_test(dut):
    await rate_tests(dut, in_rate=1.0, out_rate=1.0)

# Repeat transmission tests to check for state retention issues across multiple packets
@cocotb.test
async def out_fuzz_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=1.0, out_rate=0.5)

@cocotb.test
async def in_fuzz_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=0.5, out_rate=1.0)

@cocotb.test
async def inout_fuzz_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=0.5, out_rate=0.5)

@cocotb.test
async def full_bw_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=1.0, out_rate=1.0)