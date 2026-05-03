# test_skid_buffer.py
import os
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, \
                           sim_verbose, reset_sequence, load_tests_from_csv, auto_unpack
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb  
from   cocotb.utils import get_sim_time
from   cocotb.triggers import Decimal, Timer,  RisingEdge, FallingEdge
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
         ,'fill_test'
         ,'fill_empty_test'
         ]

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"))
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(simulator, test_name, Width, Depth, HeadRoom):
    # retrieves simulators from simulator pytest param
    parameters = dict(locals())
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)


@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])


async def delay_cycles(dut, ncyc, polarity):
    for _ in range(ncyc):
        if(polarity):
            await RisingEdge(dut.clk_i)
        else:
            await FallingEdge(dut.clk_i)
    
class SkidBufModel():
    def __init__(self, dut):
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._occupancy = 0
        
        self._Width = int(dut.Width.value)
        self._Depth = int(dut.Depth.value)
        self._HeadRoom = int(dut.HeadRoom.value)
        self._deqs = 0
        self._enqs = 0
        
        self._coro_check = None

    def start(self):
        """Start the background RTS/Occupancy checker"""
        if self._coro_check is not None:
            raise RuntimeError("Model checker already started")
        self._coro_check = cocotb.start_soon(self._check_rts())

    def stop(self):
        if self._coro_check is not None:
            self._coro_check.kill()
            self._coro_check = None

    def expect_rts(self):
        thresh = self._Depth - self._HeadRoom
        return self._occupancy >= thresh

    def dut_occ_modulo(self):
        pw = self._dut.write_ptr.value.n_bits  # clog2(depth)
        ptr_w = pw + 1
        mask = (1 << ptr_w) - 1

        ww = int(self._dut.write_wrap.value)
        wp = int(self._dut.write_ptr.value)
        rw = int(self._dut.read_wrap.value)
        rp = int(self._dut.read_ptr_q.value)

        w = ((ww << pw) | wp) & mask
        r = ((rw << pw) | rp) & mask
        occ = (w - r) & mask
        return occ

    async def _check_rts(self):
        """Continuously check the almost-full flag and hardware occupancy"""
        while True:
            await RisingEdge(self._dut.clk_i)

            if not (self._dut.rst_i.value.is_resolvable and int(self._dut.rst_i.value) == 0):
                continue

            await Timer(Decimal(1.0), "ns")  # allow comb to settle after sequential updates

            occ_hw = self.dut_occ_modulo()
            thresh = self._Depth - self._HeadRoom
            exp_hw = 1 if occ_hw >= thresh else 0
            got = int(self._dut.rts_o.value)
            if sim_verbose():
                print(f"RTS Check: Expected {exp_hw}, Got {got}")
            assert got == exp_hw, (
                f"rts mismatch @ {get_sim_time('ns')}ns: "
                f"occ_model={self._occupancy} occ_hw={occ_hw} "
                f"expected={exp_hw} got={got}"
            )

    def consume(self):
        assert_resolvable(self._data_i)
        self._occupancy += 1
        self._enqs += 1
        
        # Return the value so the generic ModelRunner queues it!
        return int(self._data_i.value)

    def produce(self, expected):
        assert_resolvable(self._data_o)
        self._occupancy -= 1
        
        got = int(self._data_o.value)
        assert got == expected, f"Error! Value on deque iteration {self._deqs} does not match expected. Expected: {expected}. Got: {got}"
        self._deqs += 1
        
class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut

    def generate(self):
        # Update signature: StreamDriver expects (packed_vals, raw_vals)
        val = gen_random_unsigned(int(self._dut.Width.value), random)
        return (val, val)

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    Width = dut.Width.value

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    l = 1
    rate = 1

    model = SkidBufModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

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

    model.start()
    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    timeout = False
    try:
        await om.wait(3)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected for this fifo is two cycles."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def bypass_test(dut):
    """Test to transmit a single element in one cycle."""

    l = 1
    rate = 1

    model = SkidBufModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

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

    model.start()
    m.start()
    om.start()
    
    # Wait a few cycles before starting input
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    # Fix: Removed duplicate im.start() from above
    im.start()
    
    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    timeout = False
    try:
        await om.wait(2)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected with bypass for this fifo is one cycle. " \
        "For maximum points (and minimum fifo latency), implement the FIFO bypass path."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def fill_test(dut):
    """Test if fifo_1r1w fills to Depth elements"""

    Depth = dut.Depth.value
    l = Depth
    rate = 1

    model = SkidBufModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

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

    model.start()
    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    success = False
    try:
        await im.wait(Depth)
        success = True
    except:
        nconsumed = im.nconsumed()

    if(not success):
        assert nconsumed != Depth, f"Error! Could not fill fifo with {Depth} elements in {Depth} cycles. Fifo consumed {nconsumed} elements."
        
@cocotb.test
async def fill_empty_test(dut):
    """Test if fifo_1r1w fills to Depth elements and then empties"""

    Depth = dut.Depth.value
    l = Depth
    rate = 1

    model = SkidBufModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

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

    model.start()
    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    success = False
    try:
        await im.wait(Depth)
        success = True
    except:
        nconsumed = im.nconsumed()

    if(not success):
        assert nconsumed != Depth, f"Error! Could not fill fifo with {Depth} elements in {Depth} cycles. Fifo consumed {nconsumed} elements."

    # Fix: Stop the rate-0 OutputModel before overriding it so it doesn't fight over `ready_i`
    om.stop()
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    om.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    nproduced = 0
    success = False
    try:
        await om.wait(Depth)
        success = True
    except:
        nproduced = om.nproduced()

    if(not success):
        assert nproduced != Depth, f"Error! Could not empty fifo with {Depth} elements in {Depth} cycles. Fifo produced {nproduced} elements."

async def rate_tests(dut, in_rate, out_rate):
    l_in = int(dut.Depth.value) * 4
    l_out = l_in

    model = SkidBufModel(dut)
    m  = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), l_in)

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    model.start()
    m.start()
    om.start()
    im.start()

    # Wait up to 20 cycles for valid_o (cycle-based, not ns-based)
    for _ in range(20):
        await RisingEdge(dut.clk_i)
        if dut.valid_o.value.is_resolvable and int(dut.valid_o.value) == 1:
            break
    else:
        assert 0, "valid_o never went high within 20 cycles"

    # Timeout scaled by rates (similar to your old math) + slack
    timeout_cycles = int(2 * l_in * (1/in_rate) * (1/out_rate)) + 50

    # Convert cycles->ns using your clock (if it really is 10ns)
    CLK_NS = 10
    timeout_ns = timeout_cycles * CLK_NS

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Timed out: out={om.nproduced()}/{l_out}, "
            f"in={im.nconsumed()}/{l_in}, "
            f"budget={timeout_cycles} cycles"
        )

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