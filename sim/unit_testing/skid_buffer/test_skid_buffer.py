# test_skid_buffer.py
from   decimal import Decimal
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
from util.utilities import ReadyValidInterface, ModelRunner
tbpath = Path(__file__).parent

import cocotb  
from   cocotb.utils import get_sim_time
from   cocotb.triggers import Timer,  RisingEdge, FallingEdge, with_timeout
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

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("Width, Depth, HeadRoom", [(8, 16, 4), (8, 32, 8)])  
def test_each(simulator, test_name, Width, Depth, HeadRoom):
    # retrieves simulators from simulator pytest param
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
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
        value = random.randint(0, (1 << self._dut.Width.value) - 1)
        return value

class RateGenerator():
    def __init__(self, dut, r):
        self._rate = r

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            return (random.randint(1,int(1/self._rate)) == 1)

class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.ready_o, dut.valid_i)

        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.ready_i, dut.valid_o)
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

                success = True if (valid_o.value == 1) else False
                if (success):
                    self._nout += 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.ready_o, dut.valid_i)

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

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            din = self._data.generate()
            success = 0
            valid_i.value = produce
            data_i.value = din

            # Wait until ready
            while(produce and not success):
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)
                #assert ready_o.value.is_resolvable, f"Unresolvable value in ready_o (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

                success = True if (ready_o.value == 1) else False
                if (success):
                    self._nin += 1

            await FallingEdge(clk_i)
        return self._nin    

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
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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

    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

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
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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