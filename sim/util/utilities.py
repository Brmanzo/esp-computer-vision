# Utility functions for parsing the filelist. Each module directory
# must have filelist.json with keys for "top" and "files", like so:

# {
#     "top": "hello",
#     "files":
#     ["part1/sim/hello.sv"
#     ]
# }

# Each file in the filelist is relative to the repository root.

import os
import git

import sys
import json
import cocotb
from pathlib import Path

from cocotb_test.simulator import run
from cocotb.clock import Clock
from cocotb.regression import TestFactory
from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

def runner(simulator, timescale, tbpath, params, defs=[], testname=None, pymodule=None, jsonpath=None, jsonname="filelist.json", root=None, work_dir=None, sim_build=None, includes=None, toplevel_override=None, extra_sources=None):
    """Run the simulator on test n, with parameters params, and defines
    defs. If n is none, it will run all tests"""

    if(jsonpath is None):
        jsonpath = tbpath

    assert (os.path.exists(jsonpath)), "jsonpath directory must exist"
    
    # Check for top-level override before falling back to JSON
    if toplevel_override:
        top = toplevel_override
    else:
        top = get_top(jsonpath, jsonname)

    if(pymodule is None):
        # Default the python module to the original top name to avoid breaking things
        pymodule = "test_" + get_top(jsonpath, jsonname) 

    if(testname is None):
        testdir = "all"
    else:
        testdir=testname
    
    if(root is None):
        root = git.Repo(search_parent_directories=True).working_tree_dir

    assert (os.path.exists(root)), "root directory path must exist"

    sources = get_sources(root, tbpath)
    
    # Append any extra source files (like our wrapper)
    if extra_sources:
        sources.extend(extra_sources)

    if work_dir is None:
        work_dir = os.path.join(tbpath, "run", testdir, get_param_string(params), simulator)

    if sim_build is None:
        build_dir = os.path.join(tbpath, "build", get_param_string(params))
        if simulator.startswith("icarus"):
            build_dir = work_dir
    else:
        build_dir = sim_build

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    if simulator.startswith("icarus"):
        build_dir = work_dir

    if simulator.startswith("verilator"):
        compile_args=["-Wno-fatal", "-DVM_TRACE_FST=1", "-DVM_TRACE=1", "--timing"]
        plus_args = ["--trace", "--trace-fst"]
        if(not os.path.exists(work_dir)):
            os.makedirs(work_dir)
    else:
        compile_args=[]
        plus_args = []
        
    # Ensure includes is a list if not provided
    if includes is None:
        includes = []

    # Pass the new arguments down to the underlying run() function
    run(verilog_sources=sources,
        simulator=simulator,
        toplevel=top,
        module=pymodule,
        compile_args=compile_args,
        plus_args=plus_args,
        sim_build=build_dir,
        timescale=timescale,
        parameters=params,
        defines=defs + ["VM_TRACE_FST=1", "VM_TRACE=1"],
        includes=includes, # <--- Added here
        work_dir=work_dir,
        waves=True,
        testcase=testname)

# Function to build (run) the lint and style checks.
def lint(simulator, timescale, tbpath, params, defs=[], compile_args=[], pymodule=None, jsonpath=None, jsonname="filelist.json", root=None):

    # if json path is none, assume that it is the same as tbpath
    if(jsonpath is None):
        jsonpath = tbpath

    assert (os.path.exists(jsonpath)), "jsonpath directory must exist"
    top = get_top(jsonpath, jsonname)


    # Assume all paths in the json file are relative to the repository root.
    if(root is None):
        root = git.Repo(search_parent_directories=True).working_tree_dir

    assert (os.path.exists(root)), "root directory path must exist"
    sources = get_sources(root, tbpath)

    # if pymodule is none, assume that the python module name is test+<name of the top module>.
    if(pymodule is None):
        pymodule = "test_" + top

    # Create the expected makefile so cocotb-test won't complain.
    sim_build = "lint"
    if(not os.path.exists("lint")):
       os.mkdir("lint")

    with open("lint/Vtop.mk", 'w') as fd:
        fd.write("all:")

    make_args = ["-n"]
    compile_args += ["--lint-only"]
 
    run(verilog_sources=sources,
        simulator=simulator,
        toplevel=top,
        module=pymodule,
        compile_args=compile_args,
        sim_build=sim_build,
        timescale=timescale,
        parameters=params,
        defines=defs,
        make_args=make_args,
        compile_only=True)

def _resolve_filelist_path(p, n):
    """
    Resolve filelist path.

    Priority:
      1) Environment variable FILELIST (absolute or relative to repo root)
      2) Legacy behavior: join(p, n)
    """
    env = os.environ.get("FILELIST")
    if env:
        path = Path(env)
        # If someone passed a relative path, resolve relative to current working dir
        # (Make usually passes an absolute path, so this is just extra safety)
        return path.resolve()
    else:
        return Path(p) / n

def get_files_from_filelist(p, n):
    """Get a list of files from a json filelist."""
    filelist_path = _resolve_filelist_path(p, n)
    with open(filelist_path) as filelist:
        return json.load(filelist)["files"]

def get_sources(r, p):
    """ Get a list of source file paths from a json filelist.

    Arguments:
    r -- Absolute path to the root of the repository.
    p -- Absolute path to the directory containing filelist.json
    """
    sources = get_files_from_filelist(p, "filelist.json")
    sources = [os.path.join(r, f) for f in sources]
    return sources

def get_top(p, n="filelist.json"):
    """ Get the name of the top level module from a filelist.json.

    Arguments:
    p -- Absolute path to the directory containing json filelist
    n -- Name of the json filelist, defaults to filelist.json
    """
    return get_top_from_filelist(p, "filelist.json")

def get_top_from_filelist(p, n):
    """Get the name of the top level module from a json filelist."""
    filelist_path = _resolve_filelist_path(p, n)
    with open(filelist_path) as filelist:
        return json.load(filelist)["top"]

def get_param_string(parameters):
    """ Get a string of all the parameters concatenated together.

    Arguments:
    parameters -- a list of key value pairs
    """
    return "_".join(("{}={}".format(*i) for i in parameters.items()))


def assert_resolvable(s):
    assert s.value.is_resolvable, f"Unresolvable value in {s._path} (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

async def clock_start_sequence(clk_i, period=1, unit='ns'):
    # Set the clock to Z for 10 ns. This helps separate tests.
    clk_i.value = LogicArray(['z'])
    await Timer(10, 'ns')

    # Unrealistically fast clock, but nice for mental math (1 GHz)
    c = Clock(clk_i, period, unit)

    # Start the clock (soon). Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(c.start(start_high=False))

async def reset_sequence(clk_i, reset_i, cycles, FinishClkFalling=True, active_level=True):
    reset_i.setimmediatevalue(not active_level)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = active_level

    await ClockCycles(clk_i, cycles)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = not active_level

    reset_i._log.debug("Reset complete")

    # Always assign inputs on the falling edge
    if (not FinishClkFalling):
        await RisingEdge(clk_i)

async def delay_cycles(dut, ncyc, polarity):
    for _ in range(ncyc):
        if(polarity):
            await RisingEdge(dut.clk_i)
        else:
            await FallingEdge(dut.clk_i)

def assert_passerror(s):
    assert s.value.is_resolvable, f"Testbench pass/fail output ({s._path}) is set to x or z, but must be explicitly set to 0 at start of simulation.."

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