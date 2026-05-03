# Utility functions for parsing the filelist. Each module directory
# must have filelist.json with keys for "top" and "files", like so:

# {
#     "top": "hello",
#     "files":
#     ["part1/sim/hello.sv"
#     ]
# }

# Each file in the filelist is relative to the repository root.

import csv
import git
import inspect
import json
import os
from   pathlib import Path
import pytest
from   typing import Literal


import cocotb
from   cocotb_test.simulator  import run
from   cocotb.clock    import Clock
from   cocotb.utils    import get_sim_time
from   cocotb.triggers import Decimal, Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout
from   cocotb.types    import Logic
from   cocotb.utils    import get_sim_time

def sim_verbose() -> bool:
    """Return True when the user passed VERBOSE=1 to make.

    Usage in a cocotb test::

        from util.utilities import sim_verbose
        if sim_verbose():
            print(f"weights = {weights}")
    """
    return os.environ.get("VERBOSE", "0").strip() == "1"

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

    assert root is not None, "root directory path must exist"
    assert os.path.exists(root), "root directory path must exist"

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

    assert root is not None and os.path.exists(root), "root directory path must exist"
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

async def clock_start_sequence(clk_i, period=1, unit="ns"):
    # Set the clock to Z for 10 ns. This helps separate tests.
    clk_i.value = Logic("Z")
    await Timer(Decimal(10.0), units="ns")

    # Unrealistically fast clock, but nice for mental math (1 GHz)
    c = Clock(clk_i, period, unit)

    # Start the clock low to avoid issues on the first RisingEdge
    cocotb.start_soon(c.start(start_high=False))  # pyrefly: ignore[unused-coroutine]

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

def inject_params(parameters: dict, param: int, param_name: str, param_bits: int, work_dir: str):
    """Dynamically creates a Verilog header for massive integer parameters to avoid CLI overflow."""
    
    # 1. Safely remove from CLI parameters dict so cocotb-runner doesn't pass it
    parameters.pop(param_name, None)

    # 2. Calculate total bits and apply mask for proper two's complement hex formatting
    mask = (1 << param_bits) - 1
    packed_param = param & mask
    hex_width = (param_bits + 3) // 4
    
    # 3. Write out the Verilog Header (.vh)
    vh_path = os.path.join(work_dir, f"injected_{param_name.lower()}.vh")
    with open(vh_path, "w") as f:
        f.write(
            f"localparam logic signed [{param_bits-1}:0] INJECTED_{param_name.upper()} = "
            f"{param_bits}'h{packed_param:0{hex_width}x};\n"
        )

    # 4. Pass the integer strictly through the OS environment to Cocotb testbench
    os.environ[f"INJECTED_{param_name.upper()}_INT"] = str(param)

def inject_weights_and_biases(simulator: Literal['verilator', 'icarus'], parameters: dict, param_str: str, tbpath: Path, test_class: str, Weights: int, Biases: int, weight_bits: int, bias_bits: int, layer: int = 0):
    """Helper function to inject both weights and biases using the inject_params function."""
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    
    custom_work_dir = os.path.join(tbpath, "run", test_class, param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)
  
    inject_params(parameters, Weights, f"weights_{layer}", weight_bits, custom_work_dir)
    inject_params(parameters,  Biases,  f"biases_{layer}", bias_bits,   custom_work_dir)

    return custom_work_dir

def load_tests_from_csv(filepath, auto_rules=None, gen_rules=None):
    """Parses CSV and executes dependency injection rules, returning dictionaries."""
    test_cases = []
    
    with open(filepath, mode='r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 1. Base Map: Assume CSV headers perfectly match internal variable names
            parsed = {k: int(v) if v.lstrip('-').isdigit() else v for k, v in row.items()}
            
            # --- THE MAGIC AUTOWIRING FUNCTION ---
            def execute_with_injected_deps(func):
                sig = inspect.signature(func)
                kwargs = {param: parsed[param] for param in sig.parameters}
                return func(**kwargs)
            # -------------------------------------

            # 2. Process "AUTO" Rules
            if auto_rules:
                for var_name, csv_col, func in auto_rules:
                    raw_val = str(parsed.get(csv_col, "")).strip().upper()
                    if raw_val == "AUTO":
                        parsed[var_name] = execute_with_injected_deps(func)
                    else:
                        parsed[var_name] = int(raw_val)

            # 3. Process Generation Rules
            if gen_rules:
                for var_name, func in gen_rules:
                    parsed[var_name] = execute_with_injected_deps(func)

            test_cases.append(parsed)
                
    return test_cases

def auto_unpack(test_cases_dicts):
    """Converts dictionaries to tuples dynamically based on the test signature."""
    def decorator(func):
        if not test_cases_dicts:
            return pytest.mark.parametrize("", [])(func)
            
        # 1. Introspect the test function signature
        sig = inspect.signature(func)
        
        # 2. Get all args EXCEPT the ones handled by other decorators/fixtures
        # Add any other standard pytest fixtures you use to this ignore list
        ignore_list = ["test_name", "simulator"]
        target_args = [param for param in sig.parameters if param not in ignore_list]
        
        # 3. Build the perfectly ordered tuples for Pytest
        tuples_list = []
        for case_dict in test_cases_dicts:
            try:
                tuples_list.append(tuple(case_dict[arg] for arg in target_args))
            except KeyError as e:
                raise ValueError(f"Missing required parameter {e} for {func.__name__}")
        
        # 4. Generate the Pytest string and execute
        param_string = ", ".join(target_args)
        return pytest.mark.parametrize(param_string, tuples_list)(func)
        
    return decorator