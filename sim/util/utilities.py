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
from   typing import Literal, Optional


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

def runner(simulator, timescale, tbpath, params, defs=[], testname=None, pymodule=None, jsonpath=None, jsonname="filelist.json", root=None, work_dir=None, sim_build=None, includes=None, toplevel_override=None, extra_sources=None, filelist=None):
    if jsonname == "filelist.json" and "FILELIST" in os.environ:
        jsonname = os.path.basename(os.environ["FILELIST"])
    if(root is None):
        root = git.Repo(tbpath, search_parent_directories=True).working_tree_dir

    if(jsonpath is None):
        jsonpath = tbpath

    # Handle the filelist alias if provided
    if root and filelist:
        full_filelist_path = os.path.join(root, filelist)
        jsonpath = os.path.dirname(full_filelist_path)
        jsonname = os.path.basename(full_filelist_path)
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

    sources = get_sources(root, tbpath, jsonname, jsonpath=jsonpath)

    
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
        # Create a dynamic dump file for Icarus to avoid module name binding errors
        dump_file = os.path.join(build_dir, "iverilog_dump.v")
        with open(dump_file, 'w') as f:
            f.write("module iverilog_dump();\n")
            f.write("initial begin\n")
            f.write(f"    $dumpfile(\"{top}.vcd\");\n")
            f.write(f"    $dumpvars(0, {top});\n")
            f.write("end\n")
            f.write("endmodule\n")
        if extra_sources is None:
            extra_sources = []
        extra_sources.append(dump_file)

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
def lint(simulator, timescale, tbpath, params, defs=[], compile_args=[], pymodule=None, jsonpath=None, jsonname="filelist.json", root=None, filelist=None):
    if jsonname == "filelist.json" and "FILELIST" in os.environ:
        jsonname = os.path.basename(os.environ["FILELIST"])
    if(root is None):
        root = git.Repo(tbpath, search_parent_directories=True).working_tree_dir

    # Handle the filelist alias if provided
    if root and filelist:
        full_filelist_path = os.path.join(root, filelist)
        jsonpath = os.path.dirname(full_filelist_path)
        jsonname = os.path.basename(full_filelist_path)

    # if json path is none, assume that it is the same as tbpath
    if(jsonpath is None):
        jsonpath = tbpath

    assert (os.path.exists(jsonpath)), "jsonpath directory must exist"
    top = get_top(jsonpath, jsonname)


    # Assume all paths in the json file are relative to the repository root.
    if(root is None):
        root = git.Repo(search_parent_directories=True).working_tree_dir

    assert root is not None and os.path.exists(root), "root directory path must exist"
    sources = get_sources(root, tbpath, jsonname, jsonpath=jsonpath)

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
    if os.path.isabs(n):
        return Path(n)

    # 1) Try local path
    local_path = Path(p) / n
    if local_path.exists():
        return local_path

    # 2) Try global filelists/ directory
    # Find repo root (assuming utilities.py is in sim/util/)
    root = Path(__file__).parent.parent.parent
    global_dir = root / "filelists"
    
    # Try the provided name in global dir
    global_path = global_dir / n
    if global_path.exists():
        return global_path
    
    # Try [parent_dir_name].json in global dir (e.g. mac.json)
    module_name = Path(p).name
    module_path = global_dir / f"{module_name}.json"
    if module_path.exists():
        return module_path
        
    # 3) Fallback
    return local_path

def get_files_from_filelist(p, n):
    """Get a list of files from a json filelist."""
    filelist_path = _resolve_filelist_path(p, n)
    with open(filelist_path) as filelist:
        return json.load(filelist)["files"]

def get_sources(r, p, n="filelist.json", jsonpath=None):
    if n == "filelist.json" and "FILELIST" in os.environ:
        n = os.path.basename(os.environ["FILELIST"])
    # Priority: explicit jsonpath, then p (testbench path)
    search_dir = jsonpath if jsonpath else p
    sources = get_files_from_filelist(search_dir, n)
    sources = [os.path.join(r, f) for f in sources]
    return sources

def get_top(p, n="filelist.json"):
    """ Get the name of the top level module from a filelist.json.

    Arguments:
    p -- Absolute path to the directory containing json filelist
    n -- Name of the json filelist, defaults to filelist.json
    """
    if n == "filelist.json" and "FILELIST" in os.environ:
        n = os.path.basename(os.environ["FILELIST"])
    return get_top_from_filelist(p, n)

def get_top_from_filelist(p, n):
    """Get the name of the top level module from a json filelist."""
    filelist_path = _resolve_filelist_path(p, n)
    with open(filelist_path) as filelist:
        return json.load(filelist)["top"]

def get_param_string(parameters):
    """ Get a string of all the parameters concatenated together.

    Arguments:
    parameters -- a dictionary of key value pairs
    """
    # Filter out large or volatile parameters that shouldn't affect the build directory name
    filtered = {k: v for k, v in parameters.items() if k not in ["FileName", "Biases", "Weights"]}
    return "_".join(("{}={}".format(k, v) for k, v in filtered.items()))


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

def inject_raw_param(parameters: dict, param: int, param_name: str, param_bits: int, work_dir: str, vh_name: str, env_name: str):
    """
    More flexible version of inject_params that allows specifying exact names.
    """
    # 1. Safely remove from CLI parameters dict
    parameters.pop(param_name, None)

    # 2. Calculate total bits and apply mask
    mask = (1 << param_bits) - 1
    packed_param = param & mask
    hex_width = (param_bits + 3) // 4
    
    # 3. Write out the Verilog Header (.vh)
    vh_path = os.path.join(work_dir, vh_name)
    with open(vh_path, "w") as f:
        f.write(
            f"localparam logic signed [{param_bits-1}:0] {param_name} = "
            f"{param_bits}'h{packed_param:0{hex_width}x};\n"
        )

    # 4. Pass via environment
    os.environ[env_name] = str(param)

def inject_hex_weights(parameters: dict, weights_int: int, oc: int, ic: int, kw: int, weight_bits: int, param_name: str, work_dir: str, dsp_count: int):
    """
    Slices a large packed integer of weights and writes them to hex file for $readmemh,
    packing multiple weights per line if multiple DSPs are used in parallel.
    """
    # 1. Safely remove from CLI parameters dict
    parameters.pop(param_name, None)

    # 2. Extract dimensions
    ka = kw**2
    
    # 3. Calculate hardware-accurate partitioning (matches RTL localparams)
    effective_dsps  = min(dsp_count, oc)
    neurons_per_dsp = (oc // effective_dsps) if effective_dsps > 0 else 0
    total_terms     = ic * ka
    
    rom_depth = total_terms * neurons_per_dsp
    rom_width_bits = weight_bits * effective_dsps
    
    # 4. Setup file path and formatting
    filename = f"injected_{param_name.lower()}.hex"
    hex_path = os.path.join(work_dir, filename)
    hex_width = (rom_width_bits + 3) // 4
    mask = (1 << weight_bits) - 1
    
    # 5. Pack and write: for each local neuron in workload, for each term, pack all DSPs
    with open(hex_path, "w") as f:
        for local_neuron in range(neurons_per_dsp):
            for term in range(total_terms):
                packed_line = 0
                for dsp_idx in range(effective_dsps):
                    global_oc = dsp_idx * neurons_per_dsp + local_neuron
                    if global_oc < oc:
                        idx = global_oc * total_terms + term
                        weight = (weights_int >> (idx * weight_bits)) & mask
                        packed_line |= (weight << (dsp_idx * weight_bits))
                f.write(f"{packed_line:0{hex_width}x}\n")
            
    return os.path.abspath(hex_path)

def inject_weights_and_biases(
    simulator: Literal['verilator', 'icarus'], parameters: dict, param_str: str, tbpath: Path, test_class: str, Weights: int,
    Biases: int, weight_bits: int, bias_bits: int, weight_count: int, layer: int = 0, dsp_count: int = 0,
    custom_work_dir: Optional[str] = None
):
    """
    Helper function to inject both weights and biases.
    - If dsp_count == 0: Injects weights as a large packed parameter in a .vh header.
    - If dsp_count > 0: Injects weights into a .hex file for ROM initialization.
    """
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    
    # Write to the run directory where cocotb-test actually executes the simulator
    if custom_work_dir is None:
        custom_work_dir = os.path.join(tbpath, "run", test_class, param_str, simulator)
    
    os.makedirs(custom_work_dir, exist_ok=True)
    
    if dsp_count == 0:
        # Parallel Implementation: Use header-based injection for the giant vector
        # Total bits = weight_bits * weight_count
        inject_params(parameters, Weights, f"weights_{layer}", weight_bits * weight_count, custom_work_dir)
    else:
        # Sequential ROM Implementation: Use hex-based injection
        # We need individual dimensions for correct DSP-aware packing
        # Try to find dimensions with layer prefix (e.g. C0_OutChannels) or fallback
        oc = parameters.get(f"C{layer}_OutChannels", parameters.get("OutChannels", 1))
        # In double blocks, Layer 1's input channels are Layer 0's output channels
        ic_key = f"C{layer}_InChannels"
        prev_oc_key = f"C{layer-1}_OutChannels"
        ic = parameters.get(ic_key, parameters.get(prev_oc_key, parameters.get("InChannels", 1)))
        kw = parameters.get(f"C{layer}_KernelWidth", parameters.get("KernelWidth", 1))
        
        hex_filename = inject_hex_weights(parameters, Weights, int(oc), int(ic), int(kw), weight_bits, f"weights_{layer}", custom_work_dir, dsp_count)
        # Update the FileName parameter so the RTL can find the hex file
        # Wrap filename in quotes for Verilator/Icarus string parameter passing
        parameters[f"FileName_{layer}"] = f'"{hex_filename}"'
        if layer == 0:
            parameters["FileName"] = f'"{hex_filename}"' # Fallback for single-layer testbenches
        # Export to environment so the Python test can still access the raw weights for its reference model
        os.environ[f"INJECTED_WEIGHTS_{layer}_INT"] = str(Weights)
        
        # Also inject as header for testbench wrapper compatibility (e.g. tb_filter.sv)
        inject_params(parameters, Weights, f"weights_{layer}", weight_bits * weight_count, custom_work_dir)
        
    # Biases currently stay as header parameters
    inject_params(parameters, Biases, f"biases_{layer}", bias_bits, custom_work_dir)

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
        
        # 2. Get all args that actually exist in the CSV columns
        available_cols = test_cases_dicts[0].keys()
        target_args = [param for param in sig.parameters if param in available_cols]
        
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