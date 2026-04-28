
from util.utilities import assert_resolvable
from util.bitwise import unpack_terms, sign_extend # Adjust imports to your project
from functional_models.conv_layer import ConvLayerModel
from functional_models.pool_layer import PoolLayerModel

class SingleBlockModel:
    def __init__(self, dut, conv_params: dict, pool_params: dict):
        self._dut = dut
        
        # 1. Instantiate both models in pure software mode
        self.conv_model = ConvLayerModel(dut=None, **conv_params)
        self.pool_model = PoolLayerModel(dut=None, **pool_params)

        # 2. Store top-level pin widths for unpacking/packing
        self._InBits = int(conv_params["InBits"])
        self._InChannels = int(conv_params["InChannels"])
        
        self._OutBits = int(pool_params["OutBits"])
        self._OutChannels = int(pool_params["OutChannels"])
        
        # 3. State tracking for the final output prints
        self._deqs = 0
        self._OW = self.pool_model._OW  # Final image width comes from the Pool layer

    def step(self, raw_val, in_fire=True):
        if not in_fire:
            return None

        # 1. Step the Conv Layer (This returns a LIST of tuples, or None)
        conv_outputs = self.conv_model.step(raw_val, in_fire=True)

        final_outputs = []

        # 2. Route the burst: Step the Pool Layer for EACH valid output from Conv
        if conv_outputs is not None:
            for conv_out in conv_outputs:
                
                # Feed the single conv tuple into the pool layer
                pool_out = self.pool_model.step(conv_out, in_fire=True)
                
                if pool_out is not None:
                    # Depending on how your PoolLayerModel is written, it might return 
                    # a single tuple, or a list (if you updated it to match the Conv API). 
                    # We safely handle both here so it seamlessly chains.
                    if isinstance(pool_out, list):
                        final_outputs.extend(pool_out)
                    else:
                        final_outputs.append(pool_out)

        # 3. Return the accumulated burst of final pool outputs to the ModelRunner
        return final_outputs if len(final_outputs) > 0 else None

    def consume(self):
        """
        Called by ModelRunner when a valid INPUT handshake occurs on the top-level DUT.
        """
        assert_resolvable(self._dut.data_i)
        packed = int(self._dut.data_i.value.integer)
        
        # Unpack the raw bits from the simulator
        raw_val = unpack_terms(packed, self._InBits, self._InChannels)

        # Step entire model
        return self.step(raw_val)

    def produce(self, expected):
        """
        Called by ModelRunner when a valid OUTPUT handshake occurs on the top-level DUT.
        Checks the final hardware output against the expected pool_out.
        """
        assert_resolvable(self._dut.data_o)
        
        w = self._OutBits
        packed = int(self._dut.data_o.value.integer)

        # Calculate current coordinates for the print statement
        check_idx = self._deqs
        check_r = check_idx // self._OW
        check_c = check_idx % self._OW

        for ch in range(self._OutChannels):
            got_raw = (packed >> (ch * w)) & ((1 << w) - 1)
            
            if w == 1:
                got = got_raw
            else:
                got = sign_extend(got_raw, w)
                
            exp = int(expected[ch])

            print(f"Integrated Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}")

            assert got == exp, (
                f"Mismatch at Integrated Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}"
            )
            
        self._deqs += 1