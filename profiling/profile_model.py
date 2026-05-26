import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nn.config  import NNConfig
from nn.globals import NN_CFG

from profiling.conv_layer.conv_profile             import predict_conv_layer
from profiling.pool_layer.pool_profile             import predict_pool_layer
from profiling.classifier_layer.classifier_profile import predict_classifier_layer
from profiling.overhead.overhead_profile           import predict_overhead

def profile_model(cfg: NNConfig):
    LC, FF = 0, 0
    for layer in cfg.layers:
        lc, ff = predict_conv_layer(
            ic=layer.ConvLayer._in_ch,
            oc=layer.ConvLayer._out_ch,
            ib=layer.ConvLayer._in_bits,
            wb=layer.ConvLayer._q_schedule._q_min_bits,
            dsp=layer.ConvLayer._dsp_count,
        )
        LC += lc
        FF += ff

        if layer.PoolLayer is not None:
            lc, ff = predict_pool_layer(
                ib=layer.PoolLayer._in_bits,
                ic=layer.PoolLayer._in_ch,
                mode=layer.PoolLayer._mode,
            )
            LC += lc
            FF += ff
    

    cl = cfg.classifier_config
    lc, ff = predict_classifier_layer(
        tb=cl._in_bits,
        ic=cl._in_ch,
        cc=cl._num_classes,
        wb=cl._q_schedule._q_min_bits,
        dsp=cl._dsp_count,
    )
    LC += lc
    FF += ff

    lc, ff = predict_overhead(uw=cfg.layers[0].ConvLayer._in_bits, pn=cfg._bus_width, ple=cfg.in_dims.term_count)
    LC += lc
    FF += ff

    return LC, FF

if __name__ == "__main__":
    LC, FF = profile_model(NN_CFG)
    print(f"LC: {LC}, FF: {FF}")