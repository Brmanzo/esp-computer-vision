# nn.constants.py — iCE40 UP5K board constraints
# Kept separate from nn.globals to avoid circular imports with nn.config.

BRAM_CAP    = 30 - 1   # subtract 1 for deframer skid buffer BRAM
DSP_CAP     = 8
LC_CAP      = 5280
LC_HEADROOM = 180
BUS_WIDTH   = 8

BAUD        = 115200       # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)
CLK_FREQ_HZ = 12_000_000
