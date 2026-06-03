import re
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "profiles" / "results.txt"
CSV_PATH = Path(__file__).parent.parent / "profiles" / "accuracy_features.csv"

def export_features():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    valid_data = []
    
    for line in lines[2:]:
        if not line.strip() or "─" in line:
            continue
            
        parts = line.split()
        idx = parts[0]
        float_acc_str = parts[1]
        
        if float_acc_str == "aborted":
            float_acc = 0.10
            lc = int(parts[2].lstrip("["))
        else:
            try:
                float_acc = float(float_acc_str)
                lc = int(parts[3].lstrip("["))
            except ValueError:
                continue
            
        # The arch string is everything after the first `[`
        arch_start = line.find("[")
        if arch_start == -1:
            continue
            
        arch_str = line[arch_start:]
        
        # Extract channels: look for digits right before 'ch'
        channels = [int(x) for x in re.findall(r'(\d+)ch', arch_str)]
        
        # Extract bits: look for digits right before 'b'
        bits = [int(x) for x in re.findall(r'(\d+)b', arch_str)]
        
        if len(channels) < 2 or len(bits) < 2:
            continue
            
        depth = len(channels)
        growth_rate = channels[-1] / channels[0]
        total_channels = sum(channels)
        
        # The input bits to the final conv layer is the output bits of the second-to-last layer.
        # The input bits to the classifier is the output bits of the final conv layer.
        final_conv_ib = bits[-2]
        classifier_ib = bits[-1]
        
        in_bits_list = [1] + bits[:-1]  # MNIST input is 1 bit
        channel_bits = sum(c * b for c, b in zip(channels, in_bits_list))
        
        pct_ternary = sum(1 for b in bits if b == 2) / len(bits)
        
        ternary_bandwidth = sum(c * b for c, b in zip(channels, in_bits_list) if b == 2)
        pct_ternary_bw = ternary_bandwidth / channel_bits if channel_bits > 0 else 0
        
        bandwidth_density = channel_bits / depth
        
        valid_data.append(f"{idx},{depth},{growth_rate:.4f},{total_channels},{final_conv_ib},{classifier_ib},{channel_bits},{bandwidth_density:.4f},{pct_ternary:.4f},{pct_ternary_bw:.4f},{lc},{float_acc:.4f}\n")
        
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("idx,depth,growth_rate,total_channels,final_conv_ib,classifier_ib,channel_bits,bandwidth_density,pct_ternary,pct_ternary_bw,lc,float_acc\n")
        f.writelines(valid_data)
        
    print(f"Exported {len(valid_data)} networks to {CSV_PATH}")

if __name__ == "__main__":
    export_features()
