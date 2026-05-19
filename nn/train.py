# nn.train.py
# Bradley Manzo 2026
import torch
import argparse
from torch.utils.data import DataLoader

from pathlib import Path
from typing import List, Tuple, Callable

from nn.config     import NNConfig
from nn.export     import export_nn_to_csv
from nn.arch      import cnn, QuantConv2d
from nn.plot       import plot_training
from nn.preprocess import prepare_data, get_transforms
from nn.globals    import get_hand_gesture_cfg, GESTURE_CLASSES
from nn.export     import export_csv_to_hex


def train_network(network: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, train_aug: Callable, cfg: NNConfig, device: str, global_max:float,
                epochs: int, lr: float, weight_decay: float, network_save_path: Path, initial_best_acc: float = 0.0, is_fine_tuning: bool = False) -> Tuple[List[float], List[float]]:
    '''Executes the full training and progressive quantization schedule.'''
    
    begin_q_epochs = cfg.q_schedule[0]._q_start
    # Switch to AdamW for better weight decay handling in QAT
    opt = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Use OneCycleLR for the entire duration to manage LR better through quantization transitions
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=lr, 
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.2, 
        div_factor=10, 
        final_div_factor=100
    )

    print(f"Training on {device} for {epochs} epochs with progressive quantization starting at epoch {begin_q_epochs}")
    if not getattr(network, '_is_fine_tuning', False):
        print("\n--- STARTING FULL-PRECISION TRAINING ---")
    else:
        print("\n--- CONTINUING FINE-TUNING AT FINAL QUANTIZATION ---")

    best_test_acc = initial_best_acc
    train_acc_history: List[float] = []
    test_acc_history:  List[float] = []

    for epoch in range(epochs):
        if epoch == begin_q_epochs:
            if not getattr(network, '_is_fine_tuning', False):
                print("\n--- STARTING PROGRESSIVE QUANTIZATION ---")
            
        # 1. Scheduling Logic
        quant_layer_idx = 0
        bits_changed_this_epoch = False
        for module in network.modules():
            if isinstance(module, QuantConv2d):
                target_bits = cfg.q_schedule[quant_layer_idx].get_target_bits(epoch)
                if target_bits is not None:
                    if not getattr(module, '_quantize', False): 
                        module._quantize = True
                        module._weight_bits = target_bits
                        bits_changed_this_epoch = True
                        if not is_fine_tuning: print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} quantization ON -> {target_bits}-bit")
                    elif getattr(module, '_weight_bits', None) != target_bits:
                        module._weight_bits = target_bits
                        bits_changed_this_epoch = True
                        if not is_fine_tuning: print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} dropping bits -> {target_bits}-bit")
                quant_layer_idx += 1
        
        if bits_changed_this_epoch and not getattr(network, '_is_fine_tuning', False):
            best_test_acc = 0.0

        # 2. Training Logic
        network.train()
        for x, y in train_loader:
            x    = train_aug(x) 
            x, y = x.to(device), y.to(device)
            
            logits = network(x)
            loss   = loss_fn(logits, y)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            opt.step()
            
            # Step the scheduler every BATCH
            scheduler.step()

        # 3. Evaluation Logic
        network.eval()
        train_total = 1e-6 
        train_correct = 0.0
        
        with torch.no_grad():
            if not getattr(network, '_is_fine_tuning', False):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    pred = network(x).argmax(1)
                    train_correct += (pred == y).sum().item()
                    train_total   += y.numel()
                train_acc = train_correct / train_total
            else:
                train_acc = 0.0
            
            test_correct  = test_total = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = network(x).argmax(1)
                test_correct += (pred == y).sum().item()
                test_total   += y.numel()
            test_acc = test_correct / (test_total + 1e-6)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        print(f"epoch {epoch:02d} train_acc={train_acc:.3f} test_acc={test_acc:.3f}", end="")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(network.state_dict(), network_save_path)
            if test_acc >= global_max:
                global_max = test_acc
                print(f"\033[32m  --> New best network saved! (Acc: {best_test_acc:.3f})\033[0m")
            else:
                print(f"\033[93m  --> New best network saved! (Acc: {best_test_acc:.3f} but not global max)\033[0m")
        else:
            print("")

    return train_acc_history, test_acc_history

def main():
    # 1. Configure network Baseline (to find default epochs)
    IMG_H, IMG_W = 240, 320
    # Use a temporary config to find max epochs from schedule
    tmp_cfg = get_hand_gesture_cfg(img_h=IMG_H, img_w=IMG_W)
    max_sched_epochs = max(q_sched.total_epochs() for q_sched in tmp_cfg.q_schedule)

    # 2. Parse CLI arguments
    parser = argparse.ArgumentParser(description='Train the Hand Gesture CNN')
    parser.add_argument('--epochs', type=int, default=max_sched_epochs, help=f'Total training epochs (Default: {max_sched_epochs})')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--fine-tune', action='store_true', help='Resume from checkpoint for fine-tuning')
    parser.add_argument('--network-path', type=str, default=None, help='Path to network for resuming/fine-tuning')
    args = parser.parse_args()

    # 3. Set parameters
    DATA_SPLIT = 0.8
    BATCH_SIZE = args.batch_size
    IN_BITS = 1
    LEARNING_RATE = args.lr
    WEIGHT_DECAY  = 1e-5
    global_max = 0.0
    
    dataset_name = "roobansappani/hand-gesture-recognition"
    datapath   = Path("nn") / "data"
    plot_path  = datapath / "training_accuracy.png"
    network_path = Path(args.network_path) if args.network_path else datapath / "gesture_net_quantized.pth"
    csv_path   = datapath / "hardware_weights.csv"
    sv_path    = datapath / "hardware_weights.vh"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # 2. Configure and Create network (to find num_classes)
    cfg = get_hand_gesture_cfg()
    num_classes = cfg.num_classes

    # 3. Prepare Data (filtered to the explicit gesture class list)
    train_loader, test_loader, num_classes = prepare_data(dataset_name, IMG_H, IMG_W, IN_BITS, DATA_SPLIT, BATCH_SIZE, target_classes=GESTURE_CLASSES)
    _, train_aug = get_transforms(IMG_H, IMG_W, IN_BITS)

    # 4. Create network
    network = cnn(config = cfg)
    network = network.to(device)
    print("\n--- NETWORK ARCHITECTURE ---")
    print(network)

    # Verify schedule
    network_layers = list(network.modules())
    assert len(cfg.q_schedule) == sum(1 for m in network_layers if isinstance(m, QuantConv2d)), "Schedule must have an entry for each QuantConv2d layer"

    # 4. Train network
    if args.fine_tune:
        print(f"\n--- RESUMING TRAINING FOR FINE-TUNING FROM {network_path} ---")
        network.load_state_dict(torch.load(network_path))
        network._is_fine_tuning = True
        # Force all layers to their final quantized state
        quant_layer_idx = 0
        for module in network.modules():
            if isinstance(module, QuantConv2d):
                module._quantize = True
                module._weight_bits = cfg.q_schedule[quant_layer_idx]._q_min_bits
                quant_layer_idx += 1
        # Get initial accuracy of loaded network
        network.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = network(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.numel()
        initial_acc = correct / total
        global_max = initial_acc
        print(f"Loaded network accuracy: {initial_acc:.3f}")

        # Run for specified epochs at fine-tune LR
        train_acc_history, test_acc_history = train_network(
            network, train_loader, test_loader, train_aug, cfg, device, global_max,
            epochs=args.epochs, lr=1e-3, weight_decay=1e-6, network_save_path=network_path,
            initial_best_acc=initial_acc, is_fine_tuning=True
        )
    else:
        EPOCHS = args.epochs
        train_acc_history, test_acc_history = train_network(
            network, train_loader, test_loader, train_aug, cfg, device, global_max,
            epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, network_save_path=network_path
        )

    # 4.5 Plot training history
    plot_training(cfg, train_acc_history, test_acc_history, plot_path)

    # 5. Export Hardware Parameters
    print("\n--- EXPORTING HARDWARE WEIGHTS ---")
    export_nn_to_csv(network_path, config=cfg, output_csv=csv_path)
    export_csv_to_hex(csv_path, sv_path, datapath / "roms" / "hex", config=cfg)
    print("Run cnn.py inference hw-eval --trials 100 ?")
    print("Run cnn.py export ?")
    print("Run cnn.py render ?")

if __name__ == "__main__":
    main()