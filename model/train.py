# model.train.py
# Bradley Manzo 2026
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path
from typing import List, Tuple, Callable

from model.config     import ModelConfig
from model.export     import export_model_to_csv, export_csv_to_sv
from model.model      import cnn_model, QuantConv2d
from model.plot       import plot_training
from model.preprocess import prepare_data, get_transforms
from model.globals    import get_hand_gesture_cfg


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, train_aug: Callable, cfg: ModelConfig, device: str, 
                epochs: int, lr: float, weight_decay: float, model_save_path: Path) -> Tuple[List[float], List[float]]:
    '''Executes the full training and progressive quantization schedule.'''
    
    begin_q_epochs = cfg.q_schedule[0]._q_start
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - begin_q_epochs)

    print(f"Training on {device} for {epochs} epochs with progressive quantization starting at epoch {begin_q_epochs}")
    print("\n--- STARTING FULL-PRECISION TRAINING ---")

    best_test_acc = 0.0
    train_acc_history: List[float] = []
    test_acc_history:  List[float] = []

    for epoch in range(epochs):
        if epoch == begin_q_epochs:
            print("\n--- STARTING PROGRESSIVE QUANTIZATION ---")
            
        # 1. Scheduling Logic
        quant_layer_idx = 0
        bits_changed_this_epoch = False
        for module in model.modules():
            if isinstance(module, QuantConv2d):
                target_bits = cfg.q_schedule[quant_layer_idx].get_target_bits(epoch)
                if target_bits is not None:
                    if not getattr(module, '_quantize', False): 
                        module._quantize = True
                        module._weight_bits = target_bits
                        bits_changed_this_epoch = True
                        print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} quantization ON -> {target_bits}-bit")
                    elif getattr(module, '_weight_bits', None) != target_bits:
                        module._weight_bits = target_bits
                        bits_changed_this_epoch = True
                        print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} dropping bits -> {target_bits}-bit")
                quant_layer_idx += 1
        
        if bits_changed_this_epoch:
            best_test_acc = 0.0

        # 2. Training Logic
        model.train()
        for x, y in train_loader:
            x    = train_aug(x) 
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss   = loss_fn(logits, y)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # 3. Evaluation Logic
        model.eval()
        train_correct = train_total = 0
        test_correct  = test_total = 0
        
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                train_correct += (pred == y).sum().item()
                train_total   += y.numel()
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                test_correct += (pred == y).sum().item()
                test_total   += y.numel()

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        if epoch >= begin_q_epochs:
            scheduler.step()

        print(f"epoch {epoch:02d} train_acc={train_acc:.3f} test_acc={test_acc:.3f}", end="")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> New best model saved! (Acc: {best_test_acc:.3f})")
        else:
            print("")

    return train_acc_history, test_acc_history

def main():
    # 1. Set top-level parameters
    IMG_H, IMG_W = 240, 320
    DATA_SPLIT = 0.8
    BATCH_SIZE = 32
    IN_BITS = 1
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY  = 1e-5
    dataset_name = "roobansappani/hand-gesture-recognition"
    datapath   = Path("model") / "data"
    plot_path  = datapath / "training_accuracy.png"
    model_path = datapath / "gesture_net_quantized.pth"
    csv_path   = datapath / "hardware_weights.csv"
    sv_path    = datapath / "hardware_weights.vh"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # 2. Prepare Data (This resolves num_classes)
    train_loader, test_loader, num_classes = prepare_data(dataset_name, IMG_H, IMG_W, IN_BITS, DATA_SPLIT, BATCH_SIZE)
    _, train_aug = get_transforms(IMG_H, IMG_W, IN_BITS)

    # 3. Configure and Create Model
    cfg = get_hand_gesture_cfg(num_classes, IMG_H, IMG_W)
    model = cnn_model(config = cfg)
    model = model.to(device)

    # Verify schedule
    model_layers = model.modules()
    assert len(cfg.q_schedule) == sum(1 for m in model_layers if isinstance(m, QuantConv2d)), "Schedule must have an entry for each QuantConv2d layer"

    # 4. Train Model
    EPOCHS = max(q_sched.total_epochs() for q_sched in cfg.q_schedule)
    train_acc_history, test_acc_history = train_model(
        model, train_loader, test_loader, train_aug, cfg, device,
        EPOCHS, LEARNING_RATE, WEIGHT_DECAY, model_path
    )

    # 4.5 Plot training history
    plot_training(cfg, train_acc_history, test_acc_history, plot_path)

    # 5. Export Hardware Parameters
    print("\n--- EXPORTING HARDWARE WEIGHTS ---")
    export_model_to_csv(model_path, config=cfg, output_csv=csv_path)
    export_csv_to_sv(csv_path, sv_path)

if __name__ == "__main__":
    main()