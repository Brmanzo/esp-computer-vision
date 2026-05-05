# model.train.py
# Bradley Manzo 2026
import torch
import torch.nn as nn
import argparse
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
                epochs: int, lr: float, weight_decay: float, model_save_path: Path, initial_best_acc: float = 0.0, is_fine_tuning: bool = False) -> Tuple[List[float], List[float]]:
    '''Executes the full training and progressive quantization schedule.'''
    
    begin_q_epochs = cfg.q_schedule[0]._q_start
    # Switch to AdamW for better weight decay handling in QAT
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    if not getattr(model, '_is_fine_tuning', False):
        print("\n--- STARTING FULL-PRECISION TRAINING ---")
    else:
        print("\n--- CONTINUING FINE-TUNING AT FINAL QUANTIZATION ---")

    best_test_acc = initial_best_acc
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
                        if not is_fine_tuning: print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} quantization ON -> {target_bits}-bit")
                    elif getattr(module, '_weight_bits', None) != target_bits:
                        module._weight_bits = target_bits
                        bits_changed_this_epoch = True
                        if not is_fine_tuning: print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} dropping bits -> {target_bits}-bit")
                quant_layer_idx += 1
        
        if bits_changed_this_epoch and not getattr(model, '_is_fine_tuning', False):
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
            
            # Step the scheduler every BATCH
            scheduler.step()

        # 3. Evaluation Logic
        model.eval()
        train_total = 1e-6 
        train_correct = 0.0
        
        with torch.no_grad():
            if not getattr(model, '_is_fine_tuning', False):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    train_correct += (pred == y).sum().item()
                    train_total   += y.numel()
                train_acc = train_correct / train_total
            else:
                train_acc = 0.0
            
            test_correct  = test_total = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                test_correct += (pred == y).sum().item()
                test_total   += y.numel()
            test_acc = test_correct / (test_total + 1e-6)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        print(f"epoch {epoch:02d} train_acc={train_acc:.3f} test_acc={test_acc:.3f}", end="")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> New best model saved! (Acc: {best_test_acc:.3f})")
        else:
            print("")

    return train_acc_history, test_acc_history

def main():
    # 1. Configure Model Baseline (to find default epochs)
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
    parser.add_argument('--model-path', type=str, default=None, help='Path to model for resuming/fine-tuning')
    args = parser.parse_args()

    # 3. Set parameters
    DATA_SPLIT = 0.8
    BATCH_SIZE = args.batch_size
    IN_BITS = 1
    LEARNING_RATE = args.lr
    WEIGHT_DECAY  = 1e-5
    
    dataset_name = "roobansappani/hand-gesture-recognition"
    datapath   = Path("model") / "data"
    plot_path  = datapath / "training_accuracy.png"
    model_path = Path(args.model_path) if args.model_path else datapath / "gesture_net_quantized.pth"
    csv_path   = datapath / "hardware_weights.csv"
    sv_path    = datapath / "hardware_weights.vh"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # 2. Configure and Create Model (to find num_classes)
    cfg = get_hand_gesture_cfg()
    num_classes = cfg.num_classes

    # 3. Prepare Data (This constrained by num_classes)
    train_loader, test_loader, num_classes = prepare_data(dataset_name, IMG_H, IMG_W, IN_BITS, DATA_SPLIT, BATCH_SIZE, max_classes=num_classes)
    _, train_aug = get_transforms(IMG_H, IMG_W, IN_BITS)

    # 4. Create Model
    model = cnn_model(config = cfg)
    model = model.to(device)
    print("\n--- MODEL ARCHITECTURE ---")
    print(model)

    # Verify schedule
    model_layers = list(model.modules())
    assert len(cfg.q_schedule) == sum(1 for m in model_layers if isinstance(m, QuantConv2d)), "Schedule must have an entry for each QuantConv2d layer"

    # 4. Train Model
    if args.fine_tune:
        print(f"\n--- RESUMING TRAINING FOR FINE-TUNING FROM {model_path} ---")
        model.load_state_dict(torch.load(model_path))
        model._is_fine_tuning = True
        # Force all layers to their final quantized state
        quant_layer_idx = 0
        for module in model.modules():
            if isinstance(module, QuantConv2d):
                module._quantize = True
                module._weight_bits = cfg.q_schedule[quant_layer_idx]._q_min_bits
                quant_layer_idx += 1
        # Get initial accuracy of loaded model
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.numel()
        initial_acc = correct / total
        print(f"Loaded model accuracy: {initial_acc:.3f}")

        # Run for specified epochs at fine-tune LR
        train_acc_history, test_acc_history = train_model(
            model, train_loader, test_loader, train_aug, cfg, device,
            epochs=args.epochs, lr=1e-3, weight_decay=1e-6, model_save_path=model_path,
            initial_best_acc=initial_acc, is_fine_tuning=True
        )
    else:
        EPOCHS = args.epochs
        train_acc_history, test_acc_history = train_model(
            model, train_loader, test_loader, train_aug, cfg, device,
            epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, model_save_path=model_path
        )

    # 4.5 Plot training history
    plot_training(cfg, train_acc_history, test_acc_history, plot_path)

    # 5. Export Hardware Parameters
    print("\n--- EXPORTING HARDWARE WEIGHTS ---")
    export_model_to_csv(model_path, config=cfg, output_csv=csv_path)
    export_csv_to_sv(csv_path, sv_path)

if __name__ == "__main__":
    main()