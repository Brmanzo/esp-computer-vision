#!/usr/bin/env python3
"""
nn/tasks/generate/train.py

Sweep all generated networks on MNIST. For each candidate:
  - Train for ABORT_EPOCH epochs; abort if test accuracy ≤ random chance (10%).
  - If passing, train to full schedule completion and save the best checkpoint.
  - Append LC, FPS, and best accuracy to results.txt as each network finishes.
"""

import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import torch

from nn.arch      import cnn, QuantConv2d
from nn.config    import NNConfig
from nn.export    import export_nn_to_csv, export_csv_to_hex
from nn.globals   import CLK_FREQ_HZ, prepare_data, CLASSES, NN_CFG
from nn.inference import get_inference_from_pixels
from nn.sweep.generate import generate_networks, _fmt

NUM_CLASSES   = len(CLASSES)
IMG_W         = NN_CFG.in_dims.width
IMG_H         = NN_CFG.in_dims.height

RANDOM_CHANCE = 1.0 / NUM_CLASSES       # baseline for random classifier
ABORT_EPOCH   = 5              # survival check after this many epochs
BATCH_SIZE    = 512
LR            = 2e-4
WEIGHT_DECAY  = 1e-5

DATA_DIR     = Path("nn") / "data"
RESULTS_PATH = Path("profiling") / "nn_acc_pred" / "profiles" / "results.txt"
CKPT_DIR     = Path("nn") / "sweep" / "checkpoints"


def _evaluate(net, test_loader, device: str, max_samples: int | None = None) -> float:
    """Evaluate test accuracy, optionally capped at max_samples for speed."""
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            if max_samples is not None and total >= max_samples:
                break
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / (total or 1)


def _fully_quantized_epoch(cfg: NNConfig) -> int:
    """First epoch at which every layer has settled at its final q_min_bits.

    Before this epoch, weights are still in a higher-precision intermediate state
    so accuracy measurements don't reflect the deployed hardware model.
    """
    return max(
        q._q_start + sum(q._epochs_per_bit[:-1])
        for q in cfg.q_schedule
    )


def _train_one(
    cfg: NNConfig,
    train_loader,
    test_loader,
    device: str,
    save_path: Path,
) -> float | None:
    """Train one network with early abort.

    Returns the best test accuracy reached AFTER all layers have settled at
    q_min_bits (i.e. the accuracy of the fully-quantized deployed model).
    Returns None if the network is aborted at ABORT_EPOCH.
    Model construction is silenced (suppresses BRAM prints and skips verilog rendering).
    """
    with patch("nn.verilog.render_verilog", lambda _: None), \
         contextlib.redirect_stdout(io.StringIO()):
        try:
            net = cnn(cfg).to(device)
        except (AssertionError, Exception) as e:
            print(f"  [skip] construction failed: {e}", flush=True)
            return None

    total_epochs    = max(q.total_epochs() for q in cfg.q_schedule)
    quantized_epoch = _fully_quantized_epoch(cfg)

    no_decay = ["bias", "bn_weight", "bn_bias", "clip_val"]
    decay_p, free_p = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        (free_p if any(nd in name for nd in no_decay) else decay_p).append(param)

    opt = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": WEIGHT_DECAY},
         {"params": free_p,  "weight_decay": 0.0}],
        lr=LR,
    )
    loss_fn   = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(train_loader),
        epochs=total_epochs, pct_start=0.2, div_factor=10, final_div_factor=100,
    )

    best_acc = 0.0
    print(f"    (quantized plateau starts at epoch {quantized_epoch}/{total_epochs - 1})",
          flush=True)

    for epoch in range(total_epochs):
        # Progressive quantization schedule
        quant_idx = 0
        for module in net.modules():
            if isinstance(module, QuantConv2d):
                target = cfg.q_schedule[quant_idx].get_target_bits(epoch)
                if target is not None:
                    if not getattr(module, "_quantize", False):
                        module._quantize    = True
                        module._weight_bits = target
                    elif getattr(module, "_weight_bits", None) != target:
                        module._weight_bits = target
                quant_idx += 1

        # Train
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

        # Abort check uses a fast 1000-sample eval; all other epochs use full eval
        is_abort_epoch = (epoch == ABORT_EPOCH - 1)
        acc = _evaluate(net, test_loader, device,
                        max_samples=1000 if is_abort_epoch else None)

        # Only track best accuracy during the final 5 epochs to ensure OneCycleLR has fully decayed
        tracking_epoch = max(quantized_epoch, total_epochs - 5)
        saved = ""
        if epoch >= tracking_epoch and acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            saved = "  *"

        tag = " [Q]" if epoch >= tracking_epoch else "    "
        print(f"    e{epoch:02d}{tag}  acc={acc:.4f}{saved}", flush=True)

        if is_abort_epoch:
            if acc <= RANDOM_CHANCE * 1.4:
                print(f"    → aborted (acc={acc:.4f} ≤ {(RANDOM_CHANCE * 1.4):.2f})", flush=True)
                return None
            print(f"    → survived, best_acc tracking starts at epoch {tracking_epoch}", flush=True)

    return best_acc if best_acc > 0.0 else None


_CSV_PATH = DATA_DIR / "hardware_weights.csv"
_VH_PATH  = DATA_DIR / "hardware_weights.vh"
_HEX_DIR  = DATA_DIR / "roms" / "hex"


def _hw_eval_one(
    cfg: NNConfig,
    save_path: Path,
    hw_test_loader,
    n_trials: int = 100,
) -> float:
    """Export weights from checkpoint and run hardware-accurate integer inference.

    Returns hw accuracy over n_trials samples.
    Export is silenced (suppresses BRAM prints and skips verilog rendering).
    """
    with patch("nn.verilog.render_verilog", lambda _: None), \
         contextlib.redirect_stdout(io.StringIO()):
        export_nn_to_csv(save_path, cfg, _CSV_PATH)
        export_csv_to_hex(_CSV_PATH, _VH_PATH, _HEX_DIR, cfg)

    correct = total = 0
    for img_t, label_t in hw_test_loader:
        if total >= n_trials:
            break
        pixels = (img_t[0].flatten() > 0.5).int().tolist()
        pred   = get_inference_from_pixels(pixels, cfg, csv_path=_CSV_PATH, vh_path=_VH_PATH)
        correct += int(pred == int(label_t[0]))
        total   += 1

    acc = correct / (total or 1)
    print(f"  → hw_acc={acc:.4f} over {total} trials", flush=True)

    # Clean up export artefacts — only the .pth checkpoint is kept
    for path in (_CSV_PATH, _VH_PATH):
        path.unlink(missing_ok=True)

    return acc


def sweep(
    out_path: Path = RESULTS_PATH,
    hw_trials: int = 100,
    start_from: int = 0,
    indices: list[int] | None = None,
) -> None:
    """Run the training sweep.

    start_from: skip networks with index < this value and append to an existing
                results file rather than overwriting it. Use this to resume after
                an interrupted run (e.g. start_from=32 to continue after network 31).
    indices: train only a specific list of indices. Appends to existing results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _HEX_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, _ = prepare_data(DATA_DIR, IMG_H, IMG_W, BATCH_SIZE)
    # hw eval must use the same test set (clean transforms, no shuffle) that float best_acc uses;
    # the first return value is the train loader (augmented, shuffled) which is the wrong split.
    _, hw_test_loader, _         = prepare_data(DATA_DIR, IMG_H, IMG_W, batch_size=1)

    from nn.globals import NN_CFG
    configs = generate_networks(NN_CFG)
    total   = len(configs)
    
    if indices is not None:
        print(f"\nEvaluating {len(indices)} specific networks out of {total}\n")
    else:
        print(f"\n{total} networks to evaluate (starting from #{start_from})\n")

    file_mode = "a" if (start_from > 0 or indices is not None) else "w"
    with open(out_path, file_mode, buffering=1) as f:
        if start_from == 0 and indices is None:
            f.write(f"{'idx':>4}  {'float_acc':>9}  {'hw_acc':>6}  arch\n")
            f.write("─" * 110 + "\n")

        for i, (lc, cfg) in enumerate(configs):
            if indices is not None and i not in indices:
                continue
            if indices is None and i < start_from:
                continue

            header = _fmt(lc, cfg)
            print(f"\n[{i:3d}/{total - 1}] {header}", flush=True)

            save_path = CKPT_DIR / f"network_{i:04d}.pth"
            best_acc  = _train_one(cfg, train_loader, test_loader, device, save_path)

            if best_acc is None:
                f.write(f"{i:4d}  {'aborted':>9}  {'':>6}  {header}\n")
                print(f"  → aborted", flush=True)
                continue

            print(f"  → float best_acc={best_acc:.4f}", flush=True)
            hw_acc = _hw_eval_one(cfg, save_path, hw_test_loader, n_trials=hw_trials)

            f.write(f"{i:4d}  {best_acc:9.4f}  {hw_acc:6.4f}  {header}\n")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-from", type=int, default=0,
                    help="Resume from this network index (appends to existing results.txt)")
    ap.add_argument("--indices", type=str, default=None,
                    help="Comma-separated list of network indices to train (e.g. 0,5,10). Appends to results.txt")
    ap.add_argument("--hw-trials", type=int, default=100)
    args = ap.parse_args()
    
    indices_list = None
    if args.indices:
        indices_list = [int(x.strip()) for x in args.indices.split(",")]

    sweep(start_from=args.start_from, hw_trials=args.hw_trials, indices=indices_list)
