"""Create the gated trainer from the checked baseline; preserve baseline trainer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "scripts/train_m0_experiments.py").read_text()


def replace(old, new):
    global source
    if source.count(old) != 1:
        raise SystemExit(f"Baseline differs; cannot patch safely: {old[:90]!r}")
    source = source.replace(old, new, 1)


replace('import torch\n', '''import torch
from common.training.dual_gate import (
    BranchGates, task_gradients, assign_gradients, lookahead_losses, update_gates
)
from common.training.dual_gate_run import (
    gate_loaders, atomic_save, capture_rng, restore_rng, GateJournal
)
''')

start = source.index('from common.training.dual_gate import (')
end = source.index('\nROOT = ', start)
imports = source[start:end]
source = source[:start] + source[end:]
replace('sys.path.insert(0, str(ROOT))\n',
        'sys.path.insert(0, str(ROOT))\n\n' + imports)

replace('    return parser.parse_args()', '''    parser.add_argument("--gate-mode", choices=["learned", "fixed"], default="learned")
    parser.add_argument("--meta-count", type=int, default=40)
    parser.add_argument("--meta-seed", type=int, default=2042)
    parser.add_argument("--gate-every", type=int, default=5)
    parser.add_argument("--gate-lr", type=float, default=0.01)
    return parser.parse_args()''')

replace('    args = parse_args()\n    s = build_settings(args)', '''    args = parse_args()
    args.gradient_conflict = True
    if args.smoke:
        args.meta_count = 2
    if args.gate_every < 1 or args.gate_lr <= 0:
        raise SystemExit("gate-every and gate-lr must be positive")
    s = build_settings(args)
    s.update(gate_mode=args.gate_mode, meta_count=args.meta_count,
             meta_seed=args.meta_seed, gate_every=args.gate_every, gate_lr=args.gate_lr,
             gate_protocol="two independent gates; diagonal task meta losses; SGD lookahead")''')

replace('    family = model_config.get("model", "M0")', '''    family = model_config.get("model", "M0")
    if family != "M0" or s["label_source"] != "official":
        raise SystemExit("This experiment supports M0 with official labels only")''')

replace('    name = args.run_name or (', '''    name = args.run_name or f"dual_gate_{args.gate_mode}_seed{args.seed}"
    baseline_default_name = (''')

replace('    model_config = json.loads((ROOT / s["model_config"]).read_text(\n        encoding="utf-8"))', '''    train_loader, meta_loader, split_report = gate_loaders(
        train_loader, train_set, s, args.meta_count, args.meta_seed
    )
    s["gate_split"] = split_report
    (folder / "gate_split.json").write_text(
        json.dumps(split_report, indent=2) + "\\n", encoding="utf-8")
    log("[split] model train", len(train_loader.dataset), "| gate meta",
        len(meta_loader.dataset), "| official val", len(val_loader.dataset))
    model_config = json.loads((ROOT / s["model_config"]).read_text(
        encoding="utf-8"))''')

replace('    base_lrs = [g["lr"] for g in optimizer.param_groups]', '''    base_lrs = [g["lr"] for g in optimizer.param_groups]
    gates = BranchGates().to(device)
    gate_optimizer = torch.optim.Adam(gates.parameters(), lr=args.gate_lr)
    log("[gate]", args.gate_mode, gates.values(), "| update every", args.gate_every,
        "batches | lookahead: SGD surrogate for real AdamW")''')

replace('    loss_kwargs = {"gamma":', '''    if amp_dtype == torch.float16:
        raise SystemExit("Dual gate requires BF16 or FP32; use --no-amp on this GPU")
    loss_kwargs = {"gamma":''')

replace('"epochs", "amp", "gradient_conflict", "model_config_snapshot")', '''"epochs", "amp", "gradient_conflict", "model_config_snapshot",
                       "gate_mode", "meta_count", "meta_seed", "gate_every", "gate_lr",
                       "gate_protocol", "gate_split")''')

replace('        model.load_state_dict(state["model"])', '''        gates.load_state_dict(state["gates"])
        gate_optimizer.load_state_dict(state["gate_optimizer"])
        model.load_state_dict(state["model"])''')

replace('    for epoch in range(start_epoch, s["epochs"]):', '''    journal = GateJournal(folder, start_epoch + 1)
    if args.resume:
        restore_rng(state["rng"])
    for epoch in range(start_epoch, s["epochs"]):''')

replace('        train_set.set_epoch(epoch)', '''        train_set.set_epoch(epoch)
        train_loader.generator.manual_seed(s["seed"] + epoch)
        meta_loader.generator.manual_seed(args.meta_seed + epoch)
        meta_iterator = iter(meta_loader) if args.gate_mode == "learned" else None''')

begin = source.index('            if monitor:\n                monitor.observe(')
end = source.index('\n        if monitor:\n            summary = monitor.finish', begin)

source = source[:begin] + '''            raw = task_gradients(model, losses["cell"],
                                 s["tissue_loss_weight"] * losses["tissue"])
            if monitor:
                monitor.observe(None, None, index + 1, batch["pair_id"],
                                precomputed_gradients=raw)
            n = images[0].shape[0]
            for key, value in (("cell", losses["cell"]),
                               ("tissue", losses["tissue"]), ("total", total)):
                sums[key] += value.item() * n
            seen += n
            del out, losses, total, value

            factor = cosine_factor(step, total_steps)
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            before = gates.values()
            info, meta_ids = {}, []
            if args.gate_mode == "learned" and step % args.gate_every == 0:
                try:
                    meta_batch = next(meta_iterator)
                except StopIteration:
                    meta_iterator = iter(meta_loader)
                    meta_batch = next(meta_iterator)
                mi, mt, mb = prepare(meta_batch, device, s["size"])
                meta_ids = meta_batch["pair_id"]
                if set(meta_ids) & set(batch["pair_id"]):
                    raise RuntimeError("Training/meta batch overlap")
                rates = {id(p): group["lr"] for group in optimizer.param_groups
                         for p in group["params"]}
                rates = {name: rates[id(p)] for name, p in model.named_parameters()}
                def meta_loss(output):
                    return m0_task_losses(output["cell_logits"], output["tissue_logits"],
                                          *mt, **loss_kwargs)
                with cast:
                    meta_losses = lookahead_losses(
                        model, gates, raw, rates, (*mi, mb), meta_loss,
                        grad_clip_norm=s.get("grad_clip_norm")
                    )
                meta_values = {"meta_cell": meta_losses["cell"].item(),
                               "meta_tissue": meta_losses["tissue"].item()}
                info = update_gates(gates, gate_optimizer, meta_losses)
                info.update(meta_values)
                del meta_losses, mi, mt, mb, meta_batch
            assign_gradients(model, gates, raw)
            del raw
            if s.get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip_norm"],
                                               error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            journal.record(epoch + 1, index + 1, step, frozen, args.gate_mode,
                           before, gates, info, batch["pair_id"], meta_ids)
            step += 1
''' + source[end:]

replace('        val, records, _ = evaluate(', '''        log("[gate]", journal.finish(epoch + 1, len(train_loader)))
        val, records, _ = evaluate(''')

replace('        if val["mF1"] > best:', '''        if device.type == "cuda":
            log("[gpu] peak allocated GiB:", round(torch.cuda.max_memory_allocated(device)/1024**3, 2),
                "| peak reserved GiB:", round(torch.cuda.max_memory_reserved(device)/1024**3, 2))
        gate_state = dict(gates=gates.state_dict(),
                          gate_optimizer=gate_optimizer.state_dict(), rng=capture_rng())
        if val["mF1"] > best:''')

replace('"val": val, "settings": s}, folder / "best.pt")',
        '"val": val, "settings": s, **gate_state}, folder / "best.pt")')

replace('"step": step, "best": best, "settings": s},',
        '"step": step, "best": best, "settings": s, **gate_state},')

assert source.count('torch.save(') == 2
source = source.replace('torch.save(', 'atomic_save(')
source = (
    '"""Independent dual-gate experiment. Generated from the baseline trainer.\n'
    'See common/training/dual_gate.py for the gate protocol.\n"""\n' + source
)
compile(source, "train_m0_dual_gate.py", "exec")

monitor_path = ROOT / "common/training/gradient_conflict.py"
monitor = monitor_path.read_text()
if 'precomputed_gradients=None' not in monitor:
    old = '''    def observe(self, cell_loss, tissue_loss, batch_index, pair_ids):
        parameters = [p for _, p in self.named]
        gc = torch.autograd.grad(cell_loss, parameters, retain_graph=True,
                                 allow_unused=True)
        gt = torch.autograd.grad(tissue_loss, parameters, retain_graph=True,
                                 allow_unused=True)'''
    new = '''    def observe(self, cell_loss, tissue_loss, batch_index, pair_ids,
                precomputed_gradients=None):
        if precomputed_gradients is None:
            parameters = [p for _, p in self.named]
            gc = torch.autograd.grad(cell_loss, parameters, retain_graph=True,
                                     allow_unused=True)
            gt = torch.autograd.grad(tissue_loss, parameters, retain_graph=True,
                                     allow_unused=True)
        else:
            pairs = [precomputed_gradients[n] for n, _ in self.named]
            gc, gt = zip(*pairs)'''
    if monitor.count(old) != 1:
        raise SystemExit("Gradient monitor differs; refusing unsafe patch")
    monitor = monitor.replace(old, new, 1)

compile(monitor, str(monitor_path), "exec")
destination = ROOT / "scripts/train_m0_dual_gate.py"
if destination.exists() and destination.read_text() != source:
    raise SystemExit("Gated trainer exists with different contents; refusing overwrite")

destination.write_text(source)
monitor_path.write_text(monitor)
print("[PASS] gated trainer generated; original trainer preserved")
print("[PASS] monitor supports reuse of raw gradients")
