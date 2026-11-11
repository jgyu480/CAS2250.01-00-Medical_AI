#!/usr/bin/env bash
# VESSL recovery and automatic publication for the five-model OCELOT ensemble.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1 GIT_TERMINAL_PROMPT=0
trap 'printf "[failed] line %s. See the launcher log. Restart with the same ENSEMBLE_ROOT and ENSEMBLE_RUN_ID.\n" "$LINENO" >&2' ERR

RUN_ID=${ENSEMBLE_RUN_ID:-five_models_seed42}
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || { echo 'Invalid ENSEMBLE_RUN_ID' >&2; exit 2; }
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --publish-only ]]; }; then
  echo 'Usage: bash run_ocelot_ensemble.sh [--publish-only]' >&2
  exit 2
fi
STATE=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
mkdir -p "$STATE"
STATE=$(cd "$STATE" && pwd)
SELF=$(realpath "$0")
REPO="$STATE/repo"
ASSETS="$STATE/assets"
M2_REF=results/m2-overnight-20261110-121336
REPO_REV=14f930ac47d69bb9abc6d86c6abcb5bdaf35a8da
BRANCH="results/ensemble-$RUN_ID"
RESULT="$STATE/experiments/$RUN_ID"
exec 9>"$STATE/run.lock"
flock -n 9 || { echo '[running] This ENSEMBLE_ROOT already has an active job.' >&2; exit 1; }
mkdir -p "$ASSETS"

if [[ ! -d "$REPO/.git" ]]; then
  echo '[setup] Clone Medical AI repository'
  clone_source=${ENSEMBLE_GIT_URL:-https://github.com/jgyu480/CAS2250.01-00-Medical_AI.git}
  GIT_LFS_SKIP_SMUDGE=1 git clone --branch main --single-branch "$clone_source" "$REPO"
fi
cd "$REPO"
if [[ $(git rev-parse HEAD) != "$REPO_REV" ]]; then
  if ! git cat-file -e "$REPO_REV^{commit}" 2>/dev/null; then
    git fetch origin refs/heads/main
  fi
  GIT_LFS_SKIP_SMUDGE=1 git checkout --detach "$REPO_REV"
fi

publish() {
  local index tree parent remote_listing current commit ref destination artifact
  test -f "$RESULT/complete.ok"
  test -s "$RESULT/summary.csv"
  test -s "$RESULT/fusion.pt"
  test -s "$RESULT/run.json"
  ref="refs/heads/$BRANCH"
  destination="results/ensemble/$RUN_ID"
  remote_listing=$(git ls-remote origin "$ref")
  parent=$REPO_REV
  if [[ -n "$remote_listing" ]]; then
    git fetch --no-tags origin "$ref"
    parent=$(git rev-parse FETCH_HEAD)
  fi
  index=$(mktemp "$STATE/publish-index.XXXXXX")
  rm "$index"
  (
    export GIT_INDEX_FILE="$index"
    trap 'rm -f "$GIT_INDEX_FILE"' EXIT
    git read-tree "$parent"
    for artifact in summary.csv run.json fusion.pt fusion_last.pt; do
      [[ -s "$RESULT/$artifact" ]] || continue
      git update-index --add --cacheinfo "100644,$(git hash-object -w "$RESULT/$artifact"),$destination/$artifact"
    done
    git update-index --add --cacheinfo "100755,$(git hash-object -w "$SELF"),$destination/run_ocelot_ensemble.sh"
    tree=$(git write-tree)
    current=$(git rev-parse "$parent^{tree}")
    if [[ "$tree" == "$current" && -n "$remote_listing" ]]; then
      echo "[GitHub] Already uploaded: $BRANCH"
      exit 0
    fi
    commit=$(git -c user.name='OCELOT Experiments' -c user.email='ocelot-experiments@users.noreply.github.com' \
      commit-tree "$tree" -p "$parent" -m "Save five-model OCELOT ensemble experiment $RUN_ID")
    for attempt in 1 2 3; do
      if git push origin "$commit:$ref"; then
        echo "[GitHub] Uploaded: $BRANCH ($commit)"
        exit 0
      fi
      if [[ $attempt != 3 ]]; then sleep 5; fi
    done
    echo "[GitHub] Push failed. Fix Git authentication and rerun: bash $SELF --publish-only" >&2
    exit 1
  )
}

if [[ ${1:-} == --publish-only || -f "$RESULT/complete.ok" ]]; then
  publish
  exit 0
fi

if [[ -n ${ENSEMBLE_PYTHON:-} ]]; then
  PYTHON=$ENSEMBLE_PYTHON
elif [[ -x "$STATE/venv/bin/python" ]]; then
  PYTHON="$STATE/venv/bin/python"
elif [[ -x /opt/conda/envs/cas2250-medical-ai/bin/python ]]; then
  PYTHON=/opt/conda/envs/cas2250-medical-ai/bin/python
else
  PYTHON=python
fi
if ! command -v "$PYTHON" >/dev/null; then PYTHON=python3; fi
if ! "$PYTHON" -c 'import torch, torchvision; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  if ! nvidia-smi -L; then
    echo '[setup] CUDA GPU is not visible. Restart VESSL with a GPU attached.' >&2
    exit 1
  fi
  echo '[setup] Install CUDA PyTorch into the experiment virtual environment'
  [[ -x "$STATE/venv/bin/python" ]] || "$PYTHON" -m venv --system-site-packages "$STATE/venv"
  PYTHON="$STATE/venv/bin/python"
  "$PYTHON" -c 'import torch, torchvision; assert torch.cuda.is_available()' >/dev/null 2>&1 || \
    "$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch==2.3.1' 'torchvision==0.18.1'
fi
"$PYTHON" -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA GPU is not available"; print("[GPU]", torch.cuda.get_device_name(0))'
if ! "$PYTHON" -c 'import numpy as np, scipy, PIL, torch; torch.from_numpy(np.zeros(1, dtype=np.float32))'; then
  [[ -x "$STATE/venv/bin/python" ]] || "$PYTHON" -m venv --system-site-packages "$STATE/venv"
  PYTHON="$STATE/venv/bin/python"
  "$PYTHON" -m pip install 'numpy<2' 'scipy<1.16' Pillow
fi
"$PYTHON" -c 'import numpy as np, torch; torch.from_numpy(np.zeros(1, dtype=np.float32))'

if [[ -n ${OCELOT_DATA_ROOT:-} ]]; then
  test -f "$OCELOT_DATA_ROOT/metadata.json"
else
  for path in "$ASSETS/ocelot2023_v1.0.1" /root/repo/data/raw/ocelot2023_v1.0.1 \
      /mnt/data/ocelot2023_v1.0.1 /workspace/data/ocelot2023_v1.0.1 \
      /root/m2_overnight/assets/ocelot2023_v1.0.1 "$REPO/data/raw/ocelot2023_v1.0.1"; do
    if [[ -f "$path/metadata.json" ]]; then OCELOT_DATA_ROOT=$path; break; fi
  done
fi
if [[ -z ${OCELOT_DATA_ROOT:-} ]]; then
  zip="$ASSETS/ocelot2023_v1.0.1.zip"
  for path in /root/m2_overnight/assets/ocelot2023_v1.0.1.zip /root/repo/data/downloads/ocelot2023_v1.0.1.zip; do
    if [[ ! -f "$zip" && -f "$path" ]]; then zip=$path; fi
  done
  if [[ ! -f "$zip" ]]; then
    echo '[setup] Download the official OCELOT archive'
    curl -fLsS --retry 4 --continue-at - \
      'https://zenodo.org/records/8417503/files/ocelot2023_v1.0.1.zip?download=1' -o "$zip.part"
    mv "$zip.part" "$zip"
  fi
  printf '%s  %s\n' 215230295b0440b4dc356519ef9ff644 "$zip" | md5sum -c -
  echo '[setup] Extract OCELOT'
  unpack=$(mktemp -d "$ASSETS/extract.XXXXXX")
  "$PYTHON" -m zipfile -e "$zip" "$unpack"
  test -f "$unpack/ocelot2023_v1.0.1/metadata.json"
  mv "$unpack/ocelot2023_v1.0.1" "$ASSETS/"
  rmdir "$unpack" || true
  OCELOT_DATA_ROOT="$ASSETS/ocelot2023_v1.0.1"
fi
export OCELOT_DATA_ROOT="$(realpath "$OCELOT_DATA_ROOT")"
export OUTPUT_ROOT="$STATE/outputs"
mkdir -p "$OUTPUT_ROOT"
if [[ ! -s "$OUTPUT_ROOT/audit/report.json" || ! -s "$OUTPUT_ROOT/audit/manifest.csv" || ! -s "$STATE/data-root.txt" ]] ||
   [[ $(cat "$STATE/data-root.txt") != "$OCELOT_DATA_ROOT" ]]; then
  "$PYTHON" -u scripts/check_data.py
  printf '%s\n' "$OCELOT_DATA_ROOT" > "$STATE/data-root.txt"
fi

find_checkpoint() {
  local path
  for path in "$@"; do
    if [[ -f "$path" && -s "$path" ]] && ! LC_ALL=C grep -aqm1 '^version https://git-lfs.github.com/spec/' <(head -c 64 "$path"); then
      realpath "$path"
      return 0
    fi
  done
  return 1
}

checkpoint_override() {
  local name=$1 path=$2
  shift 2
  if [[ -n "$path" ]]; then
    find_checkpoint "$path" || { echo "[checkpoint] Invalid $name override: $path" >&2; return 2; }
  else
    find_checkpoint "$@"
  fi
}

G_RUN=M2_gaussian_s6_l0.9_official_seed42
GA_RUN=M2_ha_s6_l0.9_official_seed42
G_LFS="results/m2_overnight/runs/$G_RUN/best.pt"
GA_LFS="results/m2_overnight/runs/$GA_RUN/best.pt"
G_PATH=$(checkpoint_override G "${G_CHECKPOINT:-}" "$ASSETS/G.pt" "/root/m2_overnight/outputs/m2/$G_RUN/best.pt" "/root/repo/outputs/m2/$G_RUN/best.pt") || [[ $? == 1 ]]
GA_PATH=$(checkpoint_override GA "${GA_CHECKPOINT:-}" "$ASSETS/GA.pt" "/root/m2_overnight/outputs/m2/$GA_RUN/best.pt" "/root/repo/outputs/m2/$GA_RUN/best.pt") || [[ $? == 1 ]]
if [[ -z "$G_PATH" || -z "$GA_PATH" ]]; then
  echo '[setup] Restore original G and GA from the existing GitHub results branch'
  if ! git lfs version >/dev/null 2>&1; then
    if [[ $(id -u) != 0 ]] || ! command -v apt-get >/dev/null; then
      echo '[setup] Install git-lfs or set G_CHECKPOINT and GA_CHECKPOINT.' >&2
      exit 1
    fi
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs
  fi
  git fetch --no-tags origin "refs/heads/$M2_REF:refs/remotes/origin/$M2_REF"
  git lfs fetch --include="$G_LFS,$GA_LFS" --exclude='' origin "refs/remotes/origin/$M2_REF"
  if [[ -z "$G_PATH" ]]; then
    git show "refs/remotes/origin/$M2_REF:$G_LFS" | git lfs smudge "$G_LFS" > "$ASSETS/G.pt.part"
    find_checkpoint "$ASSETS/G.pt.part" >/dev/null
    mv "$ASSETS/G.pt.part" "$ASSETS/G.pt"
    G_PATH="$ASSETS/G.pt"
  fi
  if [[ -z "$GA_PATH" ]]; then
    git show "refs/remotes/origin/$M2_REF:$GA_LFS" | git lfs smudge "$GA_LFS" > "$ASSETS/GA.pt.part"
    find_checkpoint "$ASSETS/GA.pt.part" >/dev/null
    mv "$ASSETS/GA.pt.part" "$ASSETS/GA.pt"
    GA_PATH="$ASSETS/GA.pt"
  fi
fi

ORIGINAL_OUTPUTS=${OCELOT_ORIGINAL_OUTPUTS:-/root/repo/outputs}
M0_PATH=$(checkpoint_override M0 "${M0_CHECKPOINT:-}" "$ASSETS/M0.pt" "$ORIGINAL_OUTPUTS/m0/M0_dicefix_seed42/best.pt" /tmp/ocelot_training/m0/M0_dicefix_seed42/best.pt) || [[ $? == 1 ]]
M1_PATH=$(checkpoint_override M1 "${M1_CHECKPOINT:-}" "$ASSETS/M1.pt" "$ORIGINAL_OUTPUTS/m1/M1_dicefix_seed42/best.pt" /tmp/ocelot_training/m1/M1_dicefix_seed42/best.pt) || [[ $? == 1 ]]
M11_PATH=$(checkpoint_override M1-1 "${M11_CHECKPOINT:-}" "$ASSETS/M1-1.pt" "$ORIGINAL_OUTPUTS/m1_1/M1_1_seed42/best.pt" /tmp/ocelot_training/m1_1/M1_1_seed42/best.pt) || [[ $? == 1 ]]

if [[ -z "$M0_PATH" || -z "$M1_PATH" || -z "$M11_PATH" ]]; then
  if [[ -z ${SSL_WEIGHTS_PATH:-} ]]; then
    SSL_WEIGHTS_PATH=$(find_checkpoint /root/repo/weights/bt_rn50_ep200.torch /root/m2_overnight/assets/bt_rn50_ep200.torch "$ASSETS/bt_rn50_ep200.torch") || true
    SSL_WEIGHTS_PATH=${SSL_WEIGHTS_PATH:-$ASSETS/bt_rn50_ep200.torch}
  fi
  export SSL_WEIGHTS_PATH
  mkdir -p "$(dirname "$SSL_WEIGHTS_PATH")"
  sha=$("$PYTHON" -c 'import json; print(json.load(open("configs/components.json"))["checkpoint_sha256"])')
  if [[ ! -s "$SSL_WEIGHTS_PATH" ]]; then
    curl -fLsS --retry 4 \
      'https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/bt_rn50_ep200.torch' \
      -o "$SSL_WEIGHTS_PATH.part"
    printf '%s  %s\n' "$sha" "$SSL_WEIGHTS_PATH.part" | sha256sum -c -
    mv "$SSL_WEIGHTS_PATH.part" "$SSL_WEIGHTS_PATH"
  else
    printf '%s  %s\n' "$sha" "$SSL_WEIGHTS_PATH" | sha256sum -c -
  fi
fi

train_missing() {
  local kind=$1 run=$2 teacher=${3:-} dir
  local -a args
  dir="$OUTPUT_ROOT/$kind/$run"
  if [[ -s "$dir/best.pt" && -f "$dir/complete.ok" ]]; then return 0; fi
  args=(--device cuda --seed 42 --run-name "$run" --batch-size 2)
  if [[ -s "$dir/last.pt" ]] && "$PYTHON" -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); assert "epoch" in checkpoint and "model" in checkpoint' "$dir/last.pt"; then
    args+=(--resume "$dir/last.pt")
  else
    if [[ -f "$dir/last.pt" ]]; then
      echo "[base] $kind: unreadable last.pt. Keeping interrupted run and restarting."
      mv "$dir" "$dir.interrupted-$(date +%s)"
    fi
    if [[ "$kind" == m1_1 ]]; then args+=(--m1-checkpoint "$teacher"); fi
  fi
  echo "[base] $kind: train or resume $run"
  "$PYTHON" -u "scripts/train_$kind.py" "${args[@]}"
  test -s "$dir/best.pt"
  touch "$dir/complete.ok"
}

if [[ -z "$M0_PATH" ]]; then
  train_missing m0 M0_recovery_seed42
  M0_PATH="$OUTPUT_ROOT/m0/M0_recovery_seed42/best.pt"
fi
if [[ -z "$M1_PATH" ]]; then
  train_missing m1 M1_recovery_seed42
  M1_PATH="$OUTPUT_ROOT/m1/M1_recovery_seed42/best.pt"
fi
if [[ -z "$M11_PATH" ]]; then
  train_missing m1_1 M1_1_recovery_seed42 "$M1_PATH"
  M11_PATH="$OUTPUT_ROOT/m1_1/M1_1_recovery_seed42/best.pt"
fi
printf '[checkpoints]\n  G: %s\n  GA: %s\n  M0: %s\n  M1: %s\n  M1-1: %s\n' "$G_PATH" "$GA_PATH" "$M0_PATH" "$M1_PATH" "$M11_PATH"

cat > "$STATE/ocelot_ensemble.py.part" <<'PYTHON_OCELOT_ENSEMBLE'
#!/usr/bin/env python3
"""Train a convolutional OCELOT cell-detection ensemble from frozen checkpoints."""

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

DEFAULT_MODELS = ('G', 'GA', 'M0', 'M1', 'M1-1')
M2_RUNS = {'G': 'M2_gaussian_s6_l0.9_official_seed42',
           'GA': 'M2_ha_s6_l0.9_official_seed42'}
OTHER_RUNS = {'M0': ('m0', 'M0_dicefix_seed42'),
              'M1': ('m1', 'M1_dicefix_seed42'),
              'M1-1': ('m1_1', 'M1_1_seed42')}
GRID = (0., 1., 3.), (3, 5, 7), (.05, .1, .2, .3, .4, .5, .6)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='Ordered model names used as CNN inputs.')
    parser.add_argument('--model', action='append', default=[], metavar='NAME=BEST.PT',
                        help='Override a default checkpoint or register a new one. Repeat as needed.')
    parser.add_argument('--output', type=Path, default=Path('outputs/ensemble'))
    parser.add_argument('--run-id', help='Resume the same named experiment in --output.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--crops-per-image', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', action='store_true', help='Also evaluate frozen settings on official test.')
    parser.add_argument('--smoke', action='store_true', help='Two pairs per side, one epoch, no test.')
    args = parser.parse_args()
    if args.run_id and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_id):
        parser.error('invalid --run-id')
    if not 2 <= len(args.models) or len(set(args.models)) != len(args.models):
        parser.error('--models requires at least two distinct names')
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', name) for name in args.models):
        parser.error('model names may contain only letters, numbers, underscore, dot and hyphen')
    if min(args.epochs, args.batch_size, args.crops_per_image) < 1 or args.workers < 0:
        parser.error('epochs, batch size and crops per image must be positive, workers nonnegative')
    if not 32 <= args.crop <= 1024 or (args.smoke and args.test):
        parser.error('crop must be 32–1024, and --smoke cannot be combined with --test')
    args.overrides = {}
    for entry in args.model:
        name, sep, value = entry.partition('=')
        if not sep or not value or name not in args.models or name in args.overrides:
            parser.error(f'invalid, unused or duplicate checkpoint override: {entry}')
        args.overrides[name] = Path(value).expanduser()
    return args


def checkpoint_paths(args, repo_outputs):
    paths, missing = {}, []
    for name in args.models:
        if name in args.overrides:
            candidates = [args.overrides[name]]
        elif name in M2_RUNS:
            run = M2_RUNS[name]
            candidates = [Path('/root/m2_overnight/outputs/m2') / run / 'best.pt',
                          repo_outputs / 'm2' / run / 'best.pt',
                          args.repo / 'results/m2_overnight/runs' / run / 'best.pt']
        elif name in OTHER_RUNS:
            group, run = OTHER_RUNS[name]
            candidates = [Path('/tmp/ocelot_training') / group / run / 'best.pt',
                          repo_outputs / group / run / 'best.pt',
                          args.repo / 'outputs' / group / run / 'best.pt']
        else:
            candidates = []
        for path in dict.fromkeys(candidates):
            if path.is_file():
                with path.open('rb') as handle:
                    if handle.read(64).startswith(b'version https://git-lfs.github.com/spec/'):
                        continue
                paths[name] = path.resolve()
                break
        else:
            tried = ', '.join(map(str, dict.fromkeys(candidates))) or 'no default path'
            missing.append(f'{name}: {tried}\n  Supply --model {name}=/actual/path/best.pt (full file, not Git LFS pointer).')
    if missing:
        raise SystemExit('Missing checkpoint(s):\n' + '\n'.join(missing))
    return paths


def load_base(path):
    from common.backbones.ssl_resnet import SSLResNet50
    from common.paths import PROJECT_ROOT
    from models.M0.model import M0
    from models.M1.model import M1
    from models.M1_1.model import M1_1
    from models.M2.model import M2

    # Only load trusted project checkpoints. A full state dict contains both encoders.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    settings = saved['settings']
    config_path = PROJECT_ROOT / settings['model_config']
    config = json.loads(config_path.read_text(encoding='utf-8'))
    kind = config['model']
    builders = {'M0': M0, 'M1': M1, 'M2': M2}
    if kind in builders:
        model = builders[kind](SSLResNet50(), SSLResNet50(), config)
    elif kind == 'M1_1':
        if saved.get('stage') != 'gate':
            raise ValueError(f'{path}: M1-1 requires the stage-B best.pt')
        teacher = json.loads((PROJECT_ROOT / 'models/M1/config.json').read_text(encoding='utf-8'))
        model = M1_1(M1(SSLResNet50(), SSLResNet50(), teacher), config)
    else:
        module = '.'.join(config_path.parent.relative_to(PROJECT_ROOT).parts) + '.model'
        model = importlib.import_module(module).build_model(config)
    model.load_state_dict(saved['model'], strict=True)
    activation = config.get('cell_head', {}).get('activation', 'sigmoid' if kind == 'M2' else 'softmax')
    if activation not in ('softmax', 'sigmoid'):
        raise ValueError(f'{path}: unsupported cell activation {activation}')
    target = settings.get('cell_target', {})
    target_mode = target.get('mode') if isinstance(target, dict) else None
    return model.eval(), int(settings['size']), activation, kind, target_mode, settings.get('amp', True)


def make_split(dataset, seed):
    groups = sorted({row['slide_name'] for row in dataset.source.rows})
    if len(groups) < 2:
        raise ValueError('Validation needs at least two different WSIs')
    random.Random(seed).shuffle(groups)
    tuning_groups = set(groups[:max(1, round(len(groups) * .25))])
    fit, tune = [], []
    for index, row in enumerate(dataset.source.rows):
        (tune if row['slide_name'] in tuning_groups else fit).append(index)
    return fit, tune


def fingerprint(paths, extra):
    identity = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(repr((identity, extra)).encode()).hexdigest()[:20]


def save_array(path, array):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, array, allow_pickle=False)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@torch.inference_mode()
def cache_maps(args, dataset, indices, paths, cache, device):
    from common.data.tensors import collate_pairs
    from common.paths import resolve_paths
    from common.training.m0_loop import autocast, prepare

    paths_config = resolve_paths()
    source_files = [paths_config['outputs'] / 'audit/manifest.csv',
                    dataset.source.root / 'metadata.json', Path(__file__),
                    args.repo / 'configs/components.json']
    for directory in ('models', 'common/backbones', 'common/modules', 'common/data'):
        source_files.extend(sorted((args.repo / directory).rglob('*.py')))
    source_files.extend(sorted((args.repo / 'models').rglob('*.json')))
    folders, activations = {}, {}
    for name, checkpoint in paths.items():
        key = fingerprint([checkpoint, *source_files], (name, dataset.source.split, str(device)))
        folder = cache / dataset.source.split / key
        folder.mkdir(parents=True, exist_ok=True)
        folders[name] = folder
        pending = [i for i in indices if not (folder / (dataset.source.rows[i]['pair_id'] + '.npy')).is_file()]
        descriptor = folder / 'activation.txt'
        if not pending and descriptor.is_file():
            activations[name] = descriptor.read_text(encoding='utf-8').strip()
            print(f'[cache] {name}: {len(indices)} pairs reused', flush=True)
            continue
        model, size, activation, kind, target_mode, use_amp = load_base(checkpoint)
        expected_target = {'G': 'gaussian', 'GA': 'ha'}.get(name)
        if expected_target and (kind != 'M2' or target_mode != expected_target):
            raise ValueError(f'{name} requires M2 {expected_target}; got {kind}/{target_mode}: {checkpoint}')
        expected_model = {'M0': 'M0', 'M1': 'M1', 'M1-1': 'M1_1'}.get(name)
        if expected_model and kind != expected_model:
            raise ValueError(f'{name} requires {expected_model}; got {kind}: {checkpoint}')
        model.to(device)
        activations[name] = activation
        descriptor.write_text(activation, encoding='utf-8')
        loader = DataLoader(Subset(dataset, pending), batch_size=1, num_workers=args.workers,
                            collate_fn=collate_pairs)
        print(f'[inference] {name}: {len(pending)} new pairs from {checkpoint}', flush=True)
        for number, batch in enumerate(loader, 1):
            images, _, box = prepare(batch, device, size)
            cast, _ = autocast(device, use_amp)
            with cast:
                logits = model(*images, box)['cell_logits'].float()
            expected = 3 if activation == 'softmax' else 2
            if logits.ndim != 4 or logits.shape[1] != expected:
                raise ValueError(f'{name}: expected BCHW with {expected} channels, got {logits.shape}')
            if activation == 'softmax':
                probability = logits.softmax(dim=1)[:, 1:]
            else:
                probability = logits.sigmoid()
            if probability.shape[-2:] != (1024, 1024):
                probability = F.interpolate(probability, (1024, 1024), mode='bilinear', align_corners=False)
            array = probability[0].cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f'{name}: nonfinite values for {batch["pair_id"][0]}')
            save_array(folder / (batch['pair_id'][0] + '.npy'), array)
            if number % 25 == 0 or number == len(pending):
                print(f'  {number}/{len(pending)}', flush=True)
        del model
    return folders, activations


class FusionCNN(nn.Module):
    def __init__(self, count):
        super().__init__()
        layers = []
        channels = count * 2
        for dilation in (1, 2, 4, 8):
            layers.extend((nn.Conv2d(channels, 16, 3, padding=dilation, dilation=dilation), nn.ReLU()))
            channels = 16
        output = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.correction = nn.Sequential(*layers, output)

    def forward(self, maps):
        mean = maps.reshape(maps.shape[0], -1, 2, *maps.shape[-2:]).mean(dim=1)
        return torch.logit(mean.clamp(1e-5, 1 - 1e-5)) + self.correction(maps)


class CachedPairs:
    def __init__(self, dataset, indices, folders):
        from common.data.targets import read_points
        self.names = tuple(folders)
        self.records = []
        for i in indices:
            row = dataset.source.rows[i]
            paths = [folder / (row['pair_id'] + '.npy') for folder in folders.values()]
            points = np.asarray(read_points(dataset.source.root / row['cell_points'])).reshape(-1, 3)
            self.records.append((row, paths, points))

    def maps(self, index, slices):
        y, x = slices
        return np.concatenate([np.load(p, mmap_mode='r', allow_pickle=False)[:, y, x]
                               for p in self.records[index][1]], axis=0)

    def __len__(self):
        return len(self.records)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.part')
    torch.save(value, temporary)
    os.replace(temporary, path)


def fit(args, pairs, device, checkpoint):
    from common.data.center_heatmaps import points_to_heatmaps

    rng = np.random.default_rng(args.seed)
    model = FusionCNN(len(pairs.names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    start_epoch = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
        start_epoch = saved['epoch']
        print(f'[resume] Fusion epoch {start_epoch}/{args.epochs}', flush=True)
    if start_epoch == args.epochs:
        return model.eval()
    # Compute each label map once. The 15 px halo preserves the CNN receptive field at crop edges.
    targets = [points_to_heatmaps(points, 6., 0., mode='gaussian').astype(np.float16)
               for _, _, points in pairs.records]
    size, halo = args.crop, 15
    for epoch in range(start_epoch, args.epochs):
        work = rng.permutation(np.repeat(np.arange(len(pairs)), args.crops_per_image))
        total_loss = 0.
        for start in range(0, len(work), args.batch_size):
            # Use equal tensor shapes at all image edges by padding the available halo.
            batch_maps, batch_targets = [], []
            for i in work[start:start + args.batch_size]:
                x, y = rng.integers(0, 1024 - size + 1, size=2)
                x0, y0 = max(0, x - halo), max(0, y - halo)
                x1, y1 = min(1024, x + size + halo), min(1024, y + size + halo)
                crop = pairs.maps(i, (slice(y0, y1), slice(x0, x1)))
                crop = np.pad(crop, ((0, 0), (y0 - y + halo, y + size + halo - y1),
                                   (x0 - x + halo, x + size + halo - x1)))
                batch_maps.append(crop)
                batch_targets.append(targets[i][:, y:y + size, x:x + size])
            maps = torch.from_numpy(np.stack(batch_maps)).to(device)
            truth = torch.from_numpy(np.stack(batch_targets)).to(device, dtype=torch.float32)
            output = model(maps)[:, :, halo:halo + size, halo:halo + size]
            weights = 1 + 9 * truth
            loss = (F.binary_cross_entropy_with_logits(output, truth, reduction='none') * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_maps)
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'rng': rng.bit_generator.state, 'epoch': epoch + 1}, checkpoint)
        print(f'[fusion] epoch {epoch + 1}/{args.epochs} weighted BCE={total_loss / len(work):.5f}', flush=True)
    return model.eval()


def candidate_points(maps, native, settings):
    from common.metrics.m0_eval import gaussian_blur

    score, labels = maps.max(dim=0)
    background = (1 - maps.sum(dim=0)).clamp(0, 1) if native else None
    confidence = 1 - background if native else score
    for sigma, by_sigma in itertools.groupby(settings, key=lambda p: p[0]):
        smooth = gaussian_blur(confidence[None], sigma)[0]
        for radius, by_radius in itertools.groupby(by_sigma, key=lambda p: p[1]):
            by_radius = list(by_radius)
            local_max = F.max_pool2d(smooth[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
            mask = (smooth == local_max) & (smooth > min(setting[2] for setting in by_radius))
            if native:
                mask &= background < score
            y, x = mask.nonzero(as_tuple=True)
            values = torch.stack((x, y, labels[y, x] + 1, score[y, x], smooth[y, x]), dim=1).cpu().numpy()
            for setting in by_radius:
                yield setting, values[values[:, 4] > setting[2], :4].tolist()


@torch.inference_mode()
def evaluate(model, pairs, activations, device, selected=None):
    from common.metrics.m0_eval import CellF1
    from scripts.diagnose_m0 import localization_labels, nearest_distances

    methods = [(f'base:{name}', activations[name] == 'softmax') for name in pairs.names]
    methods.extend((('mean', False), ('fusion_cnn', False)))
    trials = {}
    for name, native in methods:
        thresholds = (0., *GRID[2]) if native else GRID[2]
        settings = [selected[name]] if selected else itertools.product(*GRID[:2], thresholds)
        trials[name] = {params: CellF1() for params in settings}
    for i, (_, _, points) in enumerate(pairs.records):
        points = np.rint(points).astype(np.int32)
        x = torch.from_numpy(pairs.maps(i, (slice(None), slice(None)))).to(device)
        predictions = [*x.reshape(len(pairs.names), 2, 1024, 1024),
                       x.reshape(len(pairs.names), 2, 1024, 1024).mean(dim=0), model(x[None])[0].sigmoid()]
        for (name, native), maps in zip(methods, predictions):
            for setting, pred in candidate_points(maps, native, trials[name]):
                trials[name][setting].update(pred, points)
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f'[metrics] {i + 1}/{len(pairs)} pairs', flush=True)
    chosen, summaries = {}, {}
    for name, native in methods:
        options = trials[name]
        chosen[name] = max(options, key=lambda p: options[p].compute()['mF1'])
        metric = options[chosen[name]]
        summaries[name] = {**metric.compute(), 'FP_BC': metric.counts[1][1], 'FP_TC': metric.counts[2][1]}
    close = {name: [0, 0] for name, _ in methods}
    for i, (_, _, points) in enumerate(pairs.records):
        points = np.rint(points).astype(np.int32)
        distances = nearest_distances(points[:, :2])
        near_tc = (points[:, 2] == 2) & (distances >= 14) & (distances < 30)
        if not near_tc.any():
            continue
        x = torch.from_numpy(pairs.maps(i, (slice(None), slice(None)))).to(device)
        by_model = x.reshape(len(pairs.names), 2, 1024, 1024)
        maps_by_method = [*by_model, by_model.mean(dim=0), model(x[None])[0].sigmoid()]
        for (name, native), maps in zip(methods, maps_by_method):
            _, pred = next(candidate_points(maps, native, [chosen[name]]))
            labels, _ = localization_labels(pred, points)
            close[name][0] += int(((labels == 'missed') & near_tc).sum())
            close[name][1] += int(near_tc.sum())
    rows = []
    for name, _ in methods:
        sigma, min_distance, threshold = chosen[name]
        misses, count = close[name]
        rows.append({'method': name, **summaries[name], 'near_TC_missed': misses, 'near_TC_count': count,
                     'near_TC_miss': misses / count if count else None,
                     'sigma': sigma, 'min_distance': min_distance, 'threshold': threshold})
        print(f'  {name:<16} mF1={summaries[name]["mF1"]:.4f}  near TC missed={misses}/{count}', flush=True)
    return chosen, rows


def main():
    args = arguments()
    args.repo = args.repo.expanduser().resolve()
    if not (args.repo / 'models/M1_1/model.py').is_file():
        raise SystemExit('Run in the updated Medical AI repository or pass --repo /path/to/repo.')
    sys.path.insert(0, str(args.repo))
    from common.data.tensors import TorchOcelotDataset
    from common.paths import resolve_paths

    if not os.environ.get('OUTPUT_ROOT') and Path('/root/m2_overnight/outputs/audit/report.json').is_file():
        os.environ['OUTPUT_ROOT'] = '/root/m2_overnight/outputs'
    overnight_data = Path('/root/m2_overnight/assets/ocelot2023_v1.0.1')
    if not os.environ.get('OCELOT_DATA_ROOT') and not (resolve_paths()['data_root'] / 'metadata.json').is_file():
        if (overnight_data / 'metadata.json').is_file():
            os.environ['OCELOT_DATA_ROOT'] = str(overnight_data)
    from common.training.m0_loop import pick_device, seed_everything

    if args.smoke:
        args.epochs, args.crops_per_image = 1, 1
    project_paths = resolve_paths()
    paths = checkpoint_paths(args, project_paths['outputs'])
    seed_everything(args.seed)
    device = pick_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Activate the existing training environment or select --device cpu.')
    audit = project_paths['outputs'] / 'audit/report.json'
    if not audit.is_file():
        raise SystemExit(f'Missing {audit}. Set OCELOT_DATA_ROOT and OUTPUT_ROOT to the existing data and audit, '
                         'or run python scripts/check_data.py from the repository first.')
    output = args.output.expanduser().resolve()
    result = output / (args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    result.mkdir(parents=True, exist_ok=bool(args.run_id))
    print(f'[run] {result} | device={device} | models={", ".join(paths)}', flush=True)
    val = TorchOcelotDataset(split='val')
    fit_idx, tune_idx = make_split(val, args.seed)
    if args.smoke:
        fit_idx, tune_idx = fit_idx[:2], tune_idx[:2]
    config = {'model_order': list(paths), 'checkpoints': {k: str(v) for k, v in paths.items()},
              'fit_ids': [val.source.rows[i]['pair_id'] for i in fit_idx],
              'tune_ids': [val.source.rows[i]['pair_id'] for i in tune_idx],
              'seed': args.seed, 'epochs': args.epochs, 'crop': args.crop,
              'crops_per_image': args.crops_per_image, 'batch_size': args.batch_size,
              'smoke': args.smoke, 'device': str(device), 'test': args.test,
              'data_root': str(val.source.root),
              'checkpoint_identity': {k: fingerprint([v], '') for k, v in paths.items()}}
    config_path = result / 'run.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise SystemExit('Run settings changed. Use a new --run-id.')
    else:
        temporary = config_path.with_suffix('.part')
        temporary.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, config_path)
    done = result / 'complete.ok'
    fusion = result / 'fusion.pt'
    csv_path = result / 'summary.csv'
    if done.is_file() and fusion.is_file() and csv_path.is_file():
        print(f'[already complete] {csv_path}', flush=True)
        return
    cache = output / 'cache'
    if fusion.is_file():
        saved = torch.load(fusion, map_location='cpu', weights_only=False)
        model = FusionCNN(len(paths)).to(device)
        model.load_state_dict(saved['model'])
        model.eval()
        selected, rows = saved['postprocess'], saved['val_rows']
        print('[resume] Reusing the trained CNN and validation thresholds', flush=True)
    else:
        folders, activations = cache_maps(args, val, sorted(fit_idx + tune_idx), paths, cache, device)
        model = fit(args, CachedPairs(val, fit_idx, folders), device, result / 'fusion_last.pt')
        print('[tune] Select postprocessing on held-out validation WSIs', flush=True)
        selected, rows = evaluate(model, CachedPairs(val, tune_idx, folders), activations, device)
        atomic_save({'model': model.state_dict(), 'config': config,
                     'postprocess': selected, 'val_rows': rows}, fusion)
    summary = [{'split': 'val_tune', **row} for row in rows]

    def write_summary():
        temporary = csv_path.with_suffix('.part')
        with temporary.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        os.replace(temporary, csv_path)

    write_summary()
    if args.test:
        test = TorchOcelotDataset(split='test')
        indices = list(range(len(test)))
        folders, activations = cache_maps(args, test, indices, paths, cache, device)
        print('[test] Evaluate without retuning', flush=True)
        _, rows = evaluate(model, CachedPairs(test, indices, folders), activations, device, selected)
        summary.extend({'split': 'test', **row} for row in rows)
        write_summary()
    done.touch()
    print(f'[done] {csv_path}', flush=True)


if __name__ == '__main__':
    main()
PYTHON_OCELOT_ENSEMBLE
if ! cmp -s "$STATE/ocelot_ensemble.py.part" "$STATE/ocelot_ensemble.py"; then
  mv "$STATE/ocelot_ensemble.py.part" "$STATE/ocelot_ensemble.py"
else
  rm "$STATE/ocelot_ensemble.py.part"
fi

"$PYTHON" -u "$STATE/ocelot_ensemble.py" --repo "$REPO" --output "$STATE/experiments" \
  --device cuda --run-id "$RUN_ID" --epochs 12 --test --models G GA M0 M1 M1-1 \
  --model "G=$G_PATH" --model "GA=$GA_PATH" --model "M0=$M0_PATH" \
  --model "M1=$M1_PATH" --model "M1-1=$M11_PATH"
publish
