"""Exercise the REAL trainer with tiny synthetic data; exact interrupted resume."""
import contextlib
import copy
import csv
import gc
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.train_m0_pcgrad_gate as trainer
import common.training.dual_gate_run as data_tools
from common.data.tensors import collate_pairs
from scripts.check_pcgrad_gate import equal


class Data(Dataset):
    def __init__(self,split="train",**kwargs):
        self.split,self.epoch = split,0
        count = 8 if split=="train" else 2
        rows = [dict(pair_id=f"{split}{i}",slide_name=f"CASE{i}") for i in range(count)]
        self.source = SimpleNamespace(rows=rows,metadata={r["pair_id"]:dict(organ="kidney") for r in rows})
    def set_epoch(self,epoch):
        self.epoch = epoch
    def __len__(self):
        return len(self.source.rows)
    def __getitem__(self,i):
        g = torch.Generator().manual_seed(100+i)
        return dict(cell_image=torch.randn(3,8,8,generator=g),
                    tissue_image=torch.randn(3,8,8,generator=g),
                    cell_target=torch.randint(0,3,(8,8),generator=g),
                    tissue_target=torch.randint(0,2,(8,8),generator=g),
                    cell_box=torch.tensor([.1,.1,.8,.8]),
                    cell_points=torch.zeros(0,3),pair_id=self.source.rows[i]["pair_id"])


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell,self.tissue = nn.Module(),nn.Module()
        self.cell.encoder = nn.Conv2d(3,4,1)
        self.tissue.encoder = nn.Conv2d(3,4,1)
        self.encoder_c2t = nn.Conv2d(4,4,1)
        self.tissue.aspp = nn.Sequential(nn.Conv2d(4,4,1),nn.Dropout(.2))
        self.tissue.decoder = nn.Conv2d(4,4,1)
        self.cell.head = nn.Conv2d(4,3,1)
        self.tissue.head = nn.Conv2d(4,3,1)
    def forward(self,c,t,boxes):
        c,t = self.cell.encoder(c),self.tissue.encoder(t)
        t = self.tissue.decoder(self.tissue.aspp(t+self.encoder_c2t(c)))
        return dict(cell_logits=self.cell.head(c+t),tissue_logits=self.tissue.head(t))
    def encoder_parameters(self):
        return [*self.cell.encoder.parameters(),*self.tissue.encoder.parameters()]
    def new_module_parameters(self):
        old = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in old]


def loader(split,seed,batch_size,num_workers,**kwargs):
    data = Data(split)
    return DataLoader(data,batch_size=batch_size,shuffle=kwargs.get("shuffle",False),
                      generator=torch.Generator().manual_seed(seed),collate_fn=collate_pairs),data


@torch.no_grad()
def evaluate(model,loader,device,*args,**kwargs):
    model.eval()
    total = dict(cell=0.,tissue=0.)
    for batch in loader:
        images,targets,boxes = trainer.prepare(batch,device,128)
        out = model(*images,boxes)
        loss = trainer.m0_task_losses(out["cell_logits"],out["tissue_logits"],*targets)
        for k in total:
            total[k] += float(loss[k])/len(loader)
    return dict(loss_cell=total["cell"],loss_tissue=total["tissue"],
                loss_total=sum(total.values()),mF1=.4,f1_BC=.3,f1_TC=.5,
                mIoU=.6,iou_BG=.5,iou_CA=.7),[],None


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="pcgrad-integration-") as td:
        root = Path(td)
        (root/"configs").mkdir()
        (root/"models/M0").mkdir(parents=True)
        for relative in ("configs/m0_train.json","models/M0/config.json"):
            (root/relative).write_bytes((ROOT/relative).read_bytes())
        trainer.ROOT = root
        trainer.resolve_paths = lambda:dict(outputs=root/"outputs",data_root=root/"data",ssl_weights=root/"weights")
        trainer.make_loader = loader
        trainer.build_model = lambda *a,**k:Mini()
        trainer.evaluate = evaluate
        data_tools.TorchOcelotDataset = Data
        def run(name,mode,resume=False,stop=False):
            sys.argv = ["trainer","--device","cpu","--no-amp","--gate-mode",mode,
                        "--num-workers","0","--size","128","--epochs","2",
                        "--freeze-epochs","1","--meta-count","2","--gate-every","1",
                        "--run-name",name]
            if resume:
                sys.argv += ["--resume",str(root/"outputs/m0"/name/"last.pt")]
            if stop:
                sys.argv += ["--stop-after-epoch","1"]
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
        run("full","conditional")
        run("resumed","conditional",stop=True)
        run("resumed","conditional",resume=True)
        run("static","static")
        def state(name):
            return torch.load(root/"outputs/m0"/name/"last.pt",map_location="cpu",weights_only=False)
        a,b,c = state("full"),state("resumed"),state("static")
        for key in ("model","optimizer","gates","gate_optimizer","rng","step","meta_scales"):
            if key=="rng":
                for name in ("python","torch","cuda"):
                    equal(a[key][name],b[key][name])
            else:
                equal(a[key],b[key])
        for relative in ("pcgrad_gate/batches.csv","pcgrad_gate/epochs.csv",
                         "pcgrad_gate/coefficients.csv","gradient_conflict/batches.csv",
                         "gradient_conflict/epochs.csv"):
            assert (root/"outputs/m0/full"/relative).read_bytes()==(root/"outputs/m0/resumed"/relative).read_bytes(),relative
        assert a["settings"]["initial_model_sha256"]==c["settings"]["initial_model_sha256"]
        assert a["settings"]["gate_split"]==c["settings"]["gate_split"]
        assert a["meta_scales"]==c["meta_scales"]
        assert a["epoch"]==b["epoch"]==c["epoch"]==1
        # Batch records include both frozen and unfrozen parameter scopes.
        with (root/"outputs/m0/full/pcgrad_gate/batches.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows)==2*3*5
        assert all(r["active"]=="0" for r in rows if r["epoch"]=="1" and r["group"]=="cell_encoder")
        assert all(r["active"]=="1" for r in rows if r["epoch"]=="2")
        print("[PASS] real trainer on synthetic data: both modes, matched initialization/split/scales")
        print("[PASS] uninterrupted vs resumed model, both optimizers, gates, EMA, batch/epoch journals are EXACTLY equal")
        print("[PASS] frozen/unfrozen scope and history cleanup; no research training launched")
    # Verify detached-suite sequencing and fail-stop semantics without launching
    # subprocess training. Numerical/trainer checks above remain real executions.
    import scripts.run_pcgrad_gate_suite as suite
    old = (suite.resolve_paths,suite.source_hashes,suite.verify_run,suite.subprocess.run)
    with tempfile.TemporaryDirectory(prefix="pcgrad-suite-test-") as td:
        output = Path(td)
        suite.resolve_paths = lambda:dict(outputs=output)
        suite.source_hashes = lambda:dict(test="only")
        (output/"pcgrad_gate_preflight_latest.json").write_text(json.dumps(dict(status="pass",source_sha256=dict(test="only"))))
        def verification(folder,mode,epochs):
            return dict(settings=dict(gate_mode=mode,meta_scales=[1.,2.],same=True),
                        split=dict(model_train_ids=["a"],gate_meta_ids=["b"]),
                        initial="same",scales=[1.,2.],sum_epoch_seconds=1.,
                        best=dict(epoch="1",val_mF1=".5",val_f1_BC=".4",val_f1_TC=".6",val_mIoU=".7"))
        suite.verify_run = verification
        calls,fail = [],[False]
        def fake_run(argv,**kwargs):
            if len(argv)>2 and argv[2]=="scripts/train_m0_pcgrad_gate.py":
                mode = argv[argv.index("--gate-mode")+1]
                calls.append(mode)
                if fail[0]:
                    raise suite.subprocess.CalledProcessError(1,argv)
            return suite.subprocess.CompletedProcess(argv,0)
        suite.subprocess.run = fake_run
        try:
            sys.argv = ["suite","--execute"]
            with contextlib.redirect_stdout(io.StringIO()):
                suite.main()
            assert calls==["static","conditional"]
            status_path = output/"logs/pcgrad_gate_suite_seed42.status.json"
            assert json.loads(status_path.read_text())["phase"]=="completed"
            calls.clear()
            fail[0] = True
            sys.argv = ["suite","--execute","--resume"]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    suite.main()
            except suite.subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("Failed training did not stop the suite")
            assert calls==["static"]
            assert json.loads(status_path.read_text())["phase"]=="failed"
        finally:
            suite.resolve_paths,suite.source_hashes,suite.verify_run,suite.subprocess.run = old
    print("[PASS] suite runs static THEN conditional; a failed first job stops the second and records failure")


if __name__=="__main__":
    main()
