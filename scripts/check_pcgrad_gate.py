"""Numerical checks; does not use the dataset, SSL weights or GPU."""
import copy
import io
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.training.conditional_pcgrad import (
    GROUPS, PCGradGates, assign_mixed, clip_factor, meta_update,
    virtual_adamw_parameters, evaluating,
)
from common.training.dual_gate import task_gradients


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Module()
        self.cell.encoder = nn.Linear(2,2,bias=False)
        self.tissue = nn.Module()
        self.tissue.encoder = nn.Linear(2,2,bias=False)
        self.tissue.aspp = nn.Linear(2,2,bias=False)
        self.tissue.decoder = nn.Linear(2,2,bias=False)
        self.encoder_c2t = nn.Linear(2,2,bias=False)
        self.private = nn.Parameter(torch.tensor([.2]))
        self.register_buffer("counter",torch.tensor(0.))

    def forward(self,x):
        if self.training:
            self.counter.add_(1)
        z = sum(layer(x) for layer in (
            self.cell.encoder,self.tissue.encoder,self.tissue.aspp,
            self.tissue.decoder,self.encoder_c2t))
        return {"cell":z+self.private,"tissue":z*.6}


def losses(out):
    return dict(cell=(out["cell"]-1.1).square().mean(),
                tissue=(out["tissue"]+.8).square().mean())


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            equal(x,y)
    else:
        assert a==b,(a,b)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = Tiny().double()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.03,foreach=False)
    raw = {}
    for name,p in model.named_parameters():
        c = torch.arange(1,p.numel()+1,dtype=p.dtype).reshape(p.shape)*.3
        t = -c+torch.full_like(c,.2)
        raw[name] = (c,t) if name!="private" else (c,None)
    gates = PCGradGates("conditional").double()
    observation = gates.observe(raw)
    zero = {k:torch.zeros(2,dtype=torch.double) for k in observation}
    one = {k:torch.ones(2,dtype=torch.double) for k in observation}
    m0, m1 = gates.mix(raw,observation,zero),gates.mix(raw,observation,one)
    for name,(c,t) in raw.items():
        torch.testing.assert_close(m0[name],c if t is None else c+t,atol=0,rtol=0)
    # Independently assemble each GROUP as a flat vector and apply PCGrad.
    for key,prefix in GROUPS.items():
        names = [n for n in raw if n.startswith(prefix)]
        c = torch.cat([raw[n][0].flatten() for n in names])
        t = torch.cat([raw[n][1].flatten() for n in names])
        dot = c@t
        expected = c+t-dot/t.square().sum()*t-dot/c.square().sum()*c
        actual = torch.cat([m1[n].flatten() for n in names])
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    half = gates.mix(raw,observation,{k:v+.5 for k,v in zero.items()})
    for n in half:
        torch.testing.assert_close(half[n],.5*(m0[n]+m1[n]))
    print("[PASS] zero/full/partial projection matches independent modulewise PCGrad; private gradients preserved")

    static = PCGradGates("static").double()
    second = copy.deepcopy(observation)
    for o in second.values():
        o["features"][:,1:] += 2
    for key in observation:
        torch.testing.assert_close(static.ratios(observation)[key],static.ratios(second)[key])
    with torch.no_grad():
        gates.coefficients[:,:,1] = .3
    assert any(not torch.equal(gates.ratios(observation)[k],gates.ratios(second)[k]) for k in observation)
    previous = gates.ema.clone()
    gates.ratios(observation)
    assert torch.equal(previous,gates.ema)
    gates.advance(observation)
    assert bool((gates.ema_count==1).all())
    print("[PASS] static gates ignore batch features; conditional gates respond; EMA advances only explicitly")

    # Compare cloned functional AdamW to a real step: fresh AND warmed state,
    # including a frozen parameter and global gradient clipping.
    for warm in (False,True):
        if warm:
            assign_mixed(model,m0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.cell.encoder.weight.requires_grad_(False)
        data = {n:v for n,v in raw.items() if n!="cell.encoder.weight"}
        obs = gates.observe(data)
        mixed = gates.mix(data,obs)
        before_model,before_opt = copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict())
        virtual = virtual_adamw_parameters(model,optimizer,mixed,.7)
        equal(before_model,model.state_dict())
        equal(before_opt,optimizer.state_dict())
        clone = copy.deepcopy(model)
        other_opt = torch.optim.AdamW(clone.parameters(),lr=.02,weight_decay=.03,foreach=False)
        other_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assign_mixed(clone,mixed)
        torch.nn.utils.clip_grad_norm_(clone.parameters(),.7,foreach=False)
        other_opt.step()
        for name,p in clone.named_parameters():
            torch.testing.assert_close(virtual[name].detach(),p,rtol=1e-10,atol=1e-12)
        model.cell.encoder.weight.requires_grad_(True)
    print("[PASS] virtual AdamW equals real fresh/warmed AdamW with decay, clipping and frozen parameters; states untouched")

    # Zero-valued gradient components must have finite derivatives through AdamW.
    z = torch.zeros(2,requires_grad=True,dtype=torch.double)
    from common.training.conditional_pcgrad import _safe_sqrt
    v = z.square()
    torch.autograd.grad((z/(_safe_sqrt(v)+1e-8)).sum(),z)
    assert torch.isfinite(torch.autograd.grad((z/(_safe_sqrt(z.square())+1e-8)).sum(),z)[0]).all()

    data = raw
    obs = gates.observe(data)
    def objective():
        virtual = virtual_adamw_parameters(model,optimizer,gates.mix(data,obs),.7)
        from torch.func import functional_call
        with evaluating(model):
            out = functional_call(model,(virtual,{n:b.clone() for n,b in model.named_buffers()}),
                                  (torch.tensor([[.8,-.2]],dtype=torch.double),),strict=True)
        ls = losses(out)
        return .5*(ls["cell"]/2+ls["tissue"]/3)
    modes = [m.training for m in model.modules()]
    before_model = copy.deepcopy(model.state_dict())
    analytic = torch.autograd.grad(objective(),gates.coefficients)[0]
    for index in ((0,0,0),(0,1,1),(1,0,2),(3,1,3)):
        original = float(gates.coefficients[index].detach())
        eps = 1e-4
        values = []
        for delta in (eps,-eps):
            with torch.no_grad():
                gates.coefficients[index] = original+delta
            values.append(float(objective().detach()))
        with torch.no_grad():
            gates.coefficients[index] = original
        numeric = (values[0]-values[1])/(2*eps)
        torch.testing.assert_close(analytic[index],analytic.new_tensor(numeric),atol=1e-8,rtol=2e-4)
    equal(before_model,model.state_dict())
    assert [m.training for m in model.modules()]==modes
    print("[PASS] AdamW gate hypergradients match finite differences; real model/buffers/modes unchanged")

    # Exact checkpoint round-trip preserves next meta update, moments and EMA.
    gate_opt = torch.optim.Adam(gates.parameters(),lr=.01)
    x = torch.tensor([[.8,-.2]],dtype=torch.double)
    meta_update(model,optimizer,gates,gate_opt,data,obs,(x,),losses,[2.,3.],.7)
    buffer = io.BytesIO()
    torch.save(dict(g=gates.state_dict(),o=gate_opt.state_dict()),buffer)
    buffer.seek(0)
    saved = torch.load(buffer,weights_only=False)
    restored = PCGradGates("conditional").double()
    restored.load_state_dict(saved["g"])
    restored_opt = torch.optim.Adam(restored.parameters(),lr=.01)
    restored_opt.load_state_dict(saved["o"])
    for g,opt in ((gates,gate_opt),(restored,restored_opt)):
        ob = g.observe(data)
        meta_update(model,optimizer,g,opt,data,ob,(x,),losses,[2.,3.],.7)
        g.advance(ob)
    equal(gates.state_dict(),restored.state_dict())
    equal(gate_opt.state_dict(),restored_opt.state_dict())
    print("[PASS] checkpoint restores the exact next gate update and EMA")

    # Nonconflicting and zero-norm groups remain ordinary sums.
    positive = {n:(c,None if t is None else c.clone()) for n,(c,t) in raw.items()}
    for n,g in gates.mix(positive,gates.observe(positive)).items():
        c,t = positive[n]
        torch.testing.assert_close(g,c if t is None else c+t)
    zero_raw = {n:(torch.zeros_like(c),t) for n,(c,t) in raw.items()}
    assert all(not o["valid"] for o in gates.observe(zero_raw).values())
    print("[PASS] nonconflicting and zero-norm fallback; synthetic checks complete")


if __name__=="__main__":
    run()
