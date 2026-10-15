"""Bounded real CUDA witness before reusing an existing server environment."""
import json
from pathlib import Path
import sys
import time
import torch
from formal_protocol import ROOT,load_protocol,verify_baseline,initialize_frozen_baseline,dump,frozen
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss,DetectionLossConfig
from hwnas_fpga.models.sonar_supernet import build_sonar_supernet


def main():
    torch.set_num_threads(2);frozen.seed_all(42)
    p,b=load_protocol(ROOT/'configs/supernet_sonar_v0_1_formal.yaml');protection=verify_baseline(b)
    model,_=frozen.make_model(b)
    manifest=json.loads((ROOT/b['baseline_freeze']['manifest']).read_text())
    if frozen.state_hash(model)!=manifest['initial_state_sha256']:
        probe=ROOT/'reports/phase2c_supernet/local_seed42_initialization_probe.pt'
        differences={}
        if probe.exists():
            reference=torch.load(probe,map_location='cpu',weights_only=True)['model_state_dict']
            for k,v in model.state_dict().items():
                if not torch.equal(v,reference[k]):differences[k]=float((v-reference[k]).abs().max())
        dump(ROOT/'reports/phase2c_supernet/server_initialization_mismatch.json',dict(status='BLOCKED',expected=manifest['initial_state_sha256'],
            actual=frozen.state_hash(model),torch=torch.__version__,different_tensors=differences))
        initialize_frozen_baseline(model,p,b)
    model.cuda().train();criterion=DenseAnchorDetectionLoss(DetectionLossConfig.from_config(b)).cuda()
    train=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'train',42,0,train=True)
    images,targets=next(iter(train));start=time.perf_counter()
    out=model(images.cuda());frozen.finite_outputs(out);loss,parts=criterion(out,targets);loss.backward();frozen.gradient_summary(model)
    torch.cuda.synchronize();batch_seconds=time.perf_counter()-start
    valid=frozen.make_loader(b,ROOT/b['dataset']['data_dir'],'valid',42,0)
    metrics,_=frozen.evaluate(model,valid,criterion,b,'cuda',max_batches=1)
    supernet=build_sonar_supernet().cuda().train()
    active=dict(output_combination='P2_P3_P4_P5',kernels={f'stage{s}':(5 if s in (2,5) else 3) for s in (1,2,4,5,6,7)})
    supernet.set_active_subnet(active)
    super_loss,_=DenseAnchorDetectionLoss(supernet.loss_config(b)).cuda()(supernet(images.cuda()),targets)
    super_loss.backward()
    if not torch.isfinite(super_loss) or any(not torch.isfinite(p.grad).all() for p in supernet.parameters() if p.grad is not None):raise ValueError('Supernet CUDA witness failed')
    supernet.eval();exported=supernet.export_active_subnet().eval()
    with torch.no_grad():
        a=supernet(images[:1].cuda())['predictions'];c=exported(images[:1].cuda())['predictions']
        max_error=max(float((u-v).abs().max()) for u,v in zip(a,c))
    if max_error!=0:raise ValueError(f'Supernet CUDA export mismatch: {max_error}')
    dump(ROOT/'reports/phase2c_supernet/server_kernel_witness.json',dict(status='PASS',python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),loss=parts,validation_images=metrics['num_images'],batch_seconds=batch_seconds,
        baseline_seeded_initial_state_equal=True,supernet_cuda_train_pass=True,supernet_cuda_export_max_abs_error=max_error,protection=protection,test_evaluated=False))
    print('SERVER_KERNEL_WITNESS_PASS',batch_seconds,flush=True)


if __name__=='__main__':main()
