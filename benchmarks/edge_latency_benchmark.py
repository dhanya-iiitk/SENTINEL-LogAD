import os, sys, time, torch, numpy as np
sys.path.insert(0, '/content/SENTINEL-LogAD')

from sentinel.models.sentinel_seq import SENTINELSeqBase, SENTINELSeqFeatureExtract

os.sched_setaffinity(0, {0})
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device('cpu')

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state']
    if any(k.startswith('base.') for k in state.keys()):
        vocab_size  = state['base.embedding.weight'].shape[0]
        task_vocab  = state['task_head.2.weight'].shape[0]
        base = SENTINELSeqBase(vocab_size=vocab_size)
        model = SENTINELSeqFeatureExtract(base, task_vocab_size=task_vocab)
    else:
        vocab_size = state['embedding.weight'].shape[0]
        model = SENTINELSeqBase(vocab_size=vocab_size)
    model.load_state_dict(state)
    model.eval()
    return model

def benchmark(model, seq_len=20, n_warmup=20, n_runs=200):
    seq     = torch.randint(1, 50, (1, seq_len))
    lengths = torch.tensor([seq_len])
    for _ in range(n_warmup):
        _ = model.anomaly_score(seq, lengths)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _  = model.anomaly_score(seq, lengths)
        latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return {
        'mean_ms' : round(arr.mean(), 2),
        'p50_ms'  : round(np.percentile(arr, 50), 2),
        'p95_ms'  : round(np.percentile(arr, 95), 2),
        'p99_ms'  : round(np.percentile(arr, 99), 2),
        'std_ms'  : round(arr.std(), 2),
    }

CKPTS = {
    'HDFS': '/content/drive/MyDrive/SENTINEL/results/checkpoints/sentinel_fe_HDFS_balanced_best.pt',
    'BGL' : '/content/drive/MyDrive/SENTINEL/results/checkpoints/sentinel_fe_BGL_best.pt',
    'TB'  : '/content/drive/MyDrive/SENTINEL/results/checkpoints/sentinel_fe_TB_best.pt',
}

if __name__ == '__main__':
    print("=" * 60)
    print("SENTINEL Edge-Simulated Latency Benchmark")
    print("Config: 1 CPU core | 1 thread | batch=1 | seq_len=20")
    print("=" * 60)
    results = {}
    for name, path in CKPTS.items():
        print(f"\nLoading {name} checkpoint...")
        model = load_model(path)
        params = model.count_parameters()
        print(f"  Trainable params: {params:,}")
        r = benchmark(model)
        results[name] = r
        print(f"  mean={r['mean_ms']}ms | p50={r['p50_ms']}ms | p95={r['p95_ms']}ms | p99={r['p99_ms']}ms | std={r['std_ms']}ms")
    print("\n" + "=" * 60)
    print(f"{'Dataset':<8} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("-" * 52)
    for name, r in results.items():
        print(f"{name:<8} {r['mean_ms']:>10} {r['p50_ms']:>10} {r['p95_ms']:>10} {r['p99_ms']:>10}")
    print("=" * 60)
    print("\nNote: Intel Xeon, single-core throttled. Not equivalent to physical ARM/RPi 4.")
