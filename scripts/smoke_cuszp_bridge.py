import torch
from boundrelay.codec.cuszp_bridge import roundtrip
x = torch.randn(8, 64, 128).to(torch.bfloat16)
eps = 0.10 * float(x.float().std())
for m in ("plain", "outlier", "fixed"):
    r = roundtrip(x, eps, m)
    print("  {:>8}: {:.2f}x  err_fp32={:.5f}  err_bf16={:.5f}  eps={:.5f}  within={}".format(
        m, r["ratio_vs_bf16"], r["max_error_fp32"], r["max_error_bf16"], eps, r["within_eps_fp32"]))
