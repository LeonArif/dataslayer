# Image augmentations for ft2_train.py, one token per augmentation so each can be switched on alone (ablation) or in groups.
# usage: env AUG="zoom,shift" or a group name (geo, photo, quality, hide, mix, weather) or a mix like "geo+hide"; AUG=0 means none.
# All ops work on a float batch (B, 3, H, W) in [0, 1] on the GPU, each sample draws its own random parameters.
# Not implemented on purpose (unrealistic for a fixed exam camera, or only for object detection): vflip, rot90, invert/solarize, channel shuffle, mosaic, copy-paste.
import math
import cv2, numpy as np, torch, torch.nn.functional as F

GROUPS = {
    "geo": "zoom,shift,rot,shear,persp,elastic",
    "photo": "bright,contrast,sat,hue,gamma,gray,autocontrast,posterize,equalize",
    "quality": "blur,motion,noise,sp,jpeg,downup,sharpen,patch",
    "hide": "erase,cutout,griddrop,coarse",
    "mix": "mixup,cutmix",
    "weather": "fog,shadow,flare,rain,snow",
    "legacy1": "bright,contrast,sat,zoom,shift,rot,blur",
}
ALL = set(",".join(GROUPS.values()).split(",")) | {"crop", "hflip"}


def parse(spec):
    ops = []
    for t in spec.replace("+", ",").split(","):
        t = t.strip()
        if t in ("", "0", "none"):
            continue
        if t in GROUPS:
            ops += GROUPS[t].split(",")
        else:
            assert t in ALL, f"unknown augmentation {t}"
            ops.append(t)
    return list(dict.fromkeys(ops))


def _u(x, lo, hi):
    return lo + (hi - lo) * torch.rand(x.shape[0], 1, 1, 1, device=x.device)


def _p(x, p):
    return (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p).float()


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _conv(x, k):
    k = k.to(x.device, x.dtype); r = k.shape[-1] // 2
    return F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), k.expand(3, 1, *k.shape[-2:]).contiguous(), groups=3)


def _gauss_k(sigma, size=5):
    a = torch.exp(-(torch.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)); a = a / a.sum()
    return (a[:, None] * a[None, :])[None, None]


def _line_k(angle, length, size=9):
    k = torch.zeros(size, size); c = size // 2
    for t in np.linspace(-length / 2, length / 2, 6 * size):
        i, j = int(round(c + t * math.sin(angle))), int(round(c + t * math.cos(angle)))
        if 0 <= i < size and 0 <= j < size:
            k[i, j] = 1
    return (k / k.sum())[None, None]


def _blob(x, sig_lo, sig_hi):
    B, _, H, W = x.shape
    X = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W); Y = (torch.linspace(0, 1, H, device=x.device) * H / W).view(1, 1, H, 1)
    cx, cy = _u(x, 0, 1), _u(x, 0, H / W)
    s = _u(x, sig_lo, sig_hi)
    return torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2))


def _boxes(x, n, hlo, hhi, wlo, whi):
    B, _, H, W = x.shape
    m = torch.zeros(B, 1, H, W, dtype=torch.bool, device=x.device)
    Y = torch.arange(H, device=x.device).view(1, 1, H, 1); X = torch.arange(W, device=x.device).view(1, 1, 1, W)
    for _ in range(n):
        h, w = _u(x, hlo, hhi) * H, _u(x, wlo, whi) * W
        y0, x0 = torch.rand(B, 1, 1, 1, device=x.device) * (H - h), torch.rand(B, 1, 1, 1, device=x.device) * (W - w)
        m |= (Y >= y0) & (Y < y0 + h) & (X >= x0) & (X < x0 + w)
    return m


def geometry(x, ops):
    if not any(o in ops for o in ("zoom", "crop", "shift", "rot", "shear", "persp", "elastic", "hflip")):
        return x
    B, _, H, W = x.shape; dev = x.device
    if "hflip" in ops:
        x = torch.where(torch.rand(B, 1, 1, 1, device=dev) < 0.5, x.flip(3), x)
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(B, device=dev)
    sx = sy = torch.ones(B, device=dev); tx = ty = torch.zeros(B, device=dev)
    if "zoom" in ops:
        sx = sy = r(0.85, 1.1)
    if "crop" in ops:
        s, ar = r(0.65, 1.0), r(0.9, 1.1).sqrt(); sx, sy = (s * ar).clamp(max=1), (s / ar).clamp(max=1)
        tx, ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - sx), (torch.rand(B, device=dev) * 2 - 1) * (1 - sy)
    if "shift" in ops:
        tx, ty = tx + r(-0.05, 0.05), ty + r(-0.05, 0.05)
    a = r(-1, 1) * math.radians(4) if "rot" in ops else torch.zeros(B, device=dev)
    k = r(-1, 1) * math.tan(math.radians(5)) if "shear" in ops else torch.zeros(B, device=dev)
    theta = torch.zeros(B, 2, 3, device=dev)
    R = torch.stack([torch.stack([a.cos(), -a.sin() * H / W], 1), torch.stack([a.sin() * W / H, a.cos()], 1)], 1)
    Sh = torch.stack([torch.stack([torch.ones(B, device=dev), k * H / W], 1), torch.stack([torch.zeros(B, device=dev), torch.ones(B, device=dev)], 1)], 1)
    S = torch.diag_embed(torch.stack([sx, sy], 1))
    theta[:, :, :2] = S @ R @ Sh; theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    if "persp" in ops:
        gx, gy = grid[..., 0], grid[..., 1]
        a1, b1, g, h = [r(-0.05, 0.05).view(B, 1, 1) for _ in range(2)] + [r(-0.12, 0.12).view(B, 1, 1) for _ in range(2)]
        den = 1 + g * gx + h * gy
        grid = torch.stack([(gx + a1 * gy) / den, (gy + b1 * gx) / den], -1)
    if "elastic" in ops:
        noise = torch.randn(B, 2, 5, 9, device=dev) * 0.03
        grid = grid + F.interpolate(noise, size=(H, W), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def photometric(x, ops):
    if "bright" in ops: x = x * _u(x, 0.75, 1.25)
    if "contrast" in ops:
        m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * _u(x, 0.75, 1.25) + m
    if "sat" in ops:
        g = _gray(x); x = (x - g) * _u(x, 0.7, 1.3) + g
    if "hue" in ops:
        th = _u(x, -0.17, 0.17); c, s = th.cos(), th.sin(); r_, g_, b_ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        Y = 0.299 * r_ + 0.587 * g_ + 0.114 * b_; I = 0.596 * r_ - 0.274 * g_ - 0.322 * b_; Q = 0.211 * r_ - 0.523 * g_ + 0.312 * b_
        I2, Q2 = I * c - Q * s, I * s + Q * c
        x = torch.cat([Y + 0.956 * I2 + 0.621 * Q2, Y - 0.272 * I2 - 0.647 * Q2, Y - 1.106 * I2 + 1.703 * Q2], 1)
    x = x.clamp(0, 1)
    if "gamma" in ops: x = x.clamp(1e-4, 1) ** _u(x, 0.8, 1.25)
    if "gray" in ops:
        m = _p(x, 0.2); x = m * _gray(x).expand_as(x) + (1 - m) * x
    if "autocontrast" in ops:
        f = x.flatten(1); lo, hi = torch.quantile(f, 0.01, 1).view(-1, 1, 1, 1), torch.quantile(f, 0.99, 1).view(-1, 1, 1, 1)
        m = _p(x, 0.3); x = m * ((x - lo) / (hi - lo + 1e-4)).clamp(0, 1) + (1 - m) * x
    if "posterize" in ops:
        L = (2 ** torch.randint(4, 7, (x.shape[0], 1, 1, 1), device=x.device)).float() - 1
        m = _p(x, 0.2); x = m * (x * L).round() / L + (1 - m) * x
    if "equalize" in ops:
        f = x.flatten(2); N = f.shape[2]
        eq = f.argsort(2).argsort(2).float().div(N - 1).view_as(x)
        m = _p(x, 0.2); x = m * eq + (1 - m) * x
    return x.clamp(0, 1)


def quality(x, ops):
    B, _, H, W = x.shape
    if "blur" in ops:
        m = _p(x, 0.5); x = m * _conv(x, _gauss_k(float(0.3 + 0.9 * torch.rand(1)))) + (1 - m) * x
    if "motion" in ops:
        m = _p(x, 0.3); x = m * _conv(x, _line_k(float(torch.rand(1)) * math.pi, 5 + 4 * float(torch.rand(1)))) + (1 - m) * x
    if "noise" in ops: x = x + torch.randn_like(x) * _u(x, 0.01, 0.05) * _p(x, 0.5)
    if "sp" in ops:
        rnd = torch.rand(B, 1, H, W, device=x.device); amt = _u(x, 0.005, 0.02) * _p(x, 0.3)
        x = torch.where(rnd < amt / 2, torch.zeros_like(x), torch.where(rnd > 1 - amt / 2, torch.ones_like(x), x))
    if "jpeg" in ops:
        arr = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy(); out = []
        for im in arr:
            if np.random.rand() < 0.5:
                im = cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(np.random.randint(30, 86))])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            out.append(im)
        x = torch.from_numpy(np.stack(out)).to(x.device).permute(0, 3, 1, 2).float() / 255
    if "downup" in ops:
        f = float(0.4 + 0.35 * torch.rand(1))
        d = F.interpolate(F.interpolate(x, scale_factor=f, mode="area"), size=(H, W), mode="bilinear", align_corners=False)
        m = _p(x, 0.5); x = m * d + (1 - m) * x
    if "sharpen" in ops:
        m = _p(x, 0.3); x = x + m * _u(x, 0.5, 1.5) * (x - _conv(x, _gauss_k(1.0)))
    if "patch" in ops:
        m = _p(x, 0.5); x = x * (1 + m * _u(x, -0.4, 0.4) * _blob(x, 0.1, 0.35))
    return x.clamp(0, 1)


def weather(x, ops):
    B, _, H, W = x.shape
    if "fog" in ops:
        a = _u(x, 0.1, 0.35) * _p(x, 0.3); x = x * (1 - a) + 0.85 * a
    if "shadow" in ops:
        th = _u(x, 0, 2 * math.pi); X = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W); Y = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        edge = torch.sigmoid(8 * (X * th.cos() + Y * th.sin() - _u(x, -0.4, 0.4)))
        x = x * (1 - _p(x, 0.3) * _u(x, 0.3, 0.6) * edge)
    if "flare" in ops:
        tint = torch.tensor([1.0, 0.9, 0.7], device=x.device).view(1, 3, 1, 1)
        x = x + _p(x, 0.3) * _u(x, 0.3, 0.8) * _blob(x, 0.05, 0.2) * tint
    if "rain" in ops:
        dots = (torch.rand(B, 1, H, W, device=x.device) < 0.002).float()
        x = x + _p(x, 0.3) * 0.5 * _conv(dots.expand(B, 3, H, W).contiguous(), _line_k(math.radians(100 + 10 * float(torch.rand(1))), 9) * 9).clamp(0, 1)
    if "snow" in ops:
        dots = (torch.rand(B, 1, H, W, device=x.device) < 0.003).float()
        x = x + _p(x, 0.3) * 0.8 * _conv(dots.expand(B, 3, H, W).contiguous(), _gauss_k(0.8, 3))
    return x.clamp(0, 1)


def hiding(x, ops):
    fill = torch.rand(x.shape[0], 3, 1, 1, device=x.device)
    if "erase" in ops:
        m = _boxes(x, 1, 0.15, 0.6, 0.1, 0.4) & (_p(x, 0.5) > 0); x = torch.where(m, fill.expand_as(x), x)
    if "cutout" in ops:
        m = _boxes(x, 2, 0.2, 0.3, 0.12, 0.17) & (_p(x, 0.5) > 0); x = torch.where(m, torch.full_like(x, 0.5), x)
    if "griddrop" in ops:
        B, _, H, W = x.shape; unit = int(np.random.randint(32, 57))
        cells = (torch.rand(B, 1, math.ceil(H / unit), math.ceil(W / unit), device=x.device) < 0.3).float()
        m = (F.interpolate(cells, size=(H, W), mode="nearest") > 0) & (_p(x, 0.5) > 0); x = torch.where(m, torch.full_like(x, 0.5), x)
    if "coarse" in ops:
        m = _boxes(x, 10, 0.05, 0.14, 0.03, 0.08) & (_p(x, 0.5) > 0); x = torch.where(m, torch.full_like(x, 0.5), x)
    return x


def mixing(x, y1h, ops):
    """Mixup / CutMix on the whole batch (half of the batches): returns mixed images and soft labels."""
    kinds = [o for o in ("mixup", "cutmix") if o in ops]
    if not kinds or np.random.rand() > 0.5:
        return x, None
    kind, B, perm = kinds[np.random.randint(len(kinds))], x.shape[0], torch.randperm(x.shape[0], device=x.device)
    lam = float(np.random.beta(0.4, 0.4))
    if kind == "mixup":
        return lam * x + (1 - lam) * x[perm], lam * y1h + (1 - lam) * y1h[perm]
    H, W = x.shape[2:]; ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W); y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x = x.clone(); x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, lam * y1h + (1 - lam) * y1h[perm]


def augment(x, y1h, ops):
    """x float (B,3,H,W) in [0,1], y1h one-hot (B,C). Returns the augmented images and soft labels (None when no label mixing happened)."""
    x = hiding(weather(quality(photometric(geometry(x, ops), ops), ops), ops), ops).clamp(0, 1)
    return mixing(x, y1h, ops)
