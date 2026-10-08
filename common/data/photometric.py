"""병리 영상의 색·염색 증강. 라벨(점·마스크)과 위치는 바꾸지 않는다.

- RandStainNA (Shen et al., MICCAI 2022): 학습셋의 LAB 평균/표준편차 분포에서
  가상 염색 템플릿을 뽑아 Reinhard 방식으로 맞춘다 = Random Stain Normalization.
- Color Jitter: 밝기·대비·채도·색조를 약하게 바꾼다.
- HED Jitter (Tellez et al., MedIA 2019): H·E·DAB 염색 농도 성분을 각각 흔든다. 기본 꺼짐.
- Gaussian Blur: 초점 차이를 흉내낸다. 기본 꺼짐(세포 경계가 흐려질 수 있음).

같은 슬라이드에서 나온 cell/tissue 사진에는 같은 파라미터를 쓴다(설정으로 변경 가능).
모든 함수는 RGB uint8 → RGB uint8이고 numpy와 Pillow만 사용한다.
"""

import copy
import json

import numpy as np
from PIL import Image, ImageFilter

from common.paths import PROJECT_ROOT

# ---------------------------------------------------------------- 색 공간

_M_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]], dtype=np.float32)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ).astype(np.float32)
_WHITE = _M_RGB2XYZ.sum(1)


def rgb_to_lab(image):
    """sRGB uint8 → CIELAB float32 (L 0~100)."""
    rgb = image.astype(np.float32) / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (rgb @ _M_RGB2XYZ.T) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1).astype(np.float32)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    rgb = np.clip(xyz @ _M_XYZ2RGB.T, 0, 1)
    rgb = np.where(rgb > 0.0031308, 1.055 * rgb ** (1 / 2.4) - 0.055, 12.92 * rgb)
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


def tissue_weight(image, low=205.0, high=235.0):
    """조직=1, 빈 유리(밝고 무채색)=0, 그 사이는 부드럽게. 염색 변환을 조직에만 적용한다."""
    rgb = image.astype(np.float32)
    gray = rgb.mean(-1)
    chroma = rgb.max(-1) - rgb.min(-1)
    w = np.clip((high - gray) / (high - low), 0, 1)
    # 밝더라도 색이 뚜렷하면(연한 분홍 조직) 조직으로 본다.
    return np.maximum(w, np.clip((chroma - 15) / 20, 0, 1))


def lab_stats(image, lab=None):
    """조직 픽셀만으로 LAB 채널 평균/표준편차를 구한다(빈 유리 제외)."""
    lab = rgb_to_lab(image) if lab is None else lab
    w = tissue_weight(image) > 0.5
    pix = lab[w] if w.mean() > 0.05 else lab.reshape(-1, 3)
    return pix.mean(0), pix.std(0)


# ---------------------------------------------------------- RandStainNA

def reinhard(image, target_mean, target_std):
    """조직 픽셀 통계로 Reinhard 변환을 하고, 빈 유리는 원래 색을 유지한다.

    빈 유리까지 통계에 넣으면 유리 비율에 따라 결과가 크게 달라지고
    흰 배경이 회색·청록으로 물드는 문제가 생겨 조직만 사용한다.
    """
    lab = rgb_to_lab(image)
    mean, std = lab_stats(image, lab)
    std = np.maximum(std, 1e-3)
    new = (lab - mean) / std * np.asarray(target_std, np.float32) + np.asarray(
        target_mean, np.float32)
    out = lab_to_rgb(new).astype(np.float32)
    w = tissue_weight(image)[..., None]
    return np.clip(np.rint(w * out + (1 - w) * image), 0, 255).astype(np.uint8)


def load_stain_stats(path):
    path = PROJECT_ROOT / path
    if not path.is_file():
        raise ValueError(
            f'RandStainNA 통계 파일이 없습니다: {path}\n'
            '먼저 python scripts/fit_randstainna.py 를 실행하세요 (train만 사용).'
        )
    stats = json.loads(path.read_text(encoding='utf-8'))
    if stats.get('color_space') != 'lab' or stats.get('split') != 'train':
        raise ValueError('RandStainNA 통계는 train으로 계산한 LAB 값이어야 합니다.')
    return stats


def sample_template(rng, stats, std_scale=1.0):
    """채널별 독립 정규분포(대각 공분산)에서 가상 템플릿을 뽑는다.

    극단값으로 비현실적인 색이 나오지 않게 평균에서 ±2σ(축소 후) 안으로 자른다.
    """
    out = []
    for key in ('mean', 'std'):
        mu = np.asarray(stats[key]['mu'], float)
        sd = np.asarray(stats[key]['sigma'], float) * std_scale
        out.append(np.clip(rng.normal(mu, sd), mu - 2 * sd, mu + 2 * sd))
    mean, std = out
    # 표준편차는 양수여야 한다. 평균값의 20% 아래로 내려가지 않게 자른다.
    std = np.maximum(std, 0.2 * np.asarray(stats['std']['mu']))
    return mean.astype(float).tolist(), std.astype(float).tolist()


# ---------------------------------------------------------- Color Jitter

def _gray(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def color_jitter(image, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0):
    """각 값은 배율(1=변화 없음), hue는 회전 비율(-0.5~0.5)."""
    rgb = image.astype(np.float32) / 255.0
    rgb = rgb * brightness
    rgb = (rgb - _gray(rgb).mean()) * contrast + _gray(rgb).mean()
    gray = _gray(rgb)[..., None]
    rgb = (rgb - gray) * saturation + gray
    if hue:
        # YIQ 평면에서 색조각을 회전한다(HSV 변환보다 빠른 근사).
        yiq = rgb @ np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.322],
                              [0.211, -0.523, 0.312]], np.float32).T
        t = 2 * np.pi * hue
        c, s = np.cos(t), np.sin(t)
        i, q = yiq[..., 1].copy(), yiq[..., 2].copy()
        yiq[..., 1], yiq[..., 2] = c * i - s * q, s * i + c * q
        rgb = yiq @ np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.106, 1.703]], np.float32).T
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


# ------------------------------------------------------------ HED Jitter

_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]], dtype=np.float32)
_RGB_FROM_HED /= np.linalg.norm(_RGB_FROM_HED, axis=1, keepdims=True)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def hed_jitter(image, alpha, beta):
    """Ruifrok 색 분해 후 성분별 x*alpha+beta. alpha/beta는 길이 3."""
    rgb = np.maximum(image.astype(np.float32), 1) / 255.0
    od = -np.log(rgb)
    hed = od @ _HED_FROM_RGB
    hed = hed * np.asarray(alpha, np.float32) + np.asarray(beta, np.float32)
    rgb = np.exp(-(hed @ _RGB_FROM_HED))
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


def gaussian_blur(image, sigma):
    return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(sigma)))


# ------------------------------------------------------------ 설정/적용

def load_augmentation(preset='default'):
    """configs/augmentation.json의 기본값에 preset 변경분을 덮어쓴다.

    preset: 이름(str) 또는 같은 구조의 dict(직접 지정).
    """
    config = json.loads(
        (PROJECT_ROOT / 'configs/augmentation.json').read_text(encoding='utf-8'))
    presets = config.pop('presets', {})
    if isinstance(preset, str):
        if preset not in presets:
            raise ValueError(f'증강 preset {preset!r} 없음. 가능: {sorted(presets)}')
        override, name = presets[preset], preset
    else:
        override, name = preset, 'custom'

    merged = copy.deepcopy(config)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    merged['preset'] = name
    return merged


def sample_photometric(rng, config, stats=None):
    """한 샘플에 쓸 색 파라미터를 정한다. 기록용으로 그대로 반환된다."""
    params = {}
    for name in config['order']:
        cfg = config[name]
        # 꺼진 증강도 난수를 같은 개수 소모해, 한 증강을 끄는 ablation에서
        # 나머지 증강의 파라미터가 바뀌지 않게 한다(공정한 비교).
        draw = rng.random()
        if name == 'randstainna':
            local = np.random.default_rng(rng.integers(2**32))
            value = sample_template(local, stats, cfg['std_scale']) if stats else None
        elif name == 'color_jitter':
            u = rng.uniform(-1, 1, 4)
            value = dict(brightness=float(1 + cfg['brightness'] * u[0]),
                         contrast=float(1 + cfg['contrast'] * u[1]),
                         saturation=float(1 + cfg['saturation'] * u[2]),
                         hue=float(cfg['hue'] * u[3]))
        elif name == 'hed_jitter':
            u = rng.uniform(-1, 1, 6)
            value = dict(alpha=(1 + cfg['sigma'] * u[:3]).tolist(),
                         beta=(cfg['sigma'] * u[3:]).tolist())
        elif name == 'gaussian_blur':
            value = dict(sigma=float(rng.uniform(0.1, cfg['sigma_max'])))
        else:
            raise ValueError(f'알 수 없는 증강: {name}')

        if cfg['enabled'] and draw < cfg['p']:
            if name == 'randstainna':
                if value is None:
                    raise ValueError('RandStainNA가 켜져 있는데 통계가 없습니다.')
                value = dict(mean=value[0], std=value[1])
            params[name] = value
    return params


def apply_photometric(image, params, order):
    for name in order:
        if name not in params:
            continue
        p = params[name]
        if name == 'randstainna':
            image = reinhard(image, p['mean'], p['std'])
        elif name == 'color_jitter':
            image = color_jitter(image, **p)
        elif name == 'hed_jitter':
            image = hed_jitter(image, p['alpha'], p['beta'])
        elif name == 'gaussian_blur':
            image = gaussian_blur(image, p['sigma'])
    return image
