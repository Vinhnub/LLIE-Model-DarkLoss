import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss với cơ chế Region-Focus (Dark / Bright / Balance).

    Dựa trên ý tưởng FGLoss (IGMamba-EdgeLCA-LLIE) nhưng tối ưu hóa chuẩn cho bài toán LLIE:
    Do ảnh GT (normal-light) thường đã phơi sáng đầy đủ, luminance của GT không phản ánh đúng
    độ khó phục hồi. Trọng số được tính từ ảnh đầu vào thiếu sáng (LQ) - nơi các vùng tối thực
    sự cần hệ số khuếch đại lớn, chịu nhiều nhiễu hạt và mất chi tiết nhất.

    Quy trình tính toán bản đồ trọng số:
    1. Trích xuất Luminance Y từ ảnh tham chiếu (mặc định là LQ) theo chuẩn ITU-R BT.601:
       Y = 0.299*R + 0.587*G + 0.114*B (hoặc kênh I nếu là HVI).
    2. Khử nhiễu cục bộ: Sử dụng AvgPool2d với kernel smooth_kernel để làm mượt bản đồ ánh sáng,
       tránh để nhiễu cảm biến (sensor noise) trong vùng tối LQ làm méo mó trọng số.
    3. Chuẩn hóa Min-Max theo từng ảnh:
       Y_rel = (Y - min(Y)) / (max(Y) - min(Y) + eps)
       Vì ảnh LQ bị tối trên toàn bộ khung hình (Y tuyệt đối rất thấp), bước này là BẮT BUỘC
       để ánh xạ về độ sáng TƯƠNG ĐỐI của từng vùng trong cảnh [0.0, 1.0].
    4. Xác định độ ưu tiên s theo chế độ focus:
       - Focus vùng tối ('dark'):   s = (1.0 - Y_rel) ^ dark_power  (vùng tối nhất có s = 1.0)
       - Focus vùng sáng ('bright'): s = (Y_rel) ^ dark_power        (vùng sáng nhất có s = 1.0)
       - Cân bằng ('balance'):      không tính trọng số, tương đương L1Loss tiêu chuẩn.
    5. Thiết lập trọng số dạng residual:
       w = 1.0 + focus_strength * s
       (Đảm bảo mọi vùng đều được học với trọng số >= 1.0, vùng focus được phạt gấp bội).
    6. Chuẩn hóa trung bình (norm_weight):
       w = w / mean(w)
       (Giữ nguyên scale và độ lớn của loss như L1 tiêu chuẩn, không làm lệch learning rate).

    Args:
        loss_weight (float): Hệ số nhân loss weight. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        dark_focus (bool | str | None): Chế độ focus:
            - True hoặc 'dark': Tập trung ưu tiên phạt nặng ở vùng tối của LQ.
            - False hoặc 'bright': Tập trung ưu tiên ở vùng sáng của LQ.
            - None hoặc 'balance': Không phân vùng, giữ nguyên L1 gốc. Default: None.
        weight_source (str): Nguồn ảnh dùng để tính trọng số: 'lq' (khuyến nghị) hoặc 'gt'.
            Nếu chọn 'lq' mà không truyền lq vào forward, loss tự động fallback sang target (GT).
            Default: 'lq'.
        focus_strength (float): Hệ số alpha điều chỉnh mức độ ưu tiên vùng focus.
            Default: 1.0 (vùng focus được phạt tối đa gấp 2 lần vùng thường trước khi chuẩn hóa).
        weight_mode (str): 'residual' (w = 1 + alpha*s) hoặc 'direct' (w = s). Default: 'residual'.
        dark_power (float): Số mũ gamma điều chỉnh độ dốc trọng số. Default: 1.0.
        smooth_kernel (int): Kích thước kernel AvgPool2d để làm mượt luminance khử nhiễu LQ.
            Default: 7 (đặt <= 1 để tắt làm mượt).
        normalize_lum (bool): Chuẩn hóa Min-Max luminance theo từng ảnh. Default: True.
        norm_weight (bool): Chuẩn hóa mean(w) = 1.0 để bảo toàn scale loss. Default: True.
        is_hvi (bool): Đặt True nếu ảnh ở không gian màu HVI. Default: False.
        eps (float): Ngưỡng cực tiểu tránh chia cho 0. Default: 1e-6.
        focus_type (str | None): Alias cho dark_focus.
        use_dark_weight (bool | None): Alias tương thích ngược với code cũ của IGMamba.
    """

    need_lq = True

    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        dark_focus=None,
        weight_source='lq',
        focus_strength=1.0,
        weight_mode='residual',
        dark_power=1.0,
        smooth_kernel=7,
        normalize_lum=True,
        norm_weight=True,
        is_hvi=False,
        eps=1e-6,
        focus_type=None,
        use_dark_weight=None,
    ):
        super(L1Loss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        if weight_source not in ['lq', 'gt']:
            raise ValueError(f"weight_source must be 'lq' or 'gt', got '{weight_source}'")
        if weight_mode not in ['residual', 'direct']:
            raise ValueError(f"weight_mode must be 'residual' or 'direct', got '{weight_mode}'")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_source = weight_source
        self.focus_strength = focus_strength
        self.weight_mode = weight_mode
        self.dark_power = dark_power
        self.smooth_kernel = int(smooth_kernel) if smooth_kernel else 0
        self.normalize_lum = normalize_lum
        self.norm_weight = norm_weight
        self.is_hvi = is_hvi
        self.eps = eps

        chosen_focus = dark_focus
        if use_dark_weight is not None:
            chosen_focus = use_dark_weight
        elif focus_type is not None:
            chosen_focus = focus_type

        self.focus_mode = self._parse_focus_mode(chosen_focus)
        self.dark_focus = chosen_focus

    @staticmethod
    def _parse_focus_mode(val):
        """Chuẩn hóa giá trị chế độ focus về 'dark' | 'bright' | 'balance'."""
        if val is None:
            return 'balance'
        if isinstance(val, bool):
            return 'dark' if val else 'bright'
        if isinstance(val, (int, float)):
            return 'dark' if val > 0 else 'bright'
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ['dark', 'true', 'dark_focus']:
                return 'dark'
            elif v in ['bright', 'false', 'bright_focus']:
                return 'bright'
            elif v in ['balance', 'balanced', 'none', 'null', 'uniform', 'normal']:
                return 'balance'
            else:
                raise ValueError(
                    f"Unsupported focus mode: '{val}'. Supported choices are: "
                    f"['dark', 'bright', 'balance', None, True, False]"
                )
        return 'balance'

    def _luminance(self, img):
        channels = img.shape[1]
        if channels == 3:
            if self.is_hvi:
                return img[:, 2:3]
            return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        elif channels == 1:
            return img
        else:
            raise ValueError(f'Unsupported channel size for weight map: {channels}')

    @torch.no_grad()
    def compute_weight_map(self, ref):
        """Tính toán bản đồ trọng số từ ảnh tham chiếu (LQ hoặc GT)."""
        if self.focus_mode == 'balance':
            return None

        # 1. Luminance
        y = self._luminance(ref.float()).clamp(0.0, 1.0)

        # 2. Làm mượt bằng AvgPool để khử nhiễu cảm biến (sensor noise)
        if self.smooth_kernel > 1:
            k = self.smooth_kernel
            if k % 2 == 0:
                k += 1
            y = F.avg_pool2d(y, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)

        # 3. Chuẩn hóa Min-Max theo từng ảnh trong batch (xét độ sáng tương đối)
        if self.normalize_lum:
            b = y.shape[0]
            y_flat = y.view(b, -1)
            y_min = y_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            y_max = y_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            y = (y - y_min) / (y_max - y_min + self.eps)

        # 4. Tính toán độ lệch trọng số s
        if self.focus_mode == 'dark':
            s = 1.0 - y
        elif self.focus_mode == 'bright':
            s = y
        else:
            return None

        if self.dark_power != 1.0:
            s = s.clamp(min=0.0).pow(self.dark_power)

        # 5. Dạng trọng số
        if self.weight_mode == 'direct':
            w = s.clamp(min=self.eps, max=1.0)
        else:
            w = 1.0 + self.focus_strength * s

        # 6. Chuẩn hóa trung bình mean(w) = 1.0 theo từng ảnh
        if self.norm_weight:
            w = w / (w.mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        return w

    def forward(self, pred, target, weight=None, lq=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Ảnh đầu ra dự đoán.
            target (Tensor): of shape (N, C, H, W). Ảnh mục tiêu Ground Truth.
            weight (Tensor, optional): Trọng số ngoài (N, C, H, W) hoặc (N, 1, H, W).
            lq (Tensor, optional): Ảnh đầu vào thiếu sáng (N, C, H, W), dùng khi weight_source='lq'.
        """
        # 1. Tính toán trọng số vùng nếu focus_mode là 'dark' hoặc 'bright'
        if self.focus_mode != 'balance':
            ref = lq if (self.weight_source == 'lq' and lq is not None) else target
            focus_w = self.compute_weight_map(ref).to(pred.dtype)
            if weight is not None:
                weight = weight * focus_w
            else:
                weight = focus_w

        # 2. Nếu chế độ cân bằng (balance) và không có external weight:
        # Sử dụng hàm l1_loss tiêu chuẩn của BasicSR (tương thích 100% bản gốc)
        if weight is None:
            return self.loss_weight * l1_loss(
                pred, target, weight=None, reduction=self.reduction)

        # 3. Tính toán sai lệch L1 có trọng số
        diff = torch.abs(pred - target) * weight

        # 4. Reduction
        if self.reduction == 'none':
            loss = diff
        elif self.reduction == 'sum':
            loss = diff.sum()
        elif self.reduction == 'mean':
            loss = diff.mean()
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}')

        return self.loss_weight * loss


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# -----------------------------------------------------------------------------
# Region-Focused / Gradient Loss Aliases and Classes (from IGMamba-EdgeLCA-LLIE)
# -----------------------------------------------------------------------------
DarkFocusedL1Loss = L1Loss
FGL1Loss = L1Loss


class DarkFocusedGradientLoss(nn.Module):
    """
    Dark-Focused / Bright-Focused Spatial Gradient Difference Loss (FGLoss / RegionLSGDLoss).
    Trích xuất từ IGMamba-EdgeLCA-LLIE nhằm bảo toàn chi tiết và biên cạnh với cơ chế tập trung vùng tối/sáng.

    - dark_focus=True / 'dark': Phạt nặng sai lệch gradient ở vùng tối của LQ/GT.
    - dark_focus=False / 'bright': Phạt nặng sai lệch gradient ở vùng sáng của LQ/GT.
    - dark_focus=None / 'balance': Phạt đồng đều toàn ảnh.
    """

    need_lq = True

    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        dark_focus=True,
        weight_source='lq',
        focus_strength=1.0,
        dark_power=1.0,
        smooth_kernel=7,
        normalize_lum=True,
        norm_weight=True,
        eps=1e-6,
        is_hvi=False,
        use_dark_weight=None,
        focus_type=None,
    ):
        super(DarkFocusedGradientLoss, self).__init__()

        if reduction not in _reduction_modes:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}'
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_source = weight_source
        self.focus_strength = focus_strength
        self.dark_power = dark_power
        self.smooth_kernel = int(smooth_kernel) if smooth_kernel else 0
        self.normalize_lum = normalize_lum
        self.norm_weight = norm_weight
        self.eps = eps
        self.is_hvi = is_hvi

        chosen_focus = dark_focus
        if use_dark_weight is not None:
            chosen_focus = use_dark_weight
        elif focus_type is not None:
            chosen_focus = focus_type

        self.focus_mode = L1Loss._parse_focus_mode(chosen_focus)
        self.dark_focus = chosen_focus

    def compute_spatial_gradient(self, img):
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        return grad_x, grad_y

    def _luminance(self, img):
        channels = img.shape[1]
        if channels == 3:
            if self.is_hvi:
                return img[:, 2:3]
            return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        elif channels == 1:
            return img
        else:
            raise ValueError(f'Unsupported channel size for weight map: {channels}')

    @torch.no_grad()
    def compute_weight_map(self, ref):
        if self.focus_mode == 'balance':
            return None

        y = self._luminance(ref.float()).clamp(0.0, 1.0)

        if self.smooth_kernel > 1:
            k = self.smooth_kernel
            if k % 2 == 0:
                k += 1
            y = F.avg_pool2d(y, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)

        if self.normalize_lum:
            b = y.shape[0]
            y_flat = y.view(b, -1)
            y_min = y_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            y_max = y_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            y = (y - y_min) / (y_max - y_min + self.eps)

        if self.focus_mode == 'dark':
            s = 1.0 - y
        elif self.focus_mode == 'bright':
            s = y
        else:
            return None

        if self.dark_power != 1.0:
            s = s.clamp(min=0.0).pow(self.dark_power)

        w = 1.0 + self.focus_strength * s

        if self.norm_weight:
            w = w / (w.mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        return w

    def reduction_fn(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        return x

    def forward(self, pred, gt, is_hvi=None, lq=None, **kwargs):
        if is_hvi is None:
            is_hvi = self.is_hvi

        pred_gx, pred_gy = self.compute_spatial_gradient(pred)
        gt_gx, gt_gy = self.compute_spatial_gradient(gt)

        diff_x = torch.abs(pred_gx - gt_gx)
        diff_y = torch.abs(pred_gy - gt_gy)

        if self.focus_mode != 'balance':
            ref = lq if (self.weight_source == 'lq' and lq is not None) else gt
            weight = self.compute_weight_map(ref)
            if weight is not None:
                weight = weight.to(pred.dtype)
                weight_x = weight[:, :, :, 1:]
                weight_y = weight[:, :, 1:, :]
                diff_x = diff_x * weight_x
                diff_y = diff_y * weight_y

        loss_x = self.reduction_fn(diff_x)
        loss_y = self.reduction_fn(diff_y)

        loss = loss_x + loss_y
        return loss * self.loss_weight


# Aliases giữ nguyên tương thích 100% với IGMamba
RegionLSGDLoss = DarkFocusedGradientLoss
LSGDLoss = DarkFocusedGradientLoss
DarkFocusedSpatialGradientLoss = DarkFocusedGradientLoss
FGLoss = DarkFocusedGradientLoss
