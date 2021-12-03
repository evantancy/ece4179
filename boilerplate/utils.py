import torch


def order_tensor(t_: torch.Tensor):
    """Ensure proper order of tensor dims since T.ToTensor() is inconsistent"""

    # i.e. (B,H,W,C) -> (B,C,H,W)
    if t_.ndim == 4:
        h, w, c = t_.shape[1:4]
        # ensure channels dim is at idx 1
        if c < h or c < w:
            return t_.permute(0, 3, 1, 2).contiguous()
        else:
            return t_
    else:
        raise NotImplementedError
