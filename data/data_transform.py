
import torch


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)


def center_crop_th_tw(clip, th, tw, top_crop):
    # import ipdb;ipdb.set_trace()
    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        # hxw 720x1280  thxtw 320x640  hw_raito 9/16 > tr_ratio 8/16  newh=1280*320/640=640  neww=1280 
        new_h = int(w * tr)
        new_w = w
    else:
        # hxw 720x1280  thxtw 480x640  hw_raito 9/16 < tr_ratio 12/16   newh=720 neww=720/(12/16)=960  
        # hxw 1080x1920  thxtw 720x1280  hw_raito 9/16 = tr_ratio 9/16   newh=1080 neww=1080/(9/16)=1920  
        new_h = h
        new_w = int(h / tr)
    
    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)


class CenterCropResizeVideo:
    '''
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    '''

    def __init__(
            self,
            size,
            top_crop=False, 
            interpolation_mode="bilinear",
    ):

        self.size = size
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_center_crop = center_crop_th_tw(clip, self.size[0], self.size[1], top_crop=self.top_crop)
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size,
                                         interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"
