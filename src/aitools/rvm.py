import os
import cv2
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from typing import Optional
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)

def __read_impil(inp_buf):
    if isinstance(inp_buf, bytes):
        img = np.asarray(bytearray(inp_buf), dtype='uint8')
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(inp_buf, str) and os.path.exists(inp_buf):
        image = Image.open(inp_buf)
        image = image.convert('RGB')
    elif isinstance(inp_buf, str) and 'http' == inp_buf.strip()[:4]:
        resp = requests.get(inp_buf)
        image = Image.open(BytesIO(resp.content))
        image = image.convert('RGB')
    elif isinstance(inp_buf, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(inp_buf, cv2.COLOR_BGR2RGB))
    elif isinstance(inp_buf, Image.Image):
        image = inp_buf
    else:
        raise ValueError("Error: Not support this type image buffer(byte, str, np.ndarry, PIL.Image)")
    
    return image

import pudb
def convert_image(model,
                  img_buf,
                  downsample_ratio: Optional[float] = None,
                  output_composition: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):

    # pudb.set_trace() 
    transform = transforms.Compose([
        # transforms.Resize([512, 512]),
        transforms.ToTensor()]
    )
        

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    try:
        with torch.no_grad():
            rec = [None] * 4
            image = __read_impil(img_buf) 
            src = transform(image)
            src = torch.unsqueeze(src, 0)

            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])

            src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
            fgr, pha, *rec = model(src, *rec, downsample_ratio)

            if output_foreground is not None:
                to_pil_image(fgr[0][0]).save('fimg.png')
            if output_composition is not None:
                fgr = fgr * pha.gt(0)
                com = torch.cat([fgr, pha], dim=-3)
                to_pil_image(com[0][0]).save('comp.png')
    except Exception as e:
        print(e)

