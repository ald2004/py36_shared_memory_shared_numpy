import torch
import torchvision
import multiprocessing
import fire
import cv2
from typing import Any, Dict, List, Tuple, Union
import sys, os, time
from gevent import monkey
import traceback
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
from utils.cuda_shared import get_cuda_memory_from_vpf
from utils.logger import setup_logger
from fvcore.common.timer import Timer
import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient import utils
from tracking.sort import Sort
from configs import get_cfg
import uuid
import shared_numpy as snp

logger = setup_logger(name='person_track')
# init default config and merge from base.yaml
# default values configs/__init__.py
cfg = get_cfg("configs/BASE.yaml")
monkey.patch_all()
model_w = 640
model_h = 480
model_c = 3
model_c = 1


class Worker_detector(multiprocessing.Process):
    def __init__(self, queue: snp.Queue, idx, **kwargs):
        super(Worker_detector, self).__init__()
        self.queue = queue
        self.idx = idx
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, name: str, value: Any) -> None:
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> Any:
        return self._fields[name]

    def preprocess(self, img, format, dtype, c, h, w, scaling, protocol):
        # Swap to CHW if necessary
        if protocol == "grpc":
            pass
        elif protocol == "cuda_shared_memory":
            if scaling == "INCEPTION":
                # make border
                if format == "FORMAT_NCHW":
                    return torch.as_tensor((img / 255.), dtype=torch.float32 if dtype == "FP32" else torch.int8)
            else:
                pass

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        intput_w = cfg.BUSI_TYPE_LIST.BUSI_A.INPUT_W
        input_h = cfg.BUSI_TYPE_LIST.BUSI_A.INPUT_H
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        r_w = intput_w / origin_w
        r_h = input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (intput_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (intput_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # print(pred.shape)
        # to a torch Tensor
        # pred = torch.Tensor(pred).cuda()
        pred = torch.Tensor(pred).cpu()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        conf_thresh = cfg.BUSI_TYPE_LIST.BUSI_A.CONF_THRESH
        iou_thresh = cfg.BUSI_TYPE_LIST.BUSI_A.IOU_THRESHOLD
        si = scores > conf_thresh
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        # logger.debug(boxes)
        # logger.debug(boxes.shape) # torch.Size([82, 4])
        # logger.debug(scores)
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=iou_thresh).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid

    def run(self):
        gpuId = self.get("gpu_id")
        encFile = self.get("rtsp")
        model_name = self.get("model_name")
        server_url = self.get("server_url")
        verbose = self.get("verbose")
        cam_id = self.get("cameraid")
        fences = self.get("fences")
        model_w = self.get("model_w")
        model_h = self.get("model_h")
        model_c = self.get("model_c")
        outputdim = self.get("outputdim")
        verbose_output_filename = self.get("verbose_output_filename")
        outputtensorname = self.get("outputtensorname")
        inputtensorname = self.get("inputtensorname")

        logger.debug(f"process {self.name} cam_id {cam_id} camera_address {encFile} gpu_id {gpuId} ")
        logger.debug(f"model_name {model_name} server_url {server_url} inputtensorname {inputtensorname}")
        logger.debug(f"outputtensorname {outputtensorname}")


class Worker_decoder(multiprocessing.Process):
    def __init__(self, queue: snp.Queue, idx, **kwargs):
        super(Worker_decoder, self).__init__()
        self.queue = queue
        self.idx = idx
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, name: str, value: Any) -> None:
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> Any:
        return self._fields[name]

    def run(self):
        gpuId = self.get("gpu_id")
        encFile = self.get("rtsp")
        verbose = self.get("verbose")
        gpuId = 0 if gpuId < 0 else gpuId

        try:
            nvDec = nvc.PyNvDecoder(encFile, gpuId,
                                    {'rtsp_transport': 'tcp', 'max_delay': '5000000', 'bufsize': '30000k'})
            nvCvt = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvDec.Format(), nvc.PixelFormat.YUV420, gpuId)
            nvRes = nvc.PySurfaceResizer(model_w, model_h, nvCvt.Format(), gpuId)
            to_rgb = nvc.PySurfaceConverter(model_w, model_h, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, gpuId)

            nvDwn = nvc.PySurfaceDownloader(model_w, model_h, nvc.PixelFormat.BGR, 0)
            # rawFrameNV12 = snp.ndarray((3000, 3000), dtype=np.float32)
            # success = nvDwn.DownloadSingleSurface(cvtNV12_BGR.Execute(nvDec.DecodeSingleSurface()), frameBGR)
            # while 1:
            for _ in range(10):
                rawSurface = nvDec.DecodeSingleSurface()
                yuvSurface = nvCvt.Execute(rawSurface)
                resSurface = nvRes.Execute(yuvSurface)
                rgb_byte = to_rgb.Execute(resSurface)
                frameBGR = snp.ndarray((model_w * model_h * model_c), dtype=np.uint8)
                # frameBGR = np.ndarray(shape=((rgb_byte.Width() * rgb_byte.Height())), dtype=np.uint8)
                success = nvDwn.DownloadSingleSurface(rgb_byte, frameBGR)
                if not (success):
                    raise RuntimeError("DecodeSingleFrame error ...") from exec
                # frame = snp.ndarray((3000, 3000), dtype=np.float32)

                self.queue.put(frameBGR)
                frameBGR.close()
            self.queue.put(None)


        except:
            traceback.print_exception(*sys.exc_info())
            raise RuntimeError() from exec


import cv2


def test_con(q: snp.Queue):
    while True:
        out: snp.ndarray = q.get()
        if out is not None:
            print(f"obtained array {out.shape}")
            out.resize((model_h,model_w, 3))
            cv2.imwrite(f'/dev/shm/{uuid.uuid4().hex}.jpg', out)
            out.close()
            out.unlink()
        else:
            break


def start_work():
    logger.debug("===============")
    for k, v in cfg.items():
        if k == "DECODER":
            logger.debug(f"k is:{k}, v is :{v}")
    logger.debug("===============")

    cameralist = cfg.DECODER.CAMERA_LIST
    available_gpus = cfg.DECODER.GPU_ASSIGNED_LIST
    assert len(cameralist) > 0

    processes = list()

    q = snp.Queue()  # Build a shared memory queue per camera ... this is a test so one queue
    p = multiprocessing.Process(target=test_con, args=(q,))
    p.start()
    # for i in range(0, len(cameralist)):
    for i in range(0, 1):
        p = Worker_decoder(queue=q, idx=i, rtsp=cameralist[i], gpu_id=available_gpus[i], verbose=False)
        p.start()
        processes.append(p)
    [proc.join() for proc in processes]
    p.join()


if __name__ == "__main__":
    fire.Fire()
