import yaml
from collections import OrderedDict

import torch

from face_alignment.modules.keypoint_detector import KPDetector
import face_alignment.modules.generator as GEN
from sync_batchnorm import DataParallelWithCallback

import face_detection

from main2 import opt
from generate_sticker import models


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num

    generator = getattr(GEN, opt.generator)(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'],
                                            **{'mbunit': opt.mbunit, 'mb_spatial': opt.mb_spatial,
                                               'mb_channel': opt.mb_channel})
    if not cpu:
        generator.cuda()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

    ckp_generator = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)

    ckp_kp_detector = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def load_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "h94/IP-Adapter",
        cache_dir="adapter"
    )
    snapshot_download(
        repo_id="latent-consistency/lcm-lora-sdv1-5",
        cache_dir="lcm"
    )

    for key in models:
        snapshot_download(
            models[key],
            cache_dir=f"{key}"
        )

    detector = face_detection.build_detector(
        "DSFDDetector",
        confidence_threshold=.5,
        nms_iou_threshold=.3
    )
