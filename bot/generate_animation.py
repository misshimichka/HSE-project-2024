import torch

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from animate import normalize_kp

from load_weights import load_checkpoints, opt


generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)


def get_frames(file_path):
    cap = cv2.VideoCapture(file_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        frame.resize((256, 256))
        frames.append(frame)

    cap.release()
    return frames


def save_video(frames, output_path):
    video = cv2.VideoWriter(output_path, -1, 1, (256, 256))
    for frame in frames:
        video.write(frame)
    video.release()


def make_animation(
        source_image,
        driving_video,
        relative=True,
        adapt_movement_scale=True,
        cpu=False
):
    sources = []
    drivings = []
    predictions = []
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])
        if not cpu:
            kp_driving_initial = kp_detector(driving[:, :, 0].cuda())

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()

            kp_driving = kp_detector(driving_frame)

            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative,
                use_relative_jacobian=relative,
                adapt_movement_scale=adapt_movement_scale
            )

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return sources, drivings, predictions


def generate_animation(image_path, video_path, output_path):
    source_image = Image.open(image_path)
    frames = []
    try:
        frames = get_frames(video_path)
    except RuntimeError as rt_e:
        print(rt_e)
        return

    source_image = np.array(source_image.resize((256, 256)))

    sources, drivings, predictions = make_animation(
        source_image,
        frames,
        relative=opt.relative,
        adapt_movement_scale=opt.adapt_scale,
        cpu=opt.cpu
    )

    save_video(output_path, predictions)
