import torch

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from skimage.transform import resize
from skimage import img_as_ubyte

from animate import normalize_kp

from load_weights import load_checkpoints, opt

styles = {
    "wow": "wow-grey.mp4",
    "sigma": "patrick-bateman-sigma.gif",
    "rock": "the-rock-eyebrow-the-rock-sus.gif"
}

generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)


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


def generate_animation(image_path, style, output_path):
    source_image = imageio.imread(image_path)
    reader = imageio.get_reader(styles[style])
    metadata = reader.get_meta_data()
    if "fps" in metadata:
        fps = metadata["fps"]
    else:
        fps = 15
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    sources, drivings, predictions = make_animation(
        source_image,
        driving_video,
        relative=opt.relative,
        adapt_movement_scale=opt.adapt_scale,
        cpu=opt.cpu
    )

    imageio.mimsave(output_path, [img_as_ubyte(p) for p in predictions], fps=fps)

    with imageio.imopen(output_path[:-4] + ".webm", "w", plugin="pyav") as out_file:
        out_file.init_video_stream("vp9", fps=fps)

        for frame in imageio.imiter(output_path, plugin="pyav"):
            out_file.write_frame(frame)
