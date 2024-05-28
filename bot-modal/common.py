import modal
import face_detection

model_flowers_id = "misshimichka/pix2pix_people_flowers_v2"
model_cat_id = "misshimichka/pix2pix_cat_ears"
model_clown_id = "misshimichka/pix2pix_clown_faces"
model_butterfly_id = "misshimichka/pix2pix_butterflies"
model_pink_id = "misshimichka/pix2pix_pink_hair"
model_id = "misshimichka/instructPix2PixCartoon_4860_ckpt"

models = {
    "default": model_id,
    "flowers": model_flowers_id,
    "cat": model_cat_id,
    "butterfly": model_butterfly_id,
    "clown": model_clown_id,
    "pink": model_pink_id
}


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


app = modal.App("hse-project-1k")
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget")
    .run_commands("pip install --no-cache-dir torch torchvision torchaudio torchdiffeq")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("gdown")
    .pip_install("git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git")
    .pip_install("git+https://github.com/huggingface/diffusers.git")
    .run_commands(
        "git clone https://github.com/harlanhong/ICCV2023-MCNET.git "
        "&& cp -r /ICCV2023-MCNET/* /root/ "
        "&& cd /root "
        "&& gdown 1dk973WGzD7n9NlIw3cl6J4bqHKObgPs2 "
        "&& wget https://media1.tenor.com/m/HnJ-a1i_Bp8AAAAC/patrick-bateman-sigma.gif "
        "&& wget https://media1.tenor.com/m/EkTCtB-0hncAAAAd/the-rock-eyebrow-the-rock-sus.gif"
        "&& mkdir checkpoint "
        "&& cd checkpoint "
        "&& gdown 1_cCo7u_7G31krLX4c6ZEWD5CURSLaQ-g"
    )
    .run_function(load_weights)
)
