import os
import io
from pathlib import Path

from datasets import load_dataset, get_dataset_split_names
from starwhale import Image, dataset, MIMEType
from starwhale.consts.env import SWEnv

ROOT_DIR = Path(__file__).parent


def build_ds_from_local_fs(ds_uri):
    """
    build by sdk and with copy
    """
    ds = dataset(ds_uri, create="empty")
    print("preparing data...")
    data_path = ROOT_DIR / "data"
    lines = open(data_path / "meta.txt", encoding="utf-8").read().strip().split("\n")

    for line in lines:
        v = line.split("\t")
        img_path = data_path / v[0]
        with open(img_path, mode="rb") as image_file:
            ds.append(
                {
                    "image": Image(
                        fp=image_file.read(),
                        display_name=v[0],
                        mime_type=MIMEType.PNG,
                    ),
                    "text": v[1],
                }
            )
    ds.commit()
    ds.close()
    print("build done!")


def build_ds_from_hf(ds_uri, dataset_name: str = "lambdalabs/pokemon-blip-captions"):
    ds = dataset(ds_uri)
    hf_ds = load_dataset(dataset_name, cache_dir="cache")
    for row in hf_ds["train"]:
        img_byte_arr = io.BytesIO()
        row.get("image").save(img_byte_arr, format='PNG')
        ds.append(
            {
                "image": Image(
                    fp=img_byte_arr.getvalue(),
                    mime_type=MIMEType.PNG,
                ),
                "text": row.get("text"),
            }
        )
    ds.commit()
    ds.close()
    print("build done!")


if __name__ == "__main__":
    instance_uri = os.getenv(SWEnv.instance_uri)
    if instance_uri:
        _ds_uri = f"{instance_uri}/project/starwhale/dataset/pokemon-blip-captions"
    else:
        _ds_uri = f"pokemon-blip-captions"
    # build_ds_from_local_fs(_ds_uri)
    build_ds_from_hf(_ds_uri)