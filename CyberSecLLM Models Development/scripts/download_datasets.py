"""
Dataset download script for CyberSecLLM.

Downloads the six publicly available security datasets from Kaggle:

IIS3D Collection (DOI: 10.34740/kaggle/dsv/12479689):
  - CSE-CIC-IDS2018
  - UNB-CIC-IoT2023
  - UNSW-NB15

ICS3D Collection (DOI: 10.34740/kaggle/dsv/12483891):
  - Microsoft Azure Cloud
  - Edge-IIoTset
  - Kubernetes/Docker Containers

Usage:
    python scripts/download_datasets.py --output_dir data/
"""

import os
import sys
import argparse
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = {
    "IIS3D": {
        "kaggle_id": "rogernickana/iis3d-integrated-idps-security-3datasets",
        "doi": "10.34740/kaggle/dsv/12479689",
        "description": (
            "Integrated IDPS Security 3Datasets: CSE-CIC-IDS2018, "
            "UNB-CIC-IoT2023, UNSW-NB15"
        ),
        "subdatasets": [
            "CSE-CIC-IDS2018",
            "UNB-CIC-IoT2023",
            "UNSW-NB15",
        ],
    },
    "ICS3D": {
        "kaggle_id": "rogernickana/ics3d-integrated-cloud-security-3datasets",
        "doi": "10.34740/kaggle/dsv/12483891",
        "description": (
            "Integrated Cloud Security 3Datasets: Microsoft Cloud, "
            "EdgeIIoT Federated Learning, Kubernetes vs Docker"
        ),
        "subdatasets": [
            "Microsoft-Azure-Cloud",
            "Edge-IIoTset",
            "Kubernetes-Docker",
        ],
    },
}


def check_kaggle_api():
    """Verify Kaggle API is configured."""
    try:
        import kaggle
        return True
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Kaggle API not configured: {e}")
        logger.info(
            "Set up credentials: https://www.kaggle.com/docs/api#authentication"
        )
        return False


def download_dataset(kaggle_id, output_dir):
    """Download a dataset from Kaggle."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "-p", output_dir,
        "--unzip",
    ]
    logger.info(f"Downloading: {kaggle_id}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Download complete: {output_dir}")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download CyberSecLLM training datasets"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/",
        help="Output directory for downloaded datasets",
    )
    parser.add_argument(
        "--collection", type=str, default="all",
        choices=["all", "IIS3D", "ICS3D"],
        help="Which dataset collection to download",
    )
    args = parser.parse_args()

    if not check_kaggle_api():
        sys.exit(1)

    collections = (
        list(DATASETS.keys())
        if args.collection == "all"
        else [args.collection]
    )

    for collection in collections:
        info = DATASETS[collection]
        logger.info(f"\n{'='*60}")
        logger.info(f"Collection: {collection}")
        logger.info(f"DOI: {info['doi']}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"Sub-datasets: {', '.join(info['subdatasets'])}")
        logger.info(f"{'='*60}")

        output = os.path.join(args.output_dir, collection)
        download_dataset(info["kaggle_id"], output)

    logger.info("\nAll downloads complete.")
    logger.info(f"Data saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
