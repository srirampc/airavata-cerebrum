import argparse
import base64
import json
import os

# notebook_path = "V1L4-Notebook.ipynb"
# output_dir = "images"
# Generated using Claude

def main(notebook_path: str, output_dir: str):
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for cell_idx, cell in enumerate(nb["cells"]):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime, content in data.items():
                if mime.startswith("image/"):
                    ext = mime.split("/")[1]
                    img_data = (
                        "".join(content) if isinstance(content, list) else content
                    )
                    filename = f"images/cell_{cell_idx}.{ext}"
                    with open(filename, "wb") as img_file:
                        img_file.write(base64.b64decode(img_data))
                    print(f"Saved: {filename}")

    with open(notebook_path, "r") as f:
        nb = json.load(f)

    for cell_idx, cell in enumerate(nb["cells"]):
        attachments = cell.get("attachments", {})
        for filename, mime_dict in attachments.items():
            for mime_type, b64_data in mime_dict.items():
                ext = mime_type.split("/")[1]  # e.g. "png", "jpeg"
                save_name = f"cell_{cell_idx}_{filename}"
                save_path = os.path.join(output_dir, save_name)
                with open(save_path, "wb") as img_file:
                    img_file.write(base64.b64decode(b64_data))
                print(f"Saved: {save_path}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from Jupyter Notebook."
    )
    parser.add_argument(
        "-i",
        "--notebook_path",
        required=True,
        help="Path to a notebook.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output directory.",
    )

    rargs = parser.parse_args()
    main(rargs.notebook_path, rargs.output_dir)
