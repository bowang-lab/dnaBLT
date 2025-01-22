import os

import fsspec
import pyarrow as pa

# pyarrow needs the initialization from this import
import pyarrow.dataset  # pyright: ignore
import typer
from pyarrow.lib import ArrowInvalid
from rich.progress import track


def is_valid_arrow_file(path: str):
    try:
        dataset = pa.dataset.dataset(path, format="arrow")
        return True
    except ArrowInvalid:
        return False


app = typer.Typer()

S3_PREFIX = "s3://"


def get_fs(path: str, s3_profile: str | None = None) -> fsspec.AbstractFileSystem:
    if path.startswith("s3://"):
        if s3_profile is None:
            return fsspec.filesystem("s3")
        else:
            return fsspec.filesystem("s3", profile=s3_profile)
    else:
        return fsspec.filesystem("file")


@app.command()
def print_local_to_delete(
    blob_dir: str, local_dirs: list[str], s3_profile: str = "blt"
):
    for s in local_dirs:
        assert s.endswith("/"), "Dirs must end with /"
    assert blob_dir.endswith("/"), "Dirs must end with /"
    blob_fs = fsspec.filesystem("s3", profile=s3_profile)
    blob_files = blob_fs.find(blob_dir)
    for f in track(blob_files):
        size = blob_fs.info(f)["Size"]
        if not f.lower().endswith(".complete"):
            assert size != 0, f"Size was invalidly zero for {f}"

    blob_relative_paths = {f[len(blob_dir) - len(S3_PREFIX) :] for f in blob_files}
    local_fs = fsspec.filesystem("file")

    files_to_delete = []
    for local_dir in local_dirs:
        local_files = local_fs.find(local_dir)
        for f in local_files:
            relative_path = f[len(local_dir) :]
            if relative_path in blob_relative_paths and not os.path.islink(f):
                files_to_delete.append(f)
    print(len(files_to_delete))
    with open("/tmp/files_to_delete.txt", "w") as f:
        for file in files_to_delete:
            f.write(f"{file}\n")


@app.command()
def compare_local_to_blob(
    source_dirs: list[str], dst_dir: str, s3_profile: str = "blt"
):
    for s in source_dirs:
        assert s.endswith("/"), "Dirs must end with /"
    assert dst_dir.endswith("/"), "Dirs must end with /"
    assert len(source_dirs) != 0
    assert dst_dir.startswith("s3://")
    local_fs = fsspec.filesystem("file")
    dst_fs = fsspec.filesystem("s3", profile=s3_profile)
    source_to_files = {}
    all_local_files = set()
    for s in source_dirs:
        skipped = []
        if s not in source_to_files:
            source_to_files[s] = []
        for f in local_fs.find(s):
            if os.path.islink(f):
                continue
            if f.endswith(".COMPLETE") or f.endswith(".complete"):
                is_complete_file = True
                assert os.path.getsize(f) == 0, ".COMPLETE files should be empty"
            else:
                is_complete_file = False

            if not is_complete_file and os.path.getsize(f) == 0:
                skipped.append(f)
                continue
            if f.endswith(".arrow"):
                if not is_valid_arrow_file(f):
                    skipped.append(f)
                    continue

            source_to_files[s].append(f)
            all_local_files.add(f[len(s) :])
        print(s, len(source_to_files[s]), "skipped", len(skipped), skipped[:10])

    dst_files = dst_fs.find(dst_dir)
    print(dst_dir, len(dst_files))

    dst_file_set = {f[len(dst_dir) - len(S3_PREFIX) :] for f in dst_files}
    diff = all_local_files.symmetric_difference(dst_file_set)
    print("Local files", len(all_local_files))
    print("DST Files", len(dst_file_set))
    print("Symmetric difference", len(diff))
    dst_only_files = dst_file_set - all_local_files
    print("DST only", len(dst_only_files), list(dst_only_files)[:10])


if __name__ == "__main__":
    app()
