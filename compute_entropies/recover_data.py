from tqdm import tqdm
import pyarrow as pa
import pyarrow.ipc as ipc
import argparse

def main(args):
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])

    # Open the corrupted arrow file for reading
    with open(f"/cluster/projects/bwanggroup/open-genome/entropies_rank{args.entropy_file}.arrow", "rb") as f:
        # Skip header (for example, 8 bytes)
        header = f.read(8)

        _ = ipc.read_message(f)  # schema message

        # Open a single output file to write all record batches
        with pa.OSFile(f"/cluster/projects/bwanggroup/open-genome/16b{args.entropy_file + 1}.arrow", 'wb') as outfile:
            writer = ipc.new_file(outfile, schema)

            # Read and write record batches continuously
            with tqdm(total=15625) as pbar:
                while True:
                    try:
                        message = ipc.read_message(f)
                        if message is None:
                            break

                        if message.type == 'record batch':
                            batch = ipc.read_record_batch(message, schema)
                            writer.write(batch)
                        pbar.update(1)
                    except Exception as e:
                        print("Error reading message:", e)
                        break

            writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data recovery"
    )
    parser.add_argument(
        "--entropy_file",
        type=int,
        required=True,
        help="The entropy file number",
    )
    args = parser.parse_args()
    main(args)