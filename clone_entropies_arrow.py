import pyarrow as pa
import pyarrow.ipc as ipc

# Input and output file paths
input_path = "outputs/entropies_validation.arrow"
output_path = "outputs/entropies_validation_cloned.arrow"

# Read the original Arrow file
def read_arrow_table(path):
    with open(path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()
    return table

def write_arrow_table(table, path):
    with open(path, "wb") as f:
        writer = ipc.RecordBatchFileWriter(f, table.schema)
        writer.write_table(table)
        writer.close()

def main():
    table = read_arrow_table(input_path)
    # Clone the table 5x
    tables = [table] * 5
    concatenated = pa.concat_tables(tables)
    write_arrow_table(concatenated, output_path)
    print(f"Cloned table written to {output_path}")

if __name__ == "__main__":
    main()
