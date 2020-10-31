import gzip
import json
import sys

if __name__ == '__main__':
    assert len(sys.argv) > 2
    worker_num = int(sys.argv[1])
    file_name = sys.argv[2]

    if ".gz" not in file_name:
        file_name += ".gz"
    with gzip.open(file_name, "wt") as output_file:
        output_dict = {}
        for worker_id in range(1, 2*worker_num, 2):
            with gzip.open(f"worker{worker_id}.json.gz") as f:
                output_dict[f"worker{worker_id}"] = json.load(f)
        for server_id in range(2, 2 * worker_num + 1, 2):
            with gzip.open(f"server{server_id}.json.gz") as f:
                output_dict[f"server{server_id}"] = json.load(f)
        json.dump(output_dict, output_file)
