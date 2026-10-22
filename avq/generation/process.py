import argparse
from pathlib import Path
from avq.generation.utils import json_load, json_dump


def process_gfid_jsons(gfid_root):
    records = []

    # gfid_root/<dataset>/<seed>/<experiment>/gfid.json
    gfid_paths = gfid_root.glob("*/*/*/gfid.json")
    for gfid_path in sorted(gfid_paths):
        record = json_load(file_path=gfid_path)

        experiment_dir = gfid_path.parent
        experiment_name = experiment_dir.name
        model_seed = experiment_dir.parent.name
        dataset = experiment_dir.parent.parent.name

        assert experiment_name == record["experiment"]
        assert model_seed.split('-')[-1] == str(record["model_seed"])
        assert dataset == record["dataset"]

        records.append(record)

    top_k = records[-1]["top_k"]
    sampling_temperature = records[-1]["sampling_temperature"]
    file_name = f"gfid_{dataset}_{model_seed}_ST{sampling_temperature}_TK{top_k}.json"

    return records, file_name


def main():
     # 0. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Compute generation FID on generated shards.")
    parser.add_argument("--gfid_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    gfid_root = Path(args.gfid_root)
    output_dir = Path(args.output_dir)

    gfid_records, output_file_name = process_gfid_jsons(gfid_root)
    output_path = output_dir / output_file_name
    json_dump(gfid_records, output_path)
    print(f"Saved gFID master file to: {output_path}")


if __name__ == "__main__":

    main()
