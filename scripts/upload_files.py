import argparse

from azureml.core import Workspace


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_path', type=str)

    args = parser.parse_args()

    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()

    datastore.upload(src_dir=args.data_dir, target_path=args.target_path, overwrite=True)


