from os import path
import importlib
import glob
from _external_packages.allennlp.allennlp.constants import PACKAGE_NAME_TRANSFORMERS__
from pytorch_transformers import AutoTokenizer


def download_all_files():
    transformers_package_path = path.dirname(importlib.util.find_spec(PACKAGE_NAME_TRANSFORMERS__).origin)
    for possible_tokenization_module_path in glob.glob(path.join(transformers_package_path, 'tokenization_*.py')):
        module_name = f'{PACKAGE_NAME_TRANSFORMERS__}.{path.splitext(path.basename(possible_tokenization_module_path))[0]}'
        print(f"Downloading tokenization files for module " + module_name)
        m = importlib.import_module(module_name)
        if hasattr(m, 'PRETRAINED_VOCAB_FILES_MAP'):
            model_vocab_file_map = getattr(m, 'PRETRAINED_VOCAB_FILES_MAP')
            for model_name in model_vocab_file_map[next(iter(model_vocab_file_map))]:
                AutoTokenizer.from_pretrained(model_name, cache_dir=path.abspath(path.dirname(__file__)))


if __name__ == '__main__':
    download_all_files()
