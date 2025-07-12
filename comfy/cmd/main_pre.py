"""
This should be imported before entrypoints to correctly configure global options prior to importing packages like torch and cv2.

Use this instead of cli_args to import the args:

>>> from comfy.cmd.main_pre import args

It will enable command line argument parsing. If this isn't desired, you must author your own implementation of these fixes.
"""
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

if 'OTEL_METRICS_EXPORTER' not in os.environ:
    os.environ['OTEL_METRICS_EXPORTER'] = 'none'

import ctypes
import importlib.util
import logging
import shutil
import sys
import warnings

from .. import options
from ..app import logger

this_logger = logging.getLogger(__name__)

options.enable_args_parsing()
if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings('ignore', category=FutureWarning, message=r'`torch\.cuda\.amp\.custom_fwd.*')
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated, please import via timm.models", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated, please import via timm.layers", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inheritance class _InstrumentedApplication from web.Application is discouraged", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Please import `gaussian_filter` from the `scipy.ndimage` namespace; the `scipy.ndimage.filters` namespace is deprecated", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support")
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version .* ONNX Runtime supports Windows 10 and above, only.")
log_msg_to_filter = "NOTE: Redirects are currently not supported in Windows or MacOs."
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
    lambda record: log_msg_to_filter not in record.getMessage()
)

from ..cli_args import args

if args.cuda_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
    this_logger.info("Set cuda device to: {}".format(args.cuda_device))

if args.deterministic:
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

if args.oneapi_device_selector is not None:
    os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
    this_logger.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

try:
    from . import cuda_malloc
except Exception:
    pass


def _fix_pytorch_240():
    """Fixes pytorch 2.4.0"""
    torch_spec = importlib.util.find_spec("torch")
    for folder in torch_spec.submodule_search_locations:
        lib_folder = os.path.join(folder, "lib")
        test_file = os.path.join(lib_folder, "fbgemm.dll")
        dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
        if os.path.exists(dest):
            break

        try:
            with open(test_file, 'rb') as f:
                contents = f.read()
                # todo: dubious
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                _ = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                this_logger.warning("Detected pytorch version with libomp issue, trying to patch")
                try:
                    shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
                except Exception as exc_info:
                    this_logger.error("While trying to patch a fix for torch 2.4.0, an error occurred, which means this is unlikely to work", exc_info=exc_info)
        except:
            pass


def _configure_logging():
    logging_level = args.logging_level
    if len(args.workflows) > 0 or args.distributed_queue_worker or args.distributed_queue_frontend or args.distributed_queue_connection_uri is not None:
        logging.basicConfig(level=logging_level, stream=sys.stderr)
    else:
        logger.setup_logger(logging_level)


_configure_logging()
_fix_pytorch_240()
__all__ = ["args"]
