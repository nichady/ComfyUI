from __future__ import annotations

import copy
import gc
import json
import threading
import uuid
from typing import Optional

from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..cmd.folder_paths import init_default_paths  # pylint: disable=import-error
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.make_mutable import make_mutable
from ..distributed.server_stub import ServerStub
from ..execution_context import current_execution_context

_prompt_executor = threading.local()


def _execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None) -> dict:
    configuration = copy.deepcopy(configuration) if configuration is not None else None
    execution_context = current_execution_context()
    if len(execution_context.folder_names_and_paths) == 0 or configuration is not None:
        init_default_paths(execution_context.folder_names_and_paths, configuration, replace_existing=True)

    return __execute_prompt(prompt, prompt_id, client_id, progress_handler, configuration)


def __execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None) -> dict:
    from .. import options
    from ..cmd.execution import PromptExecutor

    progress_handler = progress_handler or ServerStub()
    prompt_executor: PromptExecutor = None
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
    except (LookupError, AttributeError):
        if configuration is None:
            options.enable_args_parsing()
        else:
            from ..cmd.main_pre import args
            args.clear()
            args.update(configuration)

        # todo: deal with new caching features
        prompt_executor = PromptExecutor(progress_handler)
        prompt_executor.raise_exceptions = True
        _prompt_executor.executor = prompt_executor

    try:
        prompt_mut = make_mutable(prompt)
        from ..cmd.execution import validate_prompt
        validation_tuple = validate_prompt(prompt_mut)
        if not validation_tuple.valid:
            if validation_tuple.node_errors is not None and len(validation_tuple.node_errors) > 0:
                validation_error_dict = validation_tuple.node_errors
            elif validation_tuple.error is not None:
                validation_error_dict = validation_tuple.error
            else:
                validation_error_dict = {"message": "Unknown", "details": ""}
            raise ValueError(json.dumps(validation_error_dict))

        if client_id is None:
            prompt_executor.server = ServerStub()
        else:
            prompt_executor.server = progress_handler

        prompt_executor.execute(prompt_mut, prompt_id, {"client_id": client_id},
                                execute_outputs=validation_tuple.good_output_node_ids)
        return prompt_executor.outputs_ui
    except Exception as exc_info:
        raise exc_info


def _cleanup():
    from ..cmd.execution import PromptExecutor
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
        # this should clear all references to output tensors and make it easier to collect back the memory
        prompt_executor.reset()
    except (LookupError, AttributeError):
        pass
    from .. import model_management
    model_management.unload_all_models()
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except:
        pass


class Comfy:
    """
    Similar to embedded_comfy_client but not asynchronous and doesn't support queueing.
    Not safe for concurrent use.
    """

    def __init__(self, configuration: Optional[Configuration] = None):
        self._progress_handler = ServerStub()
        self._configuration = configuration

    def __enter__(self):
        return self

    def __exit__(self, *args):
        _cleanup()

    def execute_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None) -> dict:
        prompt_id = prompt_id or str(uuid.uuid4())
        client_id = client_id or self._progress_handler.client_id or None
        return _execute_prompt(
            make_mutable(prompt),
            prompt_id,
            client_id,
            self._progress_handler,
            self._configuration,
        )
