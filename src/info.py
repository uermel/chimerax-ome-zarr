# vim: set expandtab shiftwidth=4 softtabstop=4:

import logging
import traceback
from typing import Any, Dict, List, Tuple

from chimerax.core.models import Model
from chimerax.open_command import FetcherInfo, OpenerInfo


class OMEZarrOpenerInfo(OpenerInfo):
    check_path = False

    def open(self, session, path, file_name, **kw) -> Tuple[List[Model], str]:
        from .open import open_ome_zarr
        from .util.env import env_if_mac

        env_if_mac()

        try:
            ret = open_ome_zarr(session, path, **kw)
            return ret
        except Exception:
            logging.error(traceback.format_exc())

        return [], "Error opening file."

    @property
    def open_args(self) -> Dict[str, Any]:
        from chimerax.core.commands import ListOf, StringArg

        return {"scales": ListOf(StringArg)}


class NGFFFetcherInfo(FetcherInfo):
    def fetch(self, session, ident, format_name, ignore_cache, **kw) -> Tuple[List[Model], str]:
        from .open import open_ome_zarr
        from .util.env import env_if_mac

        env_if_mac()

        try:
            ret = open_ome_zarr(session, [ident], **kw)
            return ret
        except Exception:
            logging.error(traceback.format_exc())

        return [], "Error opening file."

    @property
    def fetch_args(self) -> Dict[str, Any]:
        from chimerax.core.commands import ListOf, StringArg

        return {"scales": ListOf(StringArg)}
