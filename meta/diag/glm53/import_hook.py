"""import_hook — run a callback exactly once, right after a module is imported.

The single import-system primitive shared by every GLM-5.3 engine shim
(sitecustomize patchers and comp_capture). It replaces the per-patcher
MetaPathFinder/Loader scaffolding that used to be copy-pasted (five times)
across sitecustomize.py: delegation around self, loader wrapping, re-entrancy
guards, and the already-imported direct-rebind path are all handled HERE, once.

Contract:
    run_after_import(target, on_loaded) -> bool

    * If ``target`` is already in sys.modules: calls ``on_loaded(module)``
      synchronously and returns True (the caller knows the module pre-existed).
    * Otherwise installs a meta-path finder and returns False; ``on_loaded``
      fires inside exec_module — after the module body has run, BEFORE the
      importing ``import`` statement returns to its caller. Exactly once per
      process, however many imports of ``target`` follow.

Chaining is preserved by construction: each hook delegates to the other
meta_path finders and wraps whatever loader comes back, so multiple hooks on
the SAME target all fire (innermost module body first, hooks in reverse
insertion order). Never call importlib.import_module inside ``on_loaded`` for
a module whose import may still be in progress — mutate the module in place
instead (that is the whole point of the loader-wrap).
"""
from __future__ import annotations

import importlib.abc
import sys


class _AfterImportLoader(importlib.abc.Loader):
    """Runs the real loader, then fires the hook exactly once."""

    def __init__(self, real, hook):
        self._real = real
        self._hook = hook

    def create_module(self, spec):
        # Defer to the real loader's create_module (SourceFileLoader has none
        # in CPython -> returns None -> Python creates a fresh module).
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module):
        self._real.exec_module(module)
        self._hook.fire(module)


class _AfterImportFinder(importlib.abc.MetaPathFinder):
    """Finds exactly one module name and wraps its loader with the hook.
    Returns None for every other module (zero overhead) and after firing."""

    def __init__(self, target, on_loaded):
        self._target = target
        self._on_loaded = on_loaded
        self._fired = False
        self._in_find = False  # re-entrancy guard (other hooks may target
        #                        the same module and iterate meta_path too)

    def fire(self, module):
        if self._fired:
            return
        self._fired = True
        self._on_loaded(module)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target or self._fired or self._in_find:
            return None
        self._in_find = True
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is None or spec.loader is None:
                    continue
                spec.loader = _AfterImportLoader(spec.loader, self)
                return spec
        finally:
            self._in_find = False
        return None


def run_after_import(target, on_loaded):
    """Call ``on_loaded(module)`` exactly once, after ``target`` is imported.

    See module docstring for the contract. Returns True iff the module was
    already imported (callback ran synchronously).
    """
    mod = sys.modules.get(target)
    if mod is not None:
        on_loaded(mod)
        return True
    sys.meta_path.insert(0, _AfterImportFinder(target, on_loaded))
    return False
