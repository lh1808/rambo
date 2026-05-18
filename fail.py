(generic) ubuntu@192.168.159.209 /home/ubuntu/da-hf1-rubin (master) $ copier copy --trust --vcs-ref master /home/ubuntu/copier/ /home/ubuntu/da-hf1-rubin/
Traceback (most recent call last):
  File "/opt/conda/envs/generic/bin/copier", line 6, in <module>
    from copier.cli import CopierApp
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/copier/__init__.py", line 6, in <module>
    from .main import *  # noqa: F401,F403
    ^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/copier/main.py", line 35, in <module>
    from pydantic import ConfigDict, PositiveInt
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/pydantic/_migration.py", line 4, in <module>
    from pydantic.warnings import PydanticDeprecatedSince20
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/pydantic/warnings.py", line 5, in <module>
    from .version import version_short
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/pydantic/version.py", line 7, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/pydantic_core/__init__.py", line 6, in <module>
    from typing_extensions import Sentinel
ImportError: cannot import name 'Sentinel' from 'typing_extensions' (/opt/conda/envs/generic/lib/python3.11/site-packages/typing_extensions.py)
