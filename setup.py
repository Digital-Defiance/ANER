"""Build libaner while installing the Python package."""

import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent
DYLIB = ROOT / "lib" / "build" / "libaner.0.dylib"


class build_py(_build_py):
    def run(self):
        subprocess.check_call(["make", "build/libaner.0.dylib"], cwd=ROOT / "lib")
        super().run()
        dest = Path(self.build_lib) / "aner" / "libaner.0.dylib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DYLIB, dest)


setup(
    name="aner",
    version="0.1.0",
    description="Apple Neural Engine Runtime for ensemble numerical graphs",
    package_dir={"": "python"},
    packages=["aner"],
    package_data={"aner": ["*.dylib"]},
    install_requires=["numpy"],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["aner-modal=aner.modal:cli"]},
    cmdclass={"build_py": build_py},
)
