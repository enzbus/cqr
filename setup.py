import platform
from pathlib import Path
import subprocess
from setuptools import Extension, setup

from setuptools.command.build_ext import build_ext #import SubCommand
from setuptools.command.build_py import build_py #import SubCommand
from setuptools.command.develop import develop #import SubCommand


# Better solution:
# - override build_py to run cmake with the windows specific option
# - use the editable flag to redirect the cmake install dir to the source dir
# - remove hack to rename libproject_euromir.so/dll/dylib to whatever build_ext
#   was using
# - make sure bdist_wheel takes whatever is in the build_lib directory as the
#   wheel content (might need to append to an internal list, hopefully not)
# - go back to putting the sh lib in the package directory, same level as the
#   __init__.py
# - override the bdist_wheel Command, probably only its get_tag method, to
#   rename the wheel to the correct platform. cpython version is py3-none ! 

def _run_cmake(extras=()):
    if platform.system() == 'Windows':
        extras += ["-G", "MinGW Makefiles"]
    subprocess.run(['cmake', '-Bbuild'] + list(extras), check=True)
    subprocess.run(['cmake', '--build', 'build'], check=True)
    subprocess.run(['cmake', '--install', 'build'], check=True)

class CmakeDevelop(develop):
    # maybe we don't need this class!
    def run(self):
        _run_cmake()
        super().run()

class CmakeBuild(build_py):

    def run(self):
        extras = []
        if not self.editable_mode:
            _libdir = str(Path.cwd()/self.build_lib/'project_euromir')
            extras += [f"-DMOVE_SHLIB_TO:FILEPATH={_libdir}"]
        _run_cmake(extras)
        super().run()

setup(
    name='project_euromir',
    version='0.0.1',
    packages=['project_euromir', 'project_euromir.tests'],
    #package_data={'project_euromir':['libproject_euromir.so']},
    #libraries=[('project_euromir', {'sources':['project_euromir/linear_algebra.c'],})],
    # ext_modules=[
    #     Extension(
    #         name="project_euromir",
    #         # I guess these are unused
    #         sources=['project_euromir/linear_algebra.c'],
    #     ),
    # ],
    # I guess this is unused
    # include_dirs=['project_euromir/'],
    #libraries=[True],
    cmdclass={
        #'build_ext':CMake, 
        'build_py':CmakeBuild,
        'develop':CmakeDevelop,
        },
    install_requires=["numpy", "scipy"]
)
