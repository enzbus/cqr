import platform
from pathlib import Path
import subprocess
from setuptools import Extension, setup

from setuptools.command.build_ext import build_ext #import SubCommand

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


class CMake(build_ext):
    
    def run(self):

        assert len(self.extensions) == 1
        
        # breakpoint()
        extras = []
        if platform.system() == 'Windows':
            extras += ["-G", "MinGW Makefiles"] 
        subprocess.run(
            ['cmake', f'-B{self.build_temp}',
            f'-DCMAKE_INSTALL_PREFIX={self.build_lib}'] + extras, check=True)
        subprocess.run(['cmake', '--build', self.build_temp], check=True)
        subprocess.run(['cmake', '--install', self.build_temp], check=True)

        # this is the name that setuptools wants for the shared lib
        # that goes in site-packages/
        shlib_filename = self.get_ext_filename(self.extensions[0].name)

        print('TARGET NAME', shlib_filename)

        # rename here the compiled shared library
        _builds = list(Path(self.build_lib).iterdir())

        # on win, MinGW builds a .dll.a as well, which we don't need
        _builds = [el for el in _builds if not el.name.endswith('dll.a')]

        # make sure nothing else is there
        breakpoint()
        #assert len(_builds) == 1
        _build = _builds[0]

        _build.rename(_build.parent / shlib_filename)

        # breakpoint()

        if self.inplace: # as is done in setuptools, for pip install -e
            # breakpoint()
            self.copy_extensions_to_source()
        # breakpoint()


setup(
    name='project_euromir',
    version='0.0.1',
    packages=['project_euromir', 'project_euromir.tests'],
    #package_data={'project_euromir':['libproject_euromir.so']},
    #libraries=[('project_euromir', {'sources':['project_euromir/linear_algebra.c'],})],
    ext_modules=[
        Extension(
            name="project_euromir",
            # I guess these are unused
            sources=['project_euromir/linear_algebra.c'],
        ),
    ],
    # I guess this is unused
    include_dirs=['project_euromir/'],
    #libraries=[True],
    cmdclass={'build_ext':CMake},
    install_requires=["numpy", "scipy"]
)
