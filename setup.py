import platform
from pathlib import Path
import subprocess
from setuptools import Extension, setup

from setuptools.command.build_ext import build_ext #import SubCommand

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

        # rename here the compiled shared library
        _builds = list(Path(self.build_lib).iterdir())
        assert len(_builds) == 1
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
