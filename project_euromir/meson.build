py.install_sources(
    '__init__.py',
    subdir: 'project_euromir')

py.extension_module(
    'project_euromir',
    'linear_algebra.c',
    install: true,
)

# Python unit tests
subdir('tests')
