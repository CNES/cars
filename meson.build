project(
  'cars',
  'cpp',
  version: run_command(
    'python',
    meson.source_root() / 'version.py',
    'get-vcs',
  ).stdout().strip(),
  default_options: [
    'cpp_std=c++17',
  ],
)

meson.add_dist_script('python', meson.source_root() / 'version.py', 'set-dist', meson.project_version())

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')

install_subdir(
  'cars',
  install_dir: py.get_install_dir() / 'cars',
  strip_directory: true,
)

subdir(
  'cars/applications/dense_matching/cpp'
)

subdir(
  'cars/applications/dense_match_filling/cpp'
)
