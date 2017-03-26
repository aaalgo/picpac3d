from distutils.core import setup, Extension

picpac3d = Extension('picpac3d',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        libraries = ['opencv_highgui', 'opencv_core', 'boost_filesystem', 'boost_system', 'boost_python', 'glog', 'x265', 'de265', 'glfw', 'GLEW', 'GL'],
        include_dirs = ['/usr/local/include', 'picpac', 'picpac/json11'],
        library_dirs = ['/usr/local/lib'],
        sources = ['picpac3d.cpp', 'picpac/picpac.cpp', 'picpac/picpac-cv.cpp', 'picpac/json11/json11.cpp'],
        depends = ['picpac/json11/json11.hpp', 'picpac/picpac.h', 'picpac/picpac-cv.h']
        )

setup (name = 'cubic',
       version = '0.0.1',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'LGPL',
       description = 'This is a demo package',
       ext_modules = [picpac3d],
       )
