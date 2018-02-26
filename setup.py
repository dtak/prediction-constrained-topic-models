from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    HAS_CYTHON = True
except ImportError:
    from distutils.command.build_ext import build_ext
    HAS_CYTHON = False

def make_cython_extension__nef_map_pi_d_K():
    ext = Extension(
        "pc_toolbox/model_slda/est_local_params__single_doc_map/calc_nef_map_pi_d_K__cython",
        ["pc_toolbox/model_slda/est_local_params__single_doc_map/calc_nef_map_pi_d_K__cython.pyx"],
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math"])
    return add_directives_to_cython_ext(ext)

def make_extensions():
    ''' Assemble C++/Cython extension objects for compilation.

    Warns user if required prerequisites are not specified.

    Returns
    -------
    ext_list : list of extension objects
    '''
    ext_list = list()
    if HAS_CYTHON:
        ext_list.append(make_cython_extension__nef_map_pi_d_K())
    return ext_list

def add_directives_to_cython_ext(ext):
    ''' Improve speed of cython code extensions

    References
    ----------
    http://docs.cython.org/src/reference/compilation.html#compiler-directives
    '''
    ext.cython_directives = {
        'embedsignature':True,
        'boundscheck':False,
        'nonecheck':False,
        'wraparound':False,
        'cdivision':True}
    return ext

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = make_extensions(),
    setup_requires=["Cython>=0.25"],
    )
