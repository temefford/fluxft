import os

def suppress_xformers_warning():
    os.environ['XFORMERS_MORE_DETAILS'] = '0'
    # Patch xformers warning if present
    import warnings
    warnings.filterwarnings('ignore', message='xFormers can\'t load C\+\+/CUDA extensions')
    warnings.filterwarnings('ignore', message='Memory-efficient attention, SwiGLU, sparse and more won\'t be available.')

if __name__ == '__main__':
    suppress_xformers_warning()
    print('xFormers warnings suppressed.')
