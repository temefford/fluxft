import os

def accelerate_config_defaults():
    # Set environment variables to avoid accelerate config warning
    os.environ['ACCELERATE_USE_CPU'] = 'false'
    os.environ['ACCELERATE_LOG_LEVEL'] = 'WARNING'
    # Optionally set other defaults here

if __name__ == '__main__':
    accelerate_config_defaults()
    print('Accelerate config warnings suppressed.')
