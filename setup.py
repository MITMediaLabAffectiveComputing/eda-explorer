from distutils.core import setup
setup(
  name='eda-explorer',         # How you named your package folder (MyLib)
  packages=['eda-explorer'],   # Chose the same as "name"
  version='0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license
  description='Scripts to detect artifacts and in electrodermal activity (EDA) data',
  author='Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R.',  # Type in your name
  author_email='natashamjaques@gmail.com',      # Type in your E-Mail
  url='https://github.com/MITMediaLabAffectiveComputing/eda-explorer',
  download_url='https://github.com/MITMediaLabAffectiveComputing/eda-explorer/archive/v_01.tar.gz',
  keywords=['EDA', 'ARTIFACT', 'PEAK'],   # Keywords that define your package best
  install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'sklearn',
          'pickle',
          'PyWavelets',
          'scipy',
          'pprint',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the
                                            # current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      # Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)