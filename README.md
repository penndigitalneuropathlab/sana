# Semi Automatic Neuropath Analysis (SANA)

## Installation
### Python
* Install [Python 3.9](https://www.python.org/downloads/release/python-390/)
  * IMPORTANT: Make sure to check the advanced options and include Python in PATH environment variables
  * You can test if python is installed properly by opening a command line and typing `python3 -V`
### OpenSlide
* Download [OpenSlide Windows binaries](https://github.com/openslide/openslide-winbuild/releases/download/v20230414/openslide-win64-20230414.zip)
* Unzip, and place in a location such as C:\Users\yourname\openslide
* Add this location to your PATH environment variable
  * Search environment variables
  * Edit PATH variable under User environment
  * Append the C:\Users\yourname\openslide location to the end
### pip packages
* Copy/paste each of the following lines into your command line, be sure that each of them succeeds
  * python3 -m pip install scipy
  * python3 -m pip install pytorch
  * python3 -m pip install tqdm
  * python3 -m pip install xlrd
  * python3 -m pip install numba
  * python3 -m pip install shapely
  * python3 -m pip install matplotlib
  * python3 -m pip install pyefd
* If any of these fail, feel free to google the error message, or reach out to Noah
### SANA code
* Following steps to create an [ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) after making your github account
* Open command line and Change Directory (cd command) into a location to store the code
  * `cd ~/`
* `git clone git@github.com:penndigitalneuropathlab/sana.git`
* `cd ~/sana/scripts`
* `python ./sana_process.py`
  * If everything installed properly, you should get a USAGE message
  * If you get a package not found, you can try running `python3 -m pip install PACKAGENAME` or google "how to pip install PACKAGENAME"
  * Any other errors reach out to Noah
