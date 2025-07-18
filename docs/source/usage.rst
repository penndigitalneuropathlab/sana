****************
SANA Quick Start
****************

This quick start guide shows you how to install SANA and get started with quantifying histology images.

Prerequisites
=============
* >=Python3.9

Installation
============

Install using ``pip``
---------------------
Create a virtual environment which will contain SANA and the python dependencies::

    # change username to your username and/or preferred location
    mkdir -p /home/username/sana
    cd /home/username/sana
    python3 -m venv .venv

Activate the environment (this must be done once per terminal session)::

    cd /home/username/sana
    source .venv/bin/activate

Next, install SANA::
    
    python3 -m pip install pdnl_sana

Install using ``git``
---------------------
If a different version of SANA is desired, you can install SANA through git. First, clone the source code::

    git clone https://github.com/penndigitalneuropathlab/sana.git /home/username/sana

Optionally, switch to the desired branch::

    git checkout experimental

Create a shell script ``env.sh`` in the directory ``/home/username/sana`` with the contents below, modified to fit your installation

.. code-block:: Bash

    #!/bin/bash
    source .venv/bin/activate

    export PYTHONPATH=/home/username/sana/src:$PYTHONPATH

Create a virtual environment which will contain SANA and the python dependencies::

    # change username to your username and/or preferred location
    cd /home/username/sana
    python3 -m venv .venv

Activate the environment (this must be done once per terminal session)::
    cd /home/username/sana
    source env.sh

Next, install SANA dependencies::

    python3 -m pip install -r src/pdnl_sana/requirements.txt

Usage
=====
Run this command to ensure that everything installed properly::

    python3 -c "import pdnl_sana.image; import pdnl_sana.slide"

Troubleshooting
===============
If you are having issues with OpenSlide, follow instructions `here <https://openslide.org/api/python/#installing>`_ to get the binaries. 