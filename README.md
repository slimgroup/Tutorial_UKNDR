# Tutorial_UKNDR
Tutorial repository for working with UK National Data Repository (UKNDR) datasets, including seismic data, checkshot files, and velocity model processing.

## 📌 Overview

This repository contains scripts and workflows for:

- Loading LAS files
- Working with SEG-Y seismic datasets
- Training data preparation in HDF5 file formats
- Visualization and analysis

## Installation

### Clone the repository

```bash
git clone https://github.com/slimgroup/Tutorial_UKNDR.git
cd Tutorial_UKNDR

### Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt

## Data Policy

Large seismic files (SEG-Y, HDF5, etc.) are not included in this repository.

Please store raw data separately.

This repository is intended for tutorial and research purposes related to:

- Seismic data curation
- Well-log processing
- Machine learning workflows in geophysics
