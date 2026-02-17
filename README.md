# FIR Filter Design – Window Method 📐

An interactive Streamlit app for learning FIR filter design via the windowing method.

## Features

- **5 Window Functions**: Rectangular, Hann, Hamming, Blackman, Kaiser (with β control)
- **4 Filter Types**: Lowpass, Highpass, Bandpass, Bandstop
- **LaTeX Formulas**: Full mathematical expressions for every window and ideal filter
- **Interactive Plots**: Plotly-based time-domain and frequency-domain visualisation
- **Compare Mode**: Overlay all windows side-by-side
- **Design Explorer**: Design from specs, what-if analysis, and self-check quiz

## Local Setup (using uv)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# Clone or cd into this directory
cd <Code Directory>

# Create virtual environment and install dependencies
uv sync

# Run the app
uv run streamlit run fir_filter_app.py
```

The app will open at `http://localhost:8501`.


## Project Structure

```
Codes/
├── fir_filter_app.py    # Main Streamlit application
├── pyproject.toml       # uv project definition
├── requirements.txt     # Dependencies (for Streamlit Cloud)
└── README.md            # This file
```

## License

For educational use – Mahesh Panicker (mahesh.signalproc@gmail.com)
