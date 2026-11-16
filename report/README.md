# Project report (LaTeX)

`main.tex` is the CS3103 PBL report. It is self-contained: all figures are in `figures/`.

Build with pdfLaTeX (run twice for the table of contents):

```
cd report
latexmk -pdf main.tex
```

or upload this folder to Overleaf with the pdfLaTeX compiler.

Figures are produced by `uv run --with matplotlib python scripts/make_figures.py` (written to `docs/figures/`, then copied here).
Layout: two-column article, 10pt. Body (Sections 1-6) is meant to stay within 6-7 pages; references and appendix are extra.
