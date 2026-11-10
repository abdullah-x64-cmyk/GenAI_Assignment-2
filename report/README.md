# IEEE technical report source

Main entry point: `main.tex`. `task1.tex` is a compatibility entry point that
includes the same paper. Compile from this directory with pdfLaTeX and BibTeX:

```
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, run `latexmk -pdf main.tex`, or upload the complete report-source
ZIP to a blank Overleaf project and select `main.tex` as its main document.
Use the pdfLaTeX compiler. The IEEEtran class is required and is included in
standard TeX Live publishers packages and Overleaf.

The report is compiled manually in Overleaf. The repository does not run an
automatic GitHub Actions report build. Compilation uses the checked-in figures
and does not retrain or rerun held-out evaluation.

The author details were supplied by the student. Research references were
checked against publisher pages, arXiv, and original model/dataset sources.
The previous draft's unrelated TrialGPT arXiv identifier and its incorrect
1,020-row dataset count were corrected. The executed release has 1,015 rows.

From the repository root, regenerate plots and the per-class table with:

```
python report/generate_figures.py
```

This reads only the saved JSON logs and the saved training archive. Matplotlib
and NumPy are the plotting dependencies. PDF figure versions are used by the
paper; PNG versions support inspection. Application screenshots were captured
from the actual running browser interface on 10 November 2026 using the same
synthetic note as the saved demonstration. The screenshot rerun shows CPU
times 21.27 s and 9.47 s; the earlier exported demonstration shows 10.70 s and
7.47 s. Both runs have the same labels and IDs. They are not pooled as a timing
benchmark, and the ECOG citation relevance error is preserved and discussed.

The paper distinguishes exact quotation from evidence relevance, reports
zero speculative acceptance without implying speedup, and records limitations
in baseline parsing, missing-information predictions, small patient groups,
truncation, software-version provenance, and incomplete token-level logs.
