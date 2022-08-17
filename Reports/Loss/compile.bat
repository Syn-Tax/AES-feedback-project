pandoc main.md --pdf-engine=pdflatex -o final.pdf --citeproc --csl ieee.csl --filter pandoc-eqnos --filter pandoc-fignos --filter pandoc-tablenos
