pandoc main.md --pdf-engine=pdflatex -o final.pdf --bibliography ../bibliography.bib --csl ieee.csl --filter pandoc-eqnos --filter pandoc-fignos --filter pandoc-tablenos
