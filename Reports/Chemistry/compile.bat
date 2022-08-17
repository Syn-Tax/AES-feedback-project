pandoc main.md --pdf-engine=pdflatex -o final.pdf --citeproc --csl american-chemical-society.csl --filter pandoc-eqnos --filter pandoc-fignos --filter pandoc-tablenos
