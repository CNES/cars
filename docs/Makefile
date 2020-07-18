# Quick Makefile for pdf generation (before sphinx installation)
# need pip install rst2pdf
.PHONY: all clean

all: cli_usage.pdf  cnes_cluster_install.pdf  generalities.pdf  install.pdf  notebooks.pdf

%.pdf: %.rst
	rst2pdf -o $@ $<
clean:
	rm -f ./*.pdf

