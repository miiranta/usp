pref := main
refs := refs.bib
cls := ppgca.cls

$(pref).pdf: $(pref).aux $(pref).bbl $(pref).idx $(cls)
	xelatex $(pref)

$(pref).aux: $(pref).tex
	xelatex $<

$(pref).bbl: refs.bib
	bibtex $(pref)

$(pref).idx:
	makeindex $(pref)

clean:
	$(RM) *.aux *.bbl *.blg *.brf *.idx *.ilg *.ind *.lof *.log *.lot *.ps *.toc \
		main.dvi main.pdf
