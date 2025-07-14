PLANTUML_JAR = /home/cardoso42/plantuml/plantuml-1.2024.4.jar

# Arquivos .puml da seção SRS
SRS_PUMLS := $(wildcard docs/diagrams/*.puml)

diagrams:
	@echo "Gerando diagramas PlantUML..."
	java -jar $(PLANTUML_JAR) $(wildcard docs/diagrams/**/*.puml)

	@dirs=$$(find docs/diagrams -mindepth 1 -type d); \
	for d in $$dirs; do \
		pngs=$$(find $$d -maxdepth 1 -name '*.png'); \
		if [ -n "$$pngs" ]; then \
			dirname=$$(basename $$d); \
			mkdir -p docs/figures/$$dirname; \
			mv $$d/*.png docs/figures/$$dirname/; \
		fi; \
	done

hci_report: WORK_DIR=docs/hci_report
hci_report: latex-pdf clean

hci_slides: WORK_DIR=docs/hci_slides
hci_slides: latex-pdf clean

ip_report: WORK_DIR=docs/ip_report
ip_report: latex-pdf clean

ip_slides: WORK_DIR=docs/ip_slides
ip_slides: latex-pdf clean

latex-pdf:
	@echo "Compilando LaTeX para PDF..."
	cd $(WORK_DIR) && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	@echo "Removendo arquivos auxiliares do LaTeX em $(WORK_DIR)..."
	cd $(WORK_DIR) && rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg *.nav *.snm
