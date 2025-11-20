NOW := $(shell date +%Y-%m-%d_%H-%M)

notes.org.gpg: notes.org
	-if [ -f $@ ]; then rm -i $@; fi
	-gpg -c $< \
	&& cp -v  $< /tmp/$(NOW)-$< \
	&& git commit -m "notes up to $(NOW)" $@ \
	&& git push && rm -i $<

notes.org:
	gpg notes.org.pgp

decript: notes.org
