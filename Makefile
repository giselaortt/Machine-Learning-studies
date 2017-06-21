parsefile:
	python parse_arq.py

mlp:
	cc -c  mlp.c

main:
	cc main.c mlp.o -o ex -lm

all: parsefile mlp main
