%option yylineno

%{
#include <stdlib.h>
#include "xyz.tab.h"
%}

%%
"main"              { yylval.s = strdup(yytext); return MAIN; }
"fn"                { return FN; }
"var"               { return VAR; }
"return"            { return RETURN; }
"if"                { return IF; }
"else"              { return ELSE; }
"while"             { return WHILE; }
"i64"               { return TYPE_INT; }
"f64"               { return TYPE_FLOAT; }
[a-zA-Z][a-zA-Z_]*  { yylval.s = strdup(yytext); return ID; }
[-]?[0-9]+          { return INT; }
[-]?[0-9]*\.[0-9]+  { return FLOAT; }
"("                 { return LPAR; }
")"                 { return RPAR; }
","                 { return COM; }
"{"                 { return LBRAC; }
"}"                 { return RBRAC; }
";"                 { return SCOL; }
":"                 { return COL; }
"="                 { return EQ; }
"++"                { return INCR; }
"--"                { return DECR; }
"-"                 { return MINUS; }
"!="                { return DIF; }
"!"                 { return EXCL; }
"+"                 { return PLUS; }
"*"                 { return ASTR; }
"/"                 { return SLSH; }
"%"                 { return PRCT; }
">="                { return GEQ; }
"<="                { return LEQ; }
">"                 { return GT; }
"<"                 { return LT; }
"&&"                { return AND; }
"||"                { return OR; }
[ \t\n]             { ; }
.                   { return *yytext; }

%%

int yywrap() {
    return 1;
}
