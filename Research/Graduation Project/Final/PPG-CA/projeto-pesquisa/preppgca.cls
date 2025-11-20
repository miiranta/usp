%% =============================================================================
%% ppgca.cls
%%
%% Copyright 2018-2020 Adriano J. Holanda, DCM/FFCLRP/USP
%%
%% This software is distributed under MIT License
%% https://opensource.org/licenses/MIT
%% =============================================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{preppgca}[2018/06/12 class to DCM/FFCLRP/USP research project template]

% --- Class structure: declaration of options part
% ---
% This class extends the abntex2 class
% Read all the documentclass options; pass them to abntex2,
% unless the file "<currentoption>.preppgca" exists, then it is loaded
\DeclareOption*{\InputIfFileExists{\CurrentOption.preppgca}{}{
    %
    \PassOptionsToClass{\CurrentOption}{abntex2}}}

% --- Class structure: execution of options part
% ---
\ProcessOptions \relax
% --- Class structure: declaration of options part
% ---

\def\@@ptsize{12pt}
\DeclareOption{10pt}{\def\@@ptsize{10pt}}
\DeclareOption{11pt}{\def\@@ptsize{11pt}}
\DeclareOption{12pt}{\def\@@ptsize{12pt}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{abnt2}}
\ProcessOptions\relax
\LoadClass[article,a4paper,\@@ptsize]{abntex2} % extends ABNTEX class

%%%%%%%%%%%%%%%% SECTIONS
\RequirePackage{titlesec}
\titleformat{\section}
  {\normalfont\Large\bfseries}
  {\thesection}{1em}{}
\titleformat{\subsection}
  {\normalfont\large\bfseries}
  {\thesection}{1em}{}

%%%%%%%%%%%%%%%% BIBLIOGRAPHY
\newcommand\mybibname{Refer\^{e}ncias}
\renewcommand\bibsection{%
  \begin{center}
    \noindent\normalsize\large\bfseries{\mybibname}%
 \end{center}
}

%%%%%%%%%%%%%%%% RESUMO font
\renewenvironment{abstract}{\subsection*{\hfil\abstractname}}{}

%%%%%%%%%%%%%%%% SWITCH XETEX AND TEX
\RequirePackage{iftex}                    % test if xetex engine is used
\ifXeTeX % XeTeX
  \RequirePackage{fontspec}
  \RequirePackage{polyglossia}
  \setdefaultlanguage{brazil}
  \defaultfontfeatures{Ligatures={TeX}}
\else % LaTeX, pdfLaTeX
     \RequirePackage[utf8]{inputenc}
     \RequirePackage[brazil]{babel}
     \RequirePackage[T1]{fontenc}
\fi

%%%%%%%%%%%%%%%% or USE LUATEX
\ifLuaTeX % LuaTeX
  \RequirePackage{fontspec}
  \RequirePackage{polyglossia}
  \setdefaultlanguage{brazil}
  \defaultfontfeatures{Ligatures={TeX}}
\fi

\RequirePackage{lmodern}			% Usa a fonte Latin Modern
\RequirePackage{indentfirst}		% Indenta o primeiro parágrafo de cada seção.
\RequirePackage{nomencl}                        % Lista de simbolos
\RequirePackage{color}				% Controle das cores
\RequirePackage{graphicx}			% Inclusão de gráficos
\RequirePackage{multirow}
\RequirePackage[alf]{abntex2cite}               % Reference style
\RequirePackage{setspace}                       % to set space between lines
\RequirePackage{xcolor}

%%%%%%%%%%%%%%%% TITULO

\def\@titulo{\empty}
\renewcommand{\titulo}[1]{\def\@titulo{#1}}
\newcommand{\show@titulo}{\@titulo}

\def\@aluno{\empty}
\newcommand{\aluno}[1]{\def\@aluno{#1}}
\newcommand{\show@aluno}{\@aluno}

\def\@emailaluno{\empty}
\newcommand{\emailaluno}[1]{\def\@emailaluno{#1}}
\newcommand{\show@emailaluno}{\@emailaluno}

\def\@orientador{\empty}
\renewcommand{\orientador}[1]{\def\@orientador{#1}}
\newcommand{\show@orientador}{Orientador: \@orientador}

\def\@emailorientador{\empty}
\newcommand{\emailorientador}[1]{\def\@emailorientador{#1}}
\newcommand{\show@emailorientador}{\@emailorientador}

\def\@data{\empty}
\renewcommand{\data}[1]{\def\@data{#1}}
\newcommand{\show@data}{\@data}

\def\commonmaketitle{
    {\LARGE\show@titulo}\\
    {\large\show@aluno}\footnote{\show@emailaluno}\\
    {\large\show@orientador}\footnote{\show@emailorientador}\\
    {\large\show@data}\\
}

\makeatletter
\def\maketitle{%
  \begin{center}
    \commonmaketitle
    \normalfont
    \vspace{.6785cm}
      \begin{minipage}[h]{.13\linewidth}
      \end{minipage}
      \begin{minipage}[h]{.53\linewidth}
        \scriptsize\parskip0.1em\color{darkgray} Proposta de projeto de
        pesquisa a ser apresentada junto ao Departamento de Computação e
        Matemática da \hbox{FFCLRP/USP} como requisito para participação
        no processo seletivo do Programa de Pós-Graduação
        em Computação Aplicada.
      \end{minipage}
      \begin{minipage}[h]{.13\linewidth}
      \end{minipage}
    \end{center}
  }
\makeatother

\newcommand{\PREPPGCAtipodocumento}[1]{

  \ifthenelse{\equal{#1}{projeto}}{
    \makeatletter
    \def\maketitle{%
      \begin{center}
        \commonmaketitle\smallskip
        {\footnotesize Programa de Pós-Graduação em Computação Aplicada\par
         Departamento de Computação e Matemática\par
         Faculdade de Filosofia, Ciência e Letras de Ribeirão Preto--USP}
      \end{center}
    }
    \makeatother
  }{\empty}
}

\definecolor{blue}{RGB}{41,5,195}
% PDF info
\makeatletter
\hypersetup{
        %pagebackref=true,
                pdftitle={\show@titulo},
                pdfauthor={\show@aluno},
                pdfsubject={\show@titulo},
                pdfcreator={LaTeX with preppgca},
                pdfkeywords={}{},
                colorlinks=true, % false: boxed links; true: colored links
                linkcolor=blue, % color of internal links
                citecolor=blue, % color of links to bibliography
                filecolor=magenta, % color of file links
                urlcolor=blue,
                bookmarksdepth=4
}
\makeatother
