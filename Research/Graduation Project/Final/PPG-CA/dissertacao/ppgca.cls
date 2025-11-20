%% =============================================================================
%% ppgca.cls
%%
%% Copyright 2017-2020 Adriano J. Holanda, DCM/FFCLRP/USP
%%
%% This software is distributed under MIT License
%% https://opensource.org/licenses/MIT
%% =============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ppgca}[2020/06/10 v2.5 (unreleased), class to DCM/FFCLRP/USP graduate manuscript template]
%
%
% Class option(s) implemented only in ppgca
%
%
\newif\if@englishwr
 \@englishwrfalse
 \DeclareOption{englishwr}{\@englishwrtrue}
%--- Class structure: execution of options part
% ---

\def\@@ptsize{12pt}
\DeclareOption{10pt}{\def\@@ptsize{10pt}}
\DeclareOption{11pt}{\def\@@ptsize{11pt}}
\DeclareOption{12pt}{\def\@@ptsize{12pt}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{abntex2}}
\ProcessOptions\relax
\LoadClass[a4paper,\@@ptsize]{abntex2} % extends ABNTEX class

%
% Document language
%
\providecommand{\documentlanguage}{}
\if@englishwr
\renewcommand\documentlanguage{english}
\else
\renewcommand\documentlanguage{brazil}
\fi

%
%
% Fonts
%
%
\renewcommand{\familydefault}{\rmdefault}
\renewcommand\ABNTEXchapterfont{\rmfamily}
\newcommand{\chapterfontsize}{\Huge}
\newcommand{\chapterfont}{\normalfont\chapterfontsize\bfseries}

%
%
% Declarations used by ppgca
%
%
% Public
\newcommand{\thetitle}{\@title}
\newcommand{\theauthor}{\@author}

%
%
% Packages
%
%
\RequirePackage{lmodern}                  % Usa a fonte Latin Modern
\RequirePackage{ifxetex,ifluatex}
\RequirePackage{lastpage}                 % Usado pela Ficha catalografica
%\RequirePackage{fancyhdr}

% We define a new if to test xetex OR luatex engines logical condition.
\newif\ifxetexorluatex
\ifxetex
 \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
    \fi
  \fi

\ifxetexorluatex
  \RequirePackage{fontspec}
  \RequirePackage{polyglossia}
  \setdefaultlanguage{\documentlanguage}
  \defaultfontfeatures{Ligatures={TeX}}
  \else % LaTeX, pdfLaTeX
      \RequirePackage[T1]{fontenc}
      \RequirePackage[utf8]{inputenc}
      % babel is already loaded by ABNTeX2 class
      % ti's only needed to select the english language
      % if englishwr is passed as argument to \documentclass
      \ifthenelse{\equal{\documentlanguage}{english}}{\selectlanguage{english}}{}
\fi

\RequirePackage{hyperref}                       % controla a formação do índice
\RequirePackage{parskip}			% espaçamento entre os parágrafos
\RequirePackage{microtype}                      % para melhorias de justificação
\RequirePackage{morefloats}			% permite mais floats
\RequirePackage{graphicx}
\RequirePackage{ifthen}

% MATH PACKS
\RequirePackage{amsfonts}                       % AMS Fonts
\RequirePackage{amsmath}                        % AMS Math Symbols
\RequirePackage{amssymb}                        % AMS Symbols

% ALGORITHMS and CODES
\RequirePackage{algorithmicx}
\RequirePackage{listings}

%
%
% Definitions
%
%

%%%%%%%%%%%%%%%% DEFINE BLANK HEADER
\newcommand{\ppgcablankheader}[1]{
  \addtocounter{abntex@bookmarkcounter}{1}%
   \begin{center}
  {\PRIVATEbookmarkthis{#1}%
    \thispagestyle{empty}
    {\chapterfont\noindent#1}}
  \end{center}
}

%%%%%%%%%%%%%%%% FOR FANCIER CHAPTER TITLE
\makechapterstyle{ppgca@chaptertitlestyle}{
  %%
  % 2020-02-26-ajh: the next uncommented line was added to solve the problem
  % of replicated title after the chapter/section is written.
  % https://tex.stackexchange.com/questions/317937/section-name-appearing-twice?rq=1
  \renewcommand*{\chaptermark}[1]{\markboth{}{}}
  \renewcommand*{\sectionmark}[1]{}
  %%
  \renewcommand*{\printchaptername}{}
  \renewcommand*{\chapnumfont}{\ABNTEXchapterfont\fontsize{80}{110}\selectfont\color{thegrey}}
  \renewcommand*{\chapnumfont}{\normalfont\fontsize{80}{110}\selectfont\color{thegrey}}
  \renewcommand*{\printchapternum}{{\hfill\chapnumfont\thechapter}\par\nobreak
    \tocprintchapter
  }

  \renewcommand{\afterchapternum}{}

  \renewcommand*{\chaptitlefont}{\chapterfont}
  \renewcommand*{\printchaptertitle}[1]{{\thispagestyle{empty}\hrulefill\par\medskip\chaptitlefont##1\hfill}}

  % impressao do capitulo nao numerado
  \renewcommand{\printchapternonum}{%
    % print only if it is not pretextual
    \ifthenelse{\boolean{ispretextual}}{
      \empty
    }{
      \tocprintchapternonum
      \setboolean{abntex@innonumchapter}{true}\nobreak
    }
  }
}

%
% SECTIONS
%

\RequirePackage{titlesec}
% %AJH% Do not assign chapter settings here.
  \titleformat{\section}
              {\normalfont\LARGE\bfseries}
              {\thesection}{1em}{}[]
              \titleformat{\subsection}
                          {\normalfont\Large\bfseries}
                          {\thesubsection}{1em}{}
                          \titleformat{\subsubsection}
                                      {\normalfont\large\bfseries}
                                      {\thesubsubsection}{1em}{}

% RESUMO
\renewenvironment{abstract}{\subsection*{\chapterfont\hfil\abstractname}}{}

%%%%%%%%%%%%%%% REDEFINE PRETEXTUAL CHAPTER TITLE
\renewcommand{\pretextualchapter}{\ppgcablankheader}

%%%%%%%%%%%%%%%% REMOVE HLINE FROM PRETEXTUAL CHAPTER TITLE
\newboolean{ispretextual}
\let\PT\pretextual
\renewcommand{\pretextual}{\PT\setboolean{ispretextual}{true}}
 \newboolean{istextual}
 \let\TE\textual
 \renewcommand{\textual}{
   \TE%
   \setboolean{istextual}{true}%
   \setboolean{ispretextual}{false}%
   % ENABLE FANCIER CHAPTER TITLE
   \chapterstyle{ppgca@chaptertitlestyle}
 }
 \newboolean{ispostextual}
 \let\PO\pretextual
 \renewcommand{\postextual}{
   \PO%
   \setboolean{ispostextual}{true}%
   \setboolean{istextual}{false}%
 }
 \AtBeginDocument{ % initialization
   \setboolean{ispretextual}{true}
   \setboolean{istextual}{false}
   \setboolean{ispostextual}{false}
 }

 %
 % Translations
 %

 % Babel e ajustes
 \addto\captionsbrazil{
      %% ajusta nomes padroes do babel
      \renewcommand{\contentsname}{\chapterfont Sum\'{a}rio}
      \renewcommand{\bibname}{\chapterfont Refer\^encias}
      \renewcommand{\indexname}{\chapterfont \'Indice}
      \renewcommand{\listfigurename}{\chapterfont Lista de figuras}
      \renewcommand{\listtablename}{\chapterfont Lista de tabelas}
      %% ajusta nomes usados com a macro \autoref
      \renewcommand{\pageautorefname}{p\'agina}
      \renewcommand{\sectionautorefname}{se{\c c}\~ao}
      \renewcommand{\subsectionautorefname}{subse{\c c}\~ao}
      \renewcommand{\paragraphautorefname}{par\'agrafo}
      \renewcommand{\subsubsectionautorefname}{subsubse{\c c}\~ao}
      \renewcommand{\paragraphautorefname}{subse{\c c}\~ao}
    }

    \addto\captionsenglish{
      % adjusts to format
      \renewcommand{\contentsname}{\chapterfont Summary}
      \renewcommand{\bibname}{\chapterfont References}
      \renewcommand{\indexname}{\chapterfont Index}
      \renewcommand{\listfigurename}{\chapterfont List of figures}
      \renewcommand{\listtablename}{\chapterfont List of tables}
      \renewcommand{\listadesiglasname}{\chapterfont List of abbreviations and acronyms}
      \renewcommand{\listadesimbolosname}{\chapterfont List of symbols}
      %% ajusta nomes padroes do babel
      \renewcommand{\folhaderostoname}{Title page}
      \renewcommand{\epigraphname}{Epigraph}
      \renewcommand{\dedicatorianame}{Dedication}
      \renewcommand{\errataname}{Errata sheet}
      \renewcommand{\agradecimentosname}{Acknowledgements}
      \renewcommand{\anexoname}{ANNEX}
      \renewcommand{\anexosname}{Annex}
      \renewcommand{\apendicename}{APPENDIX}
      \renewcommand{\apendicesname}{Appendix}
      \renewcommand{\orientadorname}{Supervisor:}
      \renewcommand{\coorientadorname}{Co-supervisor:}
      \renewcommand{\folhadeaprovacaoname}{Approval}
      \renewcommand{\resumoname}{Abstract}
      \renewcommand{\fontename}{Source}
      \renewcommand{\notaname}{Note}
      %% adjusts names used by \autoref
      \renewcommand{\pageautorefname}{page}
      \renewcommand{\sectionautorefname}{section}
      \renewcommand{\subsectionautorefname}{subsection}
      \renewcommand{\subsubsectionautorefname}{subsubsection}
      \renewcommand{\paragraphautorefname}{subsubsubsection}
      \renewcommand{\paragraphautorefname}{paragraph}
    }


% BIBLIOGRAPHY: change to adjust to summary
\renewcommand\bibsection{%
  \chapter*{\bibname}%
  \bibmark
  \ifnobibintoc\else
  \phantomsection
  % align bibname in toc and change its size
  \addcontentsline{toc}{chapter}{\texorpdfstring{\MakeTextUppercase{\bibname}}{\bibname}}
  \fi
  \prebibhook
}


\RequirePackage{color}
\definecolor{thered}{rgb}{0.65,0.04,0.07}
\definecolor{thegreen}{rgb}{0.06,0.44,0.08}
\definecolor{thegrey}{gray}{0.5}
\definecolor{theshade}{rgb}{1,1,0.97}
\definecolor{theframe}{gray}{0.6}
\definecolor{blue}{RGB}{41,5,195}

%%%%%%%%%%%%%%%% CITACOES
\RequirePackage[alf]{abntex2cite}	% citacoes

%%%%%%%%%%%%%%%%
% ---
% Posiciona figuras e tabelas no topo da página quando adicionadas sozinhas
% em um página em branco. Ver https://github.com/abntex/abntex2/issues/170
\makeatletter
\setlength{\@fptop}{5pt} % Set distance from top of page to first float
\makeatother
% ---

% ---
% Possibilita criação de Quadros e Lista de quadros.
% Ver https://github.com/abntex/abntex2/issues/176
%
\newcommand{\quadroname}{Quadro}
\newcommand{\listofquadrosname}{\hfil Lista de quadros}

\newfloat[chapter]{quadro}{loq}{\quadroname}
\newlistof{listofquadros}{loq}{\listofquadrosname}
\newlistentry{quadro}{loq}{0}

% configurações para atender às regras da ABNT
\setfloatadjustment{quadro}{\centering}
\counterwithout{quadro}{chapter}
\renewcommand{\cftquadroname}{\quadroname\space}
\renewcommand*{\cftquadroaftersnum}{\hfill--\hfill}
\setfloatlocations{quadro}{hbtp} % Ver https://github.com/abntex/abntex2/issues/176
% ---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All redefinitions are made to remove blank page after environment use.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% REDEFINE \PRIVATEclearpageifneeded
\renewcommand{\PRIVATEclearpageifneeded}{%
  \ifthenelse{\equal{\ABNTEXisarticle}{true}}{%
  }{ % else
    \cleardoublepage
  }
}

%
% ATTRIBUTES
%
% FMT == FORMAT
%
\makeatletter
\newcommand{\fmt@descricao}{
    \begin{flushright}
    \begin{minipage}[h]{.5\linewidth}
      {\ }
    \end{minipage}
    \begin{minipage}[h]{.5\linewidth}\normalsize
    \end{minipage}
  \end{flushright}
}

\newcommand{\fmt@city}{
  \normalfont\normalsize\imprimirlocal
  \vspace{.1cm}
}

\newcommand{\fmt@date}{
  \normalfont\normalsize\imprimirdata
}
\makeatother

%
% Descrição do documento na folha de rosto
%
\makeatletter
\if@englishwr
\def\@institution{University de São Paulo}
\def\@faculty{Faculty of Sciences and Letters of Ribeir\~{a}o Preto}
\def\@department{Department of Computing and Mathematics}
\def\@program{Graduate Program in Applied Computing}
\else
\def\@institution{Universidade de S\~{a}o Paulo}
\def\@faculty{Faculdade de Filosofia, Ci\^{e}ncias e Letras de Ribeir\~{a}o Preto}
\def\@department{Departamento de Computa\c{c}\~{a}o e Matem\'{a}tica}
\def\@program{Programa de P\'os-Gradua\c{c}\~ao em Computa\c{c}\~{a}o Aplicada}
\fi

\newcommand{\fmt@affiliation}{
  \normalfont\small
  \MakeUppercase{\@institution}\\
  \MakeUppercase{\@faculty}\\
  \MakeUppercase{\@department}
}
\makeatother


%
% REDEFINE CAPA
%
\makeatletter
\def\@@fmtauthor{\normalfont\Large}
\def\@@fmttitle{\normalfont\Large}
\renewenvironment{capa}{}{\cleardoublepage}
\renewcommand{\imprimircapa}{%
  \begin{capa}%
    \center

    \begingroup
    \fmt@affiliation
    \endgroup

    \vfill

    {\normalfont\normalsize\MakeUppercase{\@author}}

    \vfill

    {\normalfont\Large\textbf{\@title}}

    \vfill

    \fmt@city

    \fmt@date
    \vspace*{.4cm}
  \end{capa}
}
\makeatother


%
% REDEFINE FOLHA DE ROSTO
%
%
\makeatletter
\def\@ENsupervisorlabel{Supervisor}
\def\@supervisorlabel{Orientador}
\if@englishwr
\def\@supervisorlabel{\@ENsupervisorlabel}
\fi
\newcommand\thesupervisorlabel{\@supervisorlabel}

%
% DOCUMENT TYPE: Dissertação ou Tese
% muda para o inglês de acordo com a escolha
%
\providecommand{\documenttype}{}
\ifthenelse{\equal{\imprimirtipotrabalho}{Tese}}{
  \if@englishwr
  \renewcommand\documenttype{Thesis}
  \newcommand\@ENdocumenttype{\documenttype}
  \else
  \renewcommand\documenttype{Tese}
  \newcommand\@ENdocumenttype{Thesis}
  \fi
}{
  \if@englishwr
  \renewcommand\documenttype{Dissertation}
  \newcommand\@ENdocumenttype{\documenttype}
  \else
  \renewcommand\documenttype{Disserta\c{c}\~{a}o}
  \newcommand\@ENdocumenttype{Dissertation}
  \fi
}

% COMMAND TO SELECT THE VERSION OF DOCUMENT: ORIGINAL \/ CORRIGIDA
\def\@@fmt{\normalsize}

\def\@ENtextoversaodocumento{Original Version}
\def\@textoversaodocumento{ % default: Original, option: Corrigida
  \@@fmt
  \if@englishwr
  \@ENtextoversaodocumento
  \else
  Versão Original
  \fi
}

\newcommand{\versaodocumento}[1] {
  \def\@PPGCAversaodocumento{#1}
  \ifthenelse{\equal{\@PPGCAversaodocumento}{corrigida}}{
    % TRUE
    \if@englishwr
    \def\@textoversaodocumento{\@@fmt Corrected Version\\\smallskip
      The original version is found at FFCLRP/USP.}
    \else
    \def\@textoversaodocumento{\@@fmt Versão Corrigida\\\smallskip
      Versão original encontra-se na FFCLRP/USP.}
    \def\@ENtextoversaodocumento{\@@fmt Corrected Version\\\smallskip
      The original version is found at FFCLRP/USP.}
   \fi
  }{% FALSE
    \@textoversaodocumento
  }
}
  \newcommand{\supervisor} {
\if@englishwr
    \def\@@cosup{Cosupervisor}
\else
    \def\@@cosup{Coorientador}
\fi

  {\normalsize\thesupervisorlabel:~\imprimirorientador\par}
    \abntex@ifnotempty{\imprimircoorientador}{\normalsize\@@cosup:~\imprimircoorientador}
}

\def\@ENdescription{
  \@ENdocumenttype~presented to \@faculty~(FFCLRP) from
  the \@institution~(USP),
  as part of the requirements to hold the Master of
  Science degree.\par

  \vspace{.7cm}
  Field of Study: Applied Computing.\par

  \vspace{.5cm}
  \supervisor
}

\def\description{
  \if@englishwr
  \@ENdescription{}

\else
 \documenttype~apresentada à
  \@faculty~(FFCLRP) da \@institution~(USP),
  como parte das exigências
  para a obtenção do título de
  Mestre em Ciências.\par

  \vspace{.7cm}
  \noindent Área de Concentração: Computação Aplicada.\par

  \vspace{.5cm}
  \supervisor
\fi
}

\newcommand{\fmt@description}{
    \begin{flushright}
    \begin{minipage}[h]{.3\linewidth}
      {\ }
    \end{minipage}
    \begin{minipage}[h]{.7\linewidth}
      \normalsize
      \description
    \end{minipage}
  \end{flushright}
}

% Redefine to adjust dimensions
\def\@@fmtauthor{\normalfont\normalsize}
\def\@@fmttitle{\normalfont\large}
\renewenvironment{folhaderosto}[1][\folhaderostoname]{\PRIVATEbookmarkthis{#1}}{\PRIVATEclearpageifneeded}
\renewenvironment{folhaderosto*}[1][\folhaderostoname]{\PRIVATEbookmarkthis{#1}}{\newpage}%
\renewcommand{\folhaderostocontent}{
  \begin{center}

    \@@fmtauthor{\MakeUppercase{\@author}}

    \vfill

    \@@fmttitle{\textbf{\@title}}

    \vfill

    \begingroup
    \@textoversaodocumento
    \endgroup % Original ou Corrigida

    \vfill

    \fmt@description

    \vfill

    \fmt@city

    \fmt@date

    \vspace*{1cm}

  \end{center}
}

%---
% COVER SHEET
%---
\if@englishwr
\newcommand{\coversheet}[1]{}
\else
\newcommand{\coversheet}[1] {
  \begin{center}

    \@@fmtauthor{\MakeUppercase{\@author}}

    \vfill

    \@@fmttitle{\textbf{#1}}

    \vfill

    \begingroup
    \@ENtextoversaodocumento
    \endgroup % Original ou Corrigida

    \vfill

    \begin{flushright}
      \begin{minipage}[h]{.3\linewidth}
        {\ }
      \end{minipage}
      \begin{minipage}[h]{.7\linewidth}
        \normalsize
        \@ENdocumenttype~presented to \@faculty~(FFCLRP) from
        the \@institution~(USP),
        as part of the requirements to hold the Master of
        Science degree.\par

        \vspace{.7cm}
        Field of Study: Applied Computing.\par

        \vspace{.5cm}
        \@ENsupervisorlabel: \imprimirorientador
    \end{minipage}
    \end{flushright}

    \vfill

    \fmt@city

    \fmt@date

    \vspace*{1cm}

  \end{center}
  \cleardoublepage
}
\fi
\makeatother

% ---
% Configurações de aparência do PDF final
% alterando o aspecto da cor azul
\definecolor{blue}{RGB}{41,5,195}

% informações do PDF

\hypersetup{
  % pagebackref=true,
  pdftitle={\imprimirtitulo},
  pdfauthor={\theauthor},
  pdfsubject={\imprimirpreambulo},
  pdfcreator={LaTeX with abnTeX2.PPGCA},
  pdfkeywords={ABNT}{LaTeX}{abntex2}{dissertation}{PPGCA}{USP},
  colorlinks=true,              % false: boxed links; true: colored links
  linkcolor=black,              % color of internal links
  citecolor=black,              % color of links to bibliography
  filecolor=magenta,            % color of file links
  urlcolor=black,
  bookmarksdepth=4
}

\endinput
