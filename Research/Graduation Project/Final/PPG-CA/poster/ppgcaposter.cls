%% =============================================================================
%% ppgcaposter.cls
%%
%% Copyright 2017-2020 Adriano J. Holanda, DCM/FFCLRP/USP
%%
%% This software is distributed under MIT License
%% https://opensource.org/licenses/MIT
%% =============================================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ppgcaposter}[2017/11/27 class to DCM/FFCLRP/USP graduate manuscript template]

% --- Class structure: declaration of options part
% ---
% This class extends the a0poster class
% Read all the documentclass options; pass them to a0poster,
% unless the file "<currentoption>.ppgcaposter" exists, then it is loaded
\DeclareOption*{\InputIfFileExists{\CurrentOption.ppgcaposter}{}{
    %
    \PassOptionsToClass{\CurrentOption}{a0poster}}}

% --- Class structure: execution of options part
% ---
\ProcessOptions \relax
% --- Class structure: declaration of options part
% ---
\LoadClass{a0poster}

\ExecuteOptions{final}
\ProcessOptions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PACKAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{iftex}                    % test if xetex engine is used

\ifXeTeX % XeTeX
  \RequirePackage{fontspec}
  \RequirePackage{polyglossia}
  \setdefaultlanguage{brazil}
\else % LaTeX, pdfLaTeX
     \RequirePackage[utf8]{inputenc}
     \RequirePackage[brazil]{babel}
\fi

\ifLuaTeX % LuaTeX
  \RequirePackage{fontspec}
  \RequirePackage{polyglossia}
  \setdefaultlanguage{brazil}
\fi

\RequirePackage{multicol} % This is so we can have multiple columns of text side-by-side
\columnsep=100pt % This is the amount of white space between the columns in the poster
\columnseprule=3pt % This is the thickness of the black line between the columns in the poster
\RequirePackage{babelbib}

\RequirePackage[svgnames]{xcolor}
\RequirePackage{amsmath}
\RequirePackage{graphicx} % Required for including images
\RequirePackage{booktabs} % Top and bottom rules for table
\RequirePackage[font=small,labelfont=bf]{caption} % Required for specifying captions to tables and figures
\RequirePackage{amsfonts, amsmath, amsthm, amssymb} % For math fonts, symbols and environments
\RequirePackage{wrapfig} % Allows wrapping text around tables and figures
\RequirePackage{tikz}
\usetikzlibrary{calc}
\usetikzlibrary{fadings}
\RequirePackage{caption}
\RequirePackage{etoolbox}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FONTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% size 24.88pt
\renewcommand{\tiny}{\fontsize{12}{14}\selectfont}
\renewcommand{\scriptsize}{\fontsize{14.4}{18}\selectfont}
\renewcommand{\footnotesize}{\fontsize{17.28}{22}\selectfont}
\renewcommand{\small}{\fontsize{20.74}{25}\selectfont}
\renewcommand{\normalsize}{\fontsize{24.88}{30}\selectfont}
\renewcommand{\large}{\fontsize{29.86}{37}\selectfont}
\renewcommand{\Large}{\fontsize{35.83}{45}\selectfont}
\renewcommand{\LARGE}{\fontsize{43}{54}\selectfont}
\renewcommand{\huge}{\fontsize{51.6}{64}\selectfont}
\renewcommand{\Huge}{\fontsize{61.92}{77}\selectfont}
\renewcommand{\veryHuge}{\fontsize{74.3}{93}\selectfont}
\renewcommand{\VeryHuge}{\fontsize{89.16}{112}\selectfont}
\renewcommand{\VERYHuge}{\fontsize{107}{134}\selectfont}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MARGINS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A0 dimensions 841 x 1189 mm  33.1 x 46.8 in
\special{papersize=46.8in,33.1in}
\setlength{\paperwidth}{84.1cm}
\setlength{\paperheight}{118.9cm}
\setlength{\textwidth}{80.1cm}
\setlength{\textheight}{108.9cm}

\topskip0pt
\setlength{\headheight}{0pt}
\setlength{\headsep}{0pt}
\setlength{\topmargin}{0pt}
\setlength{\oddsidemargin}{0pt}

\setlength{\oddsidemargin}{3cm}

%\setlength{\columnsep}{2cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\printheader{\topskip4cm
\begin{minipage}[ht]{0.15\linewidth}
  \includegraphics[scale=.3]{usp.pdf}
\end{minipage}\vspace{2cm}
\begin{minipage}[ht]{0.75\linewidth}\bf
  {\large Programa de Pós-Graduação em Computação Aplicada}\\
  { Faculdade de Filosofia, Ciências e Letras de Ribeirão Preto}\\
   {\textsc Universidade de S\~{a}o Paulo}
\end{minipage}
}
