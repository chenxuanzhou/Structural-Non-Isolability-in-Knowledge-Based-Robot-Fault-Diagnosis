%% ===========================================================================
%% ifacconf.cls  --  LOCAL APPROXIMATION.  THIS IS NOT IFAC'S OFFICIAL CLASS.
%% ===========================================================================
%%
%% This file is a stand-in for IFAC's ifacconf.cls, written only so that the
%% SAFEPROCESS 2027 draft (safeprocess2027_final.tex) can be compiled and
%% page-counted on a machine that cannot reach the IFAC author kit.  It is
%% NOT the official class, it does not reproduce IFAC's typography (Times,
%% upper-case centred section heads, "Fig. 1." captions, starred author
%% marks, ...), and it must not be used for the camera-ready submission.
%% Replace it with the official ifacconf.cls from
%%   https://www.ifac-control.org/conferences/author-guide
%% and re-check the page count: the official geometry differs.
%%
%% What it does reproduce is the page layout of the reference build
%% safeprocess2027_final.pdf as of commit e446483 (6 pages, A4, built
%% 2026-09-03 with an earlier, uncommitted approximation of this class).  Every
%% dimension below was measured from that PDF; with the safeprocess2027_final.tex
%% of the same commit the rebuilt PDF is glyph-for-glyph identical in position
%% (<= 0.002 bp) on all six pages, so an edited source can be page-counted
%% faithfully.
%%
%% Usage: copy this file next to safeprocess2027_final.tex as ifacconf.cls,
%% then run pdflatex twice.
%%
%% Interface (same user commands as the official class, so the .tex needs
%% no change when the official class is dropped in):
%%   \begin{frontmatter} ... \end{frontmatter}
%%   \title{..}  \thanks{..}  \author[label]{..}  \address[label]{..}
%%   \begin{abstract} .. \end{abstract}   \begin{keyword} .. \sep .. \end{keyword}
%%   natbib (round brackets, author-year), \citep / \citet,
%%   two-column A4 layout with figure* / table* support,  \begin{ack}..\end{ack}
%% ===========================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ifacconf}[2026/10/11 LOCAL APPROXIMATION of IFAC ifacconf.cls
  (draft page-counting only, not the official class)]

% ---------------------------------------------------------------------------
% Base class.  article in 10pt two-column mode supplies, unchanged:
%   * 10/12pt body, \small 9/11pt, \large 12/14pt, \LARGE 17.28/22pt;
%   * \parindent = 1em (10pt);
%   * \sloppy (tolerance 9999, emergencystretch 3em) and \flushbottom, which
%     the reference shows (very loose lines, column bottoms aligned);
%   * float parameters (textfloatsep 20pt plus 2pt minus 4pt, dbltextfloatsep
%     20pt plus 2pt minus 4pt, abovecaptionskip 10pt, belowcaptionskip 0pt)
%     and the default "Figure 1:" / "Table 1:" caption layout.
% Unknown options are passed on to article.
% ---------------------------------------------------------------------------
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,twocolumn,a4paper]{article}

% ---------------------------------------------------------------------------
% Page geometry (measured): A4, text block 174 x 245 mm, left/right margins
% 18 mm, top 25 mm, bottom 27 mm, two 84 mm columns 6 mm apart.  Giving the
% four margins (rather than \textwidth=174mm) matters at the scaled-point
% level: it reproduces the reference column width exactly, so even the
% sub-0.01 bp word positions in stretched lines match.
% ---------------------------------------------------------------------------
\RequirePackage[a4paper,top=25mm,bottom=27mm,left=18mm,right=18mm,
  columnsep=6mm]{geometry}

% amsmath: the source uses \text in math (n_\text{test}); the reference shows
% it set by amsmath (7pt upright roman in subscripts).
\RequirePackage{amsmath}

% Author-year citations in round brackets: "(Brockmann et al., 2024)",
% "Blanke et al. (2016)", multiple citations separated by ";".
% natbib's default \bibsection (\section*{References}) and \bibhang (1em) are
% what the reference shows.
\RequirePackage[round,authoryear]{natbib}

% No running heads and no page numbers anywhere (none in the reference).
\pagestyle{empty}

% No paragraph skip.  article's default (0pt plus 1pt) would stretch under
% \flushbottom; the reference columns show that it does not.
\setlength\parskip{0pt}

% ---------------------------------------------------------------------------
% Sectioning.
%   section:     "1. Introduction"           10pt bold, flush left
%   subsection:  "5.1. The prevalence prior" 10pt italic, flush left
% The number is followed by a full stop and a .6em gap (6.9pt in bold,
% 6.13pt in italic, as measured); a heading that wraps hangs after the number
% (Sec. 6 and Sec. 8 of the draft).  Before-skips are POSITIVE, so the first
% paragraph after a heading is indented, as in the reference.
% Skips were fitted from the reference:
%   section     before 2.2ex plus .5ex minus .2ex, after 1.2ex plus .2ex
%   subsection  before 1.8ex plus .4ex minus .2ex, after 1ex   plus .2ex
% (\section* -- e.g. the "References" heading -- uses the section values.)
% ---------------------------------------------------------------------------
\renewcommand\@seccntformat[1]{\csname the#1\endcsname.\hskip .6em\relax}
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {2.2ex \@plus .5ex \@minus .2ex}%
  {1.2ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {1.8ex \@plus .4ex \@minus .2ex}%
  {1ex \@plus .2ex}%
  {\normalfont\normalsize\itshape}}
% Not used by the draft; given the subsection look so it stays consistent.
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {1.8ex \@plus .4ex \@minus .2ex}%
  {1ex \@plus .2ex}%
  {\normalfont\normalsize\itshape}}

% Acknowledgements environment of the IFAC template (not used by the draft).
\newenvironment{ack}{\section*{Acknowledgements}}{}

% ---------------------------------------------------------------------------
% Front matter.
%
% \begin{frontmatter} only collects material; \end{frontmatter} typesets it
% as a full-width block on top of page 1 via \twocolumn[...].  Measured
% layout (baseline to baseline unless stated):
%   top of text block -> 1st title baseline   32pt + height of 1st title line
%   title             \LARGE bold (17.28/22pt), centred
%   title -> authors  4mm + 14pt     authors \large (12pt), centred, ", "-joined
%   authors -> addr.  5.646pt + 11pt address \small italic, centred, on a
%                                    measure of 0.85\textwidth
%   addr. -> abstract 19.622pt + 12pt
%   abstract          full width, "Abstract:" bold, run in
%   abstract -> kw.   2.5mm + 12pt
%   keywords          "Keywords:" italic, run in
%   kw. -> column 1   22.862pt + dbltextfloatsep (20pt) + topskip (10pt)
% The label is followed by \space and, in the source, by the end of the
% \begin{abstract} line: two inter-word spaces after the colon (11.6pt and
% 18.9pt in the two justified first lines of the reference).
% (The address measure is only bounded by the reference -- its first line
% breaks before "2007," -- to 0.82-0.87\textwidth; 0.85 is a guess.)
%
% Authors and addresses: labels are ignored when there is only one address
% (the draft's case, giving "Chenxuan Zhou, Christy Jie Liang" with no
% marks).  With several addresses, authors and addresses get matching
% superscript marks *, **, *** in the order the addresses are declared.
% ---------------------------------------------------------------------------
\newbox\ifac@absbox
\newbox\ifac@kwbox
\newcount\ifac@naddr
\def\ifac@authors{}
\def\ifac@addresses{}
\def\ifac@thankstext{}
\def\@title{}

\renewcommand\title[1]{\gdef\@title{#1}}

% \thanks after \title (IFAC's place for funding notes): a star on the title
% and an unnumbered footnote at the foot of the first column.
\renewcommand\thanks[1]{%
  \ifx\ifac@thankstext\@empty
    \g@addto@macro\@title{\ifac@thanksmark}%
    \gdef\ifac@thankstext{#1}%
  \else
    \g@addto@macro\ifac@thankstext{ #1}%
  \fi}
\def\ifac@thanksmark{\textsuperscript{\normalfont$\star$}}

% \author[label1,label2]{Name}
\renewcommand\author[2][]{%
  \g@addto@macro\ifac@authors{\ifac@au{#1}{#2}}}
% \address[label]{Text}
\newcommand\address[2][]{%
  \global\advance\ifac@naddr\@ne
  \expandafter\xdef\csname ifac@mark@#1\endcsname{\the\ifac@naddr}%
  \g@addto@macro\ifac@addresses{\ifac@ad{#1}{#2}}}

% superscript mark for a list of labels: *, **, ***, ... (only if >1 address)
\def\ifac@marks#1{%
  \ifnum\ifac@naddr>\@ne
    \def\ifac@msep{}%
    \textsuperscript{\normalfont
      \@for\ifac@l:=#1\do{%
        \ifac@msep\def\ifac@msep{,}%
        \@ifundefined{ifac@mark@\ifac@l}{?}{%
          \@tempcnta\csname ifac@mark@\ifac@l\endcsname\relax
          \@whilenum\@tempcnta>\z@\do{$\ast$\advance\@tempcnta\m@ne}}}}%
  \fi}
\def\ifac@authorlist{%
  \def\ifac@sep{}%
  \def\ifac@au##1##2{\ifac@sep\def\ifac@sep{, }##2\ifac@marks{##1}}%
  \ifac@authors}
\def\ifac@addresslist{%
  \def\ifac@sep{}%
  \def\ifac@ad##1##2{\ifac@sep\def\ifac@sep{\par}\ifac@marks{##1}##2}%
  \ifac@addresses}

% \sep between keywords (elsarticle-style templates) -> comma
\providecommand\sep{\unskip, \ignorespaces}

% abstract / keyword: typeset at once into full-width boxes.  Each box
% starts with \null so its first baseline sits exactly \baselineskip below
% the box top; \ifac@putbox then gives baseline-to-baseline spacing.
\let\abstract\relax \let\endabstract\relax
\newenvironment{abstract}{%
  \global\setbox\ifac@absbox\vbox\bgroup
    \hsize\textwidth \linewidth\textwidth \@parboxrestore \normalsize
    \null\noindent\textbf{Abstract:}\space}%
  {\par\egroup}
\newenvironment{keyword}{%
  \global\setbox\ifac@kwbox\vbox\bgroup
    \hsize\textwidth \linewidth\textwidth \@parboxrestore \normalsize
    \null\noindent\textit{Keywords:}\space}%
  {\par\egroup}

% \ifac@putbox{skip}{box}: start the box's first baseline at
% previous baseline + skip + 12pt; afterwards \prevdepth is the depth of the
% box's last line, as if its paragraphs had been typeset in place.
\def\ifac@putbox#1#2{%
  \ifvoid#2\else
    \vskip-\prevdepth \vskip#1\relax
    \@tempdima\dp#2\relax \unvbox#2\prevdepth\@tempdima
  \fi}

\newenvironment{frontmatter}{}{\ifac@maketitle}

\def\ifac@maketitle{%
  \twocolumn[{%
    \vskip 32pt
    {\centering\LARGE\bfseries \@title\par}%
    \vskip 4mm
    {\centering\large \ifac@authorlist\par}%
    \ifx\ifac@addresses\@empty\else
      \vskip 5.646pt
      {\centering\small\itshape
       \leftskip .075\textwidth \@plus 1fil \rightskip\leftskip
       \ifac@addresslist\par}%
    \fi
    \ifac@putbox{19.622pt}\ifac@absbox
    \ifac@putbox{2.5mm}\ifac@kwbox
    \vskip-\prevdepth \vskip 22.862pt
  }]%
  \ifx\ifac@thankstext\@empty\else
    \begingroup
      \renewcommand\thefootnote{\ensuremath{\star}}%
      \footnotetext{\ifac@thankstext}%
    \endgroup
  \fi
  \global\let\ifac@thankstext\@empty}

% \maketitle is not part of the IFAC interface (the frontmatter environment
% does the job); make a stray one harmless.
\let\maketitle\relax

\endinput
%% end of ifacconf.cls (local approximation)
