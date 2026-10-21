$pdf_mode = 4;        # 4 = lualatex
$aux_dir  = 'build';  # intermediate files (.aux .log .nav .toc ...)
$out_dir  = '.';      # final PDF stays here
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode %O %S';
