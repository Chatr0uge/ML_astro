set table "cr.pgf-plot.table"; set format "%.5f"
set format "%.7e";; set samples 25; set dummy t; plot [t=-5:5] .7 * exp(-(x+1.3)**2/.05);
