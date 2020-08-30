from invoke import task
from pathlib import Path

overleaf = '/Users/deismic/Documents/Studium_TUM/MasterThesis/energy_overleaf'
basepath = '/Users/deismic/Documents/Studium_TUM/MasterThesis/prinzetal/energy_paper'
open_cmd = 'open'

fig_names = {
    '1': 'fig1_posterior',
    '2': 'fig2_histograms',
    '3': 'fig3_amortize_energy',
    '4': 'fig4_sensitivity',
    '5': 'fig5_cc',
}

@task
def syncOverleaf(c, fig):
    _convertsvg2pdf(c, fig)
    c.run('cp {bp}/{fn}/fig/*.pdf {ol}/figs/ '.format(
        bp=basepath, fn=fig_names[fig], ol=overleaf))

    _convertpdf2png(c, fig)
    c.run('cp {bp}/{fn}/fig/*.png {ol}/figs/ '.format(
        bp=basepath, fn=fig_names[fig], ol=overleaf))

########################################################################################
########################################################################################
########################################################################################
# Helpers
########################################################################################
@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path('{bp}/{fn}/fig/'.format(bp=basepath, fn=fig_names[fig])).glob('*.svg')
    for path in pathlist:
        c.run('/Applications/Inkscape.app/Contents/Resources/bin/inkscape {} --export-pdf={}.pdf'.format(str(path), str(path)[:-4]))

@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path('{bp}/{fn}/fig/'.format(bp=basepath, fn=fig_names[fig])).glob('*.pdf')
    for path in pathlist:
        c.run('/Applications/Inkscape.app/Contents/Resources/bin/inkscape {} --export-png={}.png -b "white" --export-dpi=300'.format(str(path), str(path)[:-4]))

@task
def _convert_svg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path('{bp}/{fn}/fig/'.format(bp=basepath, fn=fig_names[fig])).glob('*.svg')
    for path in pathlist:
        c.run('inkscape {} --export-pdf={}.pdf'.format(str(path), str(path)[:-4]))

@task
def _convert_pdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path('{bp}/{fn}/fig/'.format(bp=basepath, fn=fig_names[fig])).glob('*.pdf')
    for path in pathlist:
        c.run('inkscape {} --export-png={}.png -b "white" --export-dpi=300'.format(str(path), str(path)[:-4]))
