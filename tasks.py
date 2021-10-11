from invoke import task
from pathlib import Path

overleaf = "/home/michael/Documents/energy_overleaf"
basepath = "/home/michael/Documents/STG_energy"
open_cmd = "open"

fig_names = {
    "1": "paper/fig1",
    "2": "paper/fig2",
    "3": "paper/fig3",
    "4": "paper/fig4",
    "5": "paper/fig5",
    "6": "paper/fig6",
    "7": "paper/figSupp",
    "8": "cosyne/fig1",
    "9": "cosyne/fig2",
    "10": "paper/fig5_cc",
}


@task
def syncOverleaf(c, fig):
    _convertsvg2pdf(c, fig)
    c.run(
        "cp {bp}/{fn}/fig/*.pdf {ol}/figs/ ".format(
            bp=basepath, fn=fig_names[fig], ol=overleaf
        )
    )

    _convertpdf2png(c, fig)
    c.run(
        "cp {bp}/{fn}/fig/*.png {ol}/figs/ ".format(
            bp=basepath, fn=fig_names[fig], ol=overleaf
        )
    )


@task
def reduceFilesize(c):
    indizes = [2, 6]

    for ind in indizes:

        path = f"{basepath}/paper/fig{ind}/fig/fig{ind}_supp1.svg"
        c.run(f"inkscape {path} --export-pdf={path[:-4]}.pdf")
        c.run(f"cp {basepath}/paper/fig{ind}/fig/fig{ind}_supp1.pdf {overleaf}/figs/")

        c.run(
            f'inkscape {path} --export-png={path[:-4]}.png -b "white" --export-dpi=150'
        )
        c.run(f"cp {basepath}/paper/fig{ind}/fig/fig{ind}_supp1.png {overleaf}/figs/")


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
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.svg"
    )
    for path in pathlist:
        c.run("inkscape {} --export-pdf={}.pdf".format(str(path), str(path)[:-4]))


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.pdf"
    )
    for path in pathlist:
        c.run(
            'inkscape {} --export-png={}.png -b "white" --export-dpi=250'.format(
                str(path), str(path)[:-4]
            )
        )


@task
def _convert_svg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.svg"
    )
    for path in pathlist:
        c.run("inkscape {} --export-pdf={}.pdf".format(str(path), str(path)[:-4]))


@task
def _convert_pdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.pdf"
    )
    for path in pathlist:
        c.run(
            'inkscape {} --export-png={}.png -b "white" --export-dpi=300'.format(
                str(path), str(path)[:-4]
            )
        )
