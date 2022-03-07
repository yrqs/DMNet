import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

def show_feature(feature, bar_scope=None):
    norm = None
    if bar_scope:
        vmin, vmax = bar_scope
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure()
    im = plt.imshow(feature.sigmoid().clone()[0][0].cpu(), cmap='rainbow', norm=norm)
    fig.colorbar(im)
    plt.show()
