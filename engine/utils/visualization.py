# scatter_plot(scores, categories, is_adaptive=True) generates the scatter plots
# depicted in Figure. 2 and 3 of the paper. The input is a list of tuples,
# 'scores', the network categories, 'categories', considered in the scatter
# plot, and a boolean variable, 'is_adaptive', indicating whether the attack is
# adaptive or not. The list of the tuples mentioned above is returned by the
# compute_z_score function (see the comments on that function for more details). 
def scatter_plot(scores, categories, is_adaptive=True):
    import matplotlib
    import seaborn as sns
    import numpy as np
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
    from engine.config.config import get_working_dir
    # Set the font size and style, marker size for points in the scatter plot,
    # and the color of scatter points corresponding to each network category.
    matplotlib.use('pdf')
    font = {'family': 'Sans', 'size': 27}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(14, 14))
    plt.rc('text', usetex=True)
    marker_size = 100
    colors = dict(zip(["Technological", "Social", "Biological", "Transportation", "Auxiliary"],
                      ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]))
    # The following restricts the scatter points to those with the desired
    # categories.
    filtered_list = filter(lambda z: z[3] in categories, scores)
    # The following produces the x-axis, y-axis and the hue (marker) used for
    # each scattered point.
    x, y, h = zip(*[(t[is_adaptive], t[2], t[3]) for t in filtered_list])

    # The following produces the scatter plot.
    sns.scatterplot(x=x, y=y, hue=h, palette=colors, s=marker_size)
    # In the following lines we set the legend of the figure properly.
    ax1 = plt.gca()
    # We compute the lower-bound such that 97% of all points have a value at
    # least as high as this lower-bound (to exclude outliers from the figure).
    lower = - np.sqrt(np.percentile([x[i] * x[i] + y[i] * y[i] for i in range(len(x))], 97)) - 500
    # We set the boundaries of the figure such that 97% of all points (except
    # those with an x or y coordinate lower than the above lower-bound) are
    # included.
    upper = 500
    ax1.set_xlim(xmin=lower, xmax=upper * 3.5)
    ax1.set_ylim(ymin=lower, ymax=upper * 3.5)
    # In the following lines we set the legend of the figure properly.
    handles, labels = ax1.get_legend_handles_labels()
    indices = [1, 0, 2, 3, 4]
    labels = [labels[ind] for ind in indices]
    handles = handles[:len(set(h))]
    handles = [handles[ind] for ind in indices]
    ax1.legend(handles=handles, labels=labels, bbox_to_anchor=(-0.17, -0.085), loc=2,
               ncol=len(categories),
               borderaxespad=0., markerscale=2, columnspacing=0.2)
    # Below, we set the labels of the x-axis and the y-axis in the main scatter plot.
    if is_adaptive:
        ax1.set_xlabel(r"$z$-score targeted attack", fontsize=32, labelpad=3)
    else:
        ax1.set_xlabel(r"$z$-score targeted attack", fontsize=32, labelpad=3)
    ax1.set_ylabel(r"$z$-score random failure", fontsize=32, labelpad=3)
    # The following line draws the identity line in the main scatter plot.
    ax1.axline((0, 0), slope=1, linestyle="--", color="black", lw=3)

    # In the following lines of code we draw the first inset.
    ax2 = plt.axes([0, 0, 1, 1])
    sns.scatterplot(x=x, y=y, ax=ax2, hue=h, palette=colors, s=marker_size)
    # We change the color of the frame, and its line width to mark the first inset.
    for spine in ['right', 'top', 'left', 'bottom']:
        ax2.spines[spine].set_color("0.7")
        ax2.spines[spine].set_linewidth(2)
    # We also make sure to draw the identity line in the first inset.
    ax2.axline((0, 0), slope=1, linestyle="--", color="black", lw=3)
    # Depending on whether the targeted attack is adaptive to not the range of
    # x/y coordinates are specified for the first inset.
    if not is_adaptive:
        ax2.set_xlim(-7500, upper * 1.2)
        ax2.set_ylim(-7500, upper * 1.2)
        rect = patches.Rectangle((-7500, -7500), 7500 + upper, 7500 + upper, 
                                 linewidth=2, edgecolor='0.7', facecolor='none')
        ax1.add_patch(rect)
        ax2.set_xticks([0, -2500, -5000])
        ax2.set_yticks([0, -2500, -5000])
        ax2.get_legend().remove()
        ip = InsetPosition(ax1, [0.59, 0.15, 0.4, 0.4])
        ax2.set_axes_locator(ip)
    else:
        ax2.set_xlim(-8500, upper * 1.2)
        ax2.set_ylim(-8500, upper * 1.2)
        rect = patches.Rectangle((-8500, -8500), 8500 + upper, 8500 + upper, 
                                 linewidth=2, edgecolor='0.7', facecolor='none')
        ax1.add_patch(rect)
        ax2.set_xticks([0, -4250])
        ax2.set_yticks([0, -4250])
        ax2.get_legend().remove()
        ip = InsetPosition(ax1, [0.59, 0.15, 0.4, 0.4])
        ax2.set_axes_locator(ip)
    # The following lines set the location and zooming scale of the second inset.
    if not is_adaptive:
        ax3 = ax2.inset_axes([-1.01, -0.25, 0.8, 0.8 * upper / 2500])
    else:
        ax3 = ax2.inset_axes([-1.01, -0.25, 0.8, 0.8 * upper / 2500])
    # The following line scatters the points in the second inset.
    sns.scatterplot(x=x, y=y, hue=h, palette=colors, ax=ax3, s=marker_size)

    # After the points have been scattered we select the range of the x
    # coordinates and y coordinates that we desire for the inset to focus on.
    # Note that this range is dependent on whether we are considering static or
    # adaptive targeted attacks, as reflected by the if and else condition.
    # Besides what is stated in the above, in these lines, we also set the x/y
    # ticks and labels of the inset, as well as the style that distinguishes the
    # inset.
    if not is_adaptive:
        ax3.set_xlim(-3500, -1000)
        ax3.set_ylim(0, upper)
        ax3.set_xticks([-1000, -2250][::-1])
        ax3.set_yticks([0, 250, 500])
        rect = patches.Rectangle((-3500, 0), 2500, upper, linewidth=2, edgecolor='0.3',
                                 facecolor='none', fill=False,
                                 ls=(0, (1, 1, 1, 1)))
        ax2.add_patch(rect)
        for spine in ['right', 'top', 'left', 'bottom']:
            ax3.spines[spine].set_color("0.3")
            ax3.spines[spine].set_linewidth(2)
            ax3.spines[spine].set_linestyle((0, (0.1, 2, 0.1, 2)))
        ax3.get_legend().remove()
    else:
        ax3.set_xlim(-4000, -1500)
        ax3.set_ylim(0, upper)
        ax3.set_xticks([-1500, -2750][::-1])
        ax3.set_yticks([0, 250, 500])
        rect = patches.Rectangle((-4000, 0), 2500, upper, linewidth=2, edgecolor='0.3',
                                 facecolor='none', fill=False,
                                 ls=(0, (1, 1, 1, 1)))
        ax2.add_patch(rect)
        for spine in ['right', 'top', 'left', 'bottom']:
            ax3.spines[spine].set_color("0.3")
            ax3.spines[spine].set_linewidth(2)
            ax3.spines[spine].set_linestyle((0, (0.1, 2, 0.1, 2)))
        ax3.get_legend().remove()
    # In the following lines we save the figure appropriately in the desired
    # location.
    plt.subplots_adjust(left=.15, bottom=.125, right=.99, top=0.995)
    plt.savefig(get_working_dir() + "/figures/" + "robustness_" + ("adaptive" if is_adaptive else "static") + ".pdf",
                format="pdf", dpi=1200, bbox_inches='tight')


# bar_plot() generates the bar plot depicted in Figure 1. of the paper.
def bar_plot():
    import matplotlib
    import matplotlib.patches as mpatches
    from matplotlib import pyplot as plt
    from engine.utils.io import get_categories, get_networks, get_subnetworks
    from engine.config.config import get_working_dir, get_data_dir
    # Set the font size and style, hatching density, and width of each bar in
    # the bar plot.
    font = {'family': 'Sans', 'size': 28}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['ytick.major.pad'] = 15
    plt.rc('text', usetex=True)
    legend_scale = 2
    hatch_density = 1
    width = 0.7 
    # Specify the labels for each bar in the bar pot and the corresponding
    # color.
    y_labels = ['Auxiliary', 'Transportation', 'Biological', 'Social', 'Technological', 'Full Collection']
    colors = dict(zip(["Technological", "Social", "Biological", "Transportation", "Auxiliary"],
                      ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]))
    colors = [colors[x] for x in y_labels[:-1]] + ["1.0"]
    # Prepare the canvas with the desired size to draw in, remove ticks, and add
    # the title.
    plt.figure(figsize=[16, 12])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title('Network Categories')
    # We count the networks matching specific criteria, the names of the
    # variables are self-explanatory.
    route_views = 0
    tech_other = 0
    facebook = 0
    social_other = 0
    kegg = 0
    bio_other = 0
    transport = 0
    other = 0
    data_dir = get_data_dir()
    for category in get_categories(data_dir):
        for network in get_networks(data_dir, category):
            for _ in get_subnetworks(data_dir, category, network):
                if category == "Technological":
                    if network == "route_views":
                        route_views += 1
                    else:
                        tech_other += 1
                if category == "Social":
                    if network == "Facebook100":
                        facebook += 1
                    else:
                        social_other += 1
                if category == "Biological":
                    if network == "kegg_metabolic":
                        kegg += 1
                    else:
                        bio_other += 1
                if category == "Infrastructure":
                    transport += 1
                if category == "Other":
                    other += 1

    # These following tuples and proceeding temp scores are used to help overlay
    # the bar plots.
    all_other_t = (0, 0, 0, 0, 0, tech_other + social_other + bio_other + transport + other)
    other_t = (other, 0, 0, 0, 0, 0)
    transport_t = (0, transport, 0, 0, 0, 0)
    bio_other_t = (0, 0, bio_other, 0, 0, 0)
    kegg_t = (0, 0, kegg, 0, 0, kegg)
    social_other_t = (0, 0, 0, social_other, 0, 0)
    facebook_t = (0, 0, 0, facebook, 0, facebook)
    tech_other_t = (0, 0, 0, 0, tech_other, 0)
    route_views_t = (0, 0, 0, 0, route_views, route_views)

    tmp1 = [route_views_t[i] + facebook_t[i] for i in range(len(y_labels))]
    tmp2 = [tmp1[i] + kegg_t[i] for i in range(len(y_labels))]
    tmp3 = [tmp2[i] + tech_other_t[i] for i in range(len(y_labels))]
    tmp4 = [tmp3[i] + social_other_t[i] for i in range(len(y_labels))]
    tmp5 = [tmp4[i] + bio_other_t[i] for i in range(len(y_labels))]
    tmp6 = [tmp5[i] + transport_t[i] for i in range(len(y_labels))]
    tmp7 = [tmp6[i] + other_t[i] for i in range(len(y_labels))]
    full = [tmp7[i] + all_other_t[i] for i in range(len(y_labels))]

    # Using the above tmp scores we draw the bars in the bar plot.
    ax.barh(y_labels, route_views_t, width, color=colors, hatch=hatch_density * "x", alpha=0.99)
    ax.barh(y_labels, facebook_t, width, left=route_views_t, color=colors, hatch="O", alpha=0.99)
    ax.barh(y_labels, kegg_t, width, left=tmp1, color=colors, hatch=hatch_density * "+", alpha=0.99)
    ax.barh(y_labels, tech_other_t, width, left=tmp2, color=colors, alpha=0.99)
    ax.barh(y_labels, social_other_t, width, left=tmp3, color=colors, alpha=0.99)
    ax.barh(y_labels, bio_other_t, width, left=tmp4, color=colors, alpha=0.99)
    ax.barh(y_labels, transport_t, width, left=tmp5, color=colors, alpha=0.99)
    ax.barh(y_labels, other_t, width, left=tmp6, color=colors, alpha=0.99)
    ax.barh(y_labels, all_other_t, width, left=tmp7, color='0.3', alpha=0.99)

    # Here, we add labels to the right of each bar.
    ax.text(full[0] + 5, 0, str(full[0]) + r" $(0.00\%)$", ha='left', va='center')  # Other
    ax.text(full[1] + 5, 1, str(full[1]) + r" $(0.00\%)$", ha='left', va='center')  # Transport
    if kegg != 0:
        ax.text(full[2] + 5, 2, str(full[2]) + r" $(" + str(f'{100 * kegg / full[2]:.4}') + "\%)$", ha='left',
                va='center')  # Biological
    if facebook != 0:
        ax.text(full[3] + 5, 3, str(full[3]) + r" $(" + str(f'{100 * facebook / full[3]:.4}') + "\%)$", ha='left',
                va='center')  # Social
    if route_views != 0:
        ax.text(full[4] + 5, 4, str(full[4]) + r" $(" + str(f'{100 * route_views / full[4]:.4}') + "\%)$", ha='left',
                va='center')  # Technological
    if kegg != 0 or facebook != 0 or route_views != 0:
        ax.text(full[5] + 5, 5,
                str(full[5]) + r" $(" + str(f'{100 * (kegg + facebook + route_views) / full[5]:.4}') + "\%)$",
                ha='left', va='center')  # Full Collection

    # Here we create a legend manually and add it to the plot.
    circ1 = mpatches.Patch(facecolor=colors[-1], alpha=0.99, hatch=hatch_density * "x", label='Route Views AS')
    circ2 = mpatches.Patch(facecolor=colors[-1], alpha=0.99, hatch="O", label='Facebook100')
    circ3 = mpatches.Patch(facecolor=colors[-1], alpha=0.99, hatch=hatch_density * "+", label='Kegg Metabolic')
    circ4 = mpatches.Patch(facecolor='0.3', alpha=0.99, label='Various Sources')
    ax.legend(handles=[circ1, circ2, circ3, circ4], fontsize=28, shadow=True,
              handlelength=2 * legend_scale, handleheight=1.5 * legend_scale, loc='lower right', frameon=False)
    # Here, we save the generated figure.
    plt.savefig(get_working_dir() + "/figures/" + "dataset.pdf", format="pdf", bbox_inches="tight", backend="pdf")
    plt.subplots_adjust(left=0.17, bottom=.02, right=.88, top=0.94)


# draw_collins_yeast() creates and save a visualization of the Collins yeast
# interactome network.
def draw_collins_yeast():
    import graph_tool.all as gt
    from engine.config.config import get_working_dir
    # Load the graph.
    g = gt.collection.ns["collins_yeast"]
    # Create vertex property maps capturing respectively the position of the
    # vertices and vertex degrees.
    pos = gt.sfdp_layout(g, epsilon=1e-12)
    deg = g.degree_property_map("out")
    # Create a gray-scale visualization of the graph where the size of vertices
    # are proportional to vertex degrees.
    gt.graph_draw(g, pos=pos, vertex_size=gt.prop_to_size(deg, mi=2, ma=6), vertex_color='0.',
                  vertex_fill_color='0.5',
                  output=get_working_dir() + "/figures/" + "graph-draw.pdf", edge_pen_width=0.5)
