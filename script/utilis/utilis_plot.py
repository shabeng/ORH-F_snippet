import colorsys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def color_rgbs_range(min_val=0.5, max_val=4, num=10, base_color='navy'):
    color = matplotlib.colors.ColorConverter.to_rgb(base_color)
    scale_range = np.linspace(min_val, max_val, num)
    rgbs = [scale_lightness(color, scale) for scale in scale_range]
    return rgbs


def results_df_creation(df, is_return_ratio_df=False):
    """
    Take the results of the different rules and create the data for the mean graph, ratio and absolute values.
    :param df: DataFrame of the raw results of the test simulation evaluation
    :param is_return_ratio_df: Bool, whether or not to return the ratio DataFrame the function builds.
    :return: DataFrame
    """
    rule_names = list(df.columns)
    if 'Seed' in rule_names:
        rule_names.remove('Seed')
    rule_names.remove('reqNum')
    assert 'time_earliest_arriving' in rule_names, 'EA not in the simulation results'
    df_ratio = pd.DataFrame()
    df_ratio['Seed'] = df.Seed
    df_ratio['reqNum'] = df.reqNum
    for rule_name in rule_names:
        df_ratio[rule_name] = ((df[rule_name] / df['time_earliest_arriving']) - 1) * 100
    if is_return_ratio_df:
        return df.mean()[rule_names], df_ratio.mean()[rule_names], df_ratio
    else:
        return df.mean()[rule_names], df_ratio.mean()[rule_names]


def plot_bars_results_old(plot_num, x_vals_lst, y_vals_lst, x_ticks, y_label,
                          data_rules_num, color,
                          plus_label, minus_label, l_lim, h_lim, bar_width,
                          sub_titles_lst, main_title,
                          annotate_rule=(False, 0, 0, 0, 'None'),
                          save_fig=False, path_save='', graph_name='', dpi=1000, fig_size=None,
                          ):
    fig_size = fig_size if fig_size is not None else ((4 / 5) * len(x_ticks) * plot_num, 5)
    f, axes = plt.subplots(1, plot_num, sharey=True, sharex=True, figsize=fig_size, dpi=dpi)
    for j in range(plot_num):
        x_vals = x_vals_lst[j]
        y_vals = y_vals_lst[j]
        ax = axes[j] if plot_num > 1 else axes
        # Save the chart so we can loop through the bars below.
        bars = ax.bar(
            x=x_vals,
            height=list(y_vals),
            tick_label=x_ticks,
            width=bar_width,
        )
        for i in range(len(bars[:-data_rules_num])):
            bars[i].set_color((40 / 250, 90 / 250, 224 / 250))

        for i in range(1, data_rules_num + 1):
            bars[-i].set_color(color)

        bars[len(bars) - data_rules_num - 1].set_color((105 / 250, 205 / 250, 205 / 250))
        bars[len(bars) - data_rules_num - 2].set_color((105 / 250, 205 / 250, 205 / 250))

        # Axis formatting.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)

        # Add text annotations to the top of the bars.
        bar_color = bars[0].get_facecolor()
        for ind, bar in enumerate(bars):
            if ind in range(len(bars) - data_rules_num, len(bars) + 1):
                bar_color = color
            elif ind in [len(bars) - data_rules_num - 1, len(bars) - data_rules_num - 2]:
                bar_color = (105 / 250, 205 / 250, 205 / 250)
            if bar.get_height() > 0:
                h = bar.get_height() + plus_label
            else:
                h = bar.get_height() - minus_label
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                round(bar.get_height(), 2),
                horizontalalignment='center',
                color=bar_color,
                weight='bold'
            )

        ax.set_ylim((l_lim, h_lim))
        # Add brackets to Our policies
        if annotate_rule[0]:
            draw_brace(ax=ax, xspan=(annotate_rule[1], annotate_rule[2]), yy=annotate_rule[3],
                       text='Ours')

        # Add labels and a title. Note the use of `labelpad` and `pad` to add some
        # extra space between the text and the tick labels.
        # if plot_num == 1 or (plot_num > 1 and j == 1):
        #     ax.set_xlabel('Rule Name', labelpad=15, color='#333333', size=13)
        # if j == 0:
        #     ax.set_ylabel(y_label, labelpad=10, color='#333333', size=13)
        #
        #
        ax.set_title(sub_titles_lst[j], pad=40, color='#333333',
                     weight='bold', size=14)
    if annotate_rule[0]:
        f.supxlabel('Rule Name', color='#333333', size=13, y=0.1)
    else:
        f.supxlabel('Rule Name', color='#333333', size=13)

    f.supylabel(y_label, color='#333333', size=13)
    f.suptitle(main_title, fontsize=15, weight='bold')

    f.tight_layout()
    if save_fig:
        plt.savefig(f'{path_save}/{graph_name}.png', bbox_inches='tight')
    plt.show()


def plot_bars_results(plot_num, x_vals_lst, y_vals_lst, x_ticks, y_label,
                      data_rules_num, color_ours,
                      plus_label, minus_label, l_lim, h_lim, bar_width,
                      sub_titles_lst, main_title, is_hatched_bars=False, hatch_lst=[],
                      annotate_rule=(False, 0, 0, 0, 'None'),
                      save_fig=False, path_save='', graph_name='', dpi=1000, fig_size=None,
                      bar_fontsize=7.5, objtitle_fontsize=8, xticks_label_fontsize=7, yticks_label_fontsize=7,
                      yticks_diff=0.5, supxlabel_fontsize=9, supxlabel_y_loc=0.05, supylabel_fontsize=9,
                      suptitle_fontsize=9, suptitle_y_loc=0.95, our_annotate_fontsize=5,
                      legend_loc='lower right', legend_size=6):
    fig_size = fig_size if fig_size is not None else ((1 / 5) * len(x_ticks) * plot_num, 5)
    f, axes = plt.subplots(1, plot_num, sharey=True, sharex=True, figsize=fig_size, dpi=dpi)
    for j in range(plot_num):
        x_vals = x_vals_lst[j]
        y_vals = y_vals_lst[j]
        ax = axes[j] if plot_num > 1 else axes
        # Save the chart so we can loop through the bars below.
        if is_hatched_bars:
            bars = ax.bar(
                x=x_vals,
                height=list(y_vals),
                tick_label=x_ticks,
                width=bar_width,
                hatch=hatch_lst, fill=False,  # linewidth=0,
                edgecolor=[(40 / 250, 90 / 250, 224 / 250)] * 2 + [(105 / 250, 205 / 250, 205 / 250)] * 2 + [
                    color_ours] * data_rules_num,
            )
        else:
            bars = ax.bar(
                x=x_vals,
                height=list(y_vals),
                tick_label=x_ticks,
                width=bar_width,
            )
            for i in range(len(bars[:-data_rules_num])):
                bars[i].set_color((40 / 250, 90 / 250, 224 / 250))

            for i in range(1, data_rules_num + 1):
                bars[-i].set_color(color_ours)

            bars[len(bars) - data_rules_num - 1].set_color((105 / 250, 205 / 250, 205 / 250))
            bars[len(bars) - data_rules_num - 2].set_color((105 / 250, 205 / 250, 205 / 250))

        # Axis formatting.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)

        # Add text annotations to the top of the bars.
        bar_color = (40 / 250, 90 / 250, 224 / 250)  # bars[0].get_facecolor()
        for ind, bar in enumerate(bars):
            if ind in range(len(bars) - data_rules_num, len(bars) + 1):
                bar_color = color_ours
            elif ind in [len(bars) - data_rules_num - 1, len(bars) - data_rules_num - 2]:
                bar_color = (105 / 250, 205 / 250, 205 / 250)
            if bar.get_height() > 0:
                h = bar.get_height() + plus_label
            else:
                h = bar.get_height() - minus_label
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                round(bar.get_height(), 2),
                horizontalalignment='center',
                color=bar_color,
                weight='bold',
                size=bar_fontsize
            )

        ax.set_ylim((l_lim, h_lim))
        # Add brackets to Our policies
        if annotate_rule[0]:
            draw_brace(ax=ax, xspan=(annotate_rule[1], annotate_rule[2]), yy=annotate_rule[3],
                       text='Ours', font_size=our_annotate_fontsize)

        ax.set_title(sub_titles_lst[j], pad=10, color='#333333',
                     weight='bold', size=objtitle_fontsize)

        ax.set_xticklabels(x_ticks, rotation=0, size=xticks_label_fontsize)
        ax.set_yticks(np.arange(l_lim, h_lim + yticks_diff, yticks_diff), rotation=0, size=yticks_label_fontsize)
        ax.set_yticklabels(np.arange(l_lim, h_lim + yticks_diff, yticks_diff), rotation=0, size=yticks_label_fontsize)

    f.supxlabel('Rule Name', color='#333333', size=supxlabel_fontsize, y=supxlabel_y_loc)

    f.supylabel(y_label, color='#333333', size=supylabel_fontsize)
    f.suptitle(main_title, fontsize=suptitle_fontsize, weight='bold', y=suptitle_y_loc, color='#333333')

    if is_hatched_bars:
        hatch_veh_based = mpatches.Patch(edgecolor=(40 / 250, 90 / 250, 224 / 250),
                                         hatch=hatch_lst[0], label='Vehicle-based', fill=False)
        hatch_area_based = mpatches.Patch(edgecolor=(105 / 250, 205 / 250, 205 / 250),
                                          hatch=hatch_lst[2]+'---', label='Area-based', fill=False)
        hatch_ours = mpatches.Patch(edgecolor=color_ours,
                                    hatch=hatch_lst[-1], label='Ours', fill=False)
        plt.legend(handles=[hatch_veh_based, hatch_area_based, hatch_ours], prop={'size': legend_size}, loc=legend_loc)

    f.tight_layout()
    if save_fig:
        plt.savefig(f'{path_save}/{graph_name}.png', bbox_inches='tight')
    plt.show()


def draw_brace(ax, xspan, yy, text, font_size=6):
    """Draws an annotated brace outside the axes."""
    # From: https://stackoverflow.com/questions/18386210/annotating-ranges-of-data-in-matplotlib/68180887#68180887
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1  # guaranteed uneven
    beta = 300./xax_span  # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01) * yspan  # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, -y, color='black', lw=1, clip_on=False)

    ax.text((xmax+xmin)/2., -yy-.12*yspan, text, ha='center', va='bottom', fontsize=font_size)


def plot_pareto_front_results(df_rej, df_wait, color, rule_names_lst, y_ticks, x_ticks,
                              delta=0.15, dpi=100,
                              save_fig=False, path_save='', graph_name='', fig_size=(4, 4)):
    colors = {'blue': (40/250, 90/250, 224/250), 'lightblue': (105/250, 205/250, 205/250), 'data': color}
    rule2title = {'time_earliest_arriving': 'EA', 'time_nearest_available': 'CV', 'random_available': 'R',
                  'balance_crowded_zone_EA': 'MC', 'balance_balanced_zone_EA': 'MC-N'}
    fig = plt.figure(dpi=dpi, figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(color='lightgrey', linewidth=0.5)
    ax.set_axisbelow(True)
    y_points = df_rej[rule_names_lst]
    x_points = df_wait[rule_names_lst]
    for rule in rule_names_lst:
        if rule in ['time_nearest_available', 'random_available']:
            ax.scatter(x_points[rule], y_points[rule], color=colors['blue'], marker="o")
            ax.annotate(rule2title[rule], (x_points[rule]+delta, y_points[rule]))
        elif rule in ['balance_crowded_zone_EA', 'balance_balanced_zone_EA']:
            ax.scatter(x_points[rule], y_points[rule], color=colors['lightblue'], marker="^")
            plt.annotate(rule2title[rule], (x_points[rule]+delta, y_points[rule]))
        else:
            ax.scatter(x_points[rule], y_points[rule], color=colors['data'], marker="p")
            ax.annotate('Ours', (x_points[rule]+delta, y_points[rule]))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=8)
    ax.set_ylabel('Increase in Rejection Rate [%]', fontsize=9)
    ax.set_xlabel('Increase in Mean Waiting Time [%]', fontsize=9)
    ax.set_title('Increase in Mean Waiting Time and Rejection Rate\nRelative to EA - 60 minutes',
              fontsize=9, weight='bold', pad=10)
    if save_fig:
        plt.savefig(f'{path_save}/{graph_name}.png', bbox_inches='tight')
    plt.show()
    return ax
