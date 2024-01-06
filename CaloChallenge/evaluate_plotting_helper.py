# pylint: disable=invalid-name
""" helper file containing plotting functions to evaluate contributions to the
    Fast Calorimeter Challenge 2022.

    by C. Krause
"""

labels = {
        '1-photons' : "Dataset 1 (photons)",
        '1-pions' : "Dataset 1 (pions)", 
        '2' : "Dataset 2 (electrons)",
        '3' : "Dataset 3 (electrons)",}

import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib import gridspec
rc('text', usetex=True)

rc('font', family='serif')
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

# #
import matplotlib as mpl
mpl.rcParams.update({'font.size': 19})
#mpl.rcParams.update({'legend.fontsize': 18})
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({'xtick.labelsize': 18}) 
mpl.rcParams.update({'ytick.labelsize': 18}) 
mpl.rcParams.update({'axes.labelsize': 18}) 
mpl.rcParams.update({'legend.frameon': False}) 
mpl.rcParams.update({'lines.linewidth': 2})

label1 = 'CaloDiffusion'
label2 = 'Geant4'
plt_ext = 'pdf'

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=24)

    # #
    mpl.rcParams.update({'font.size': 26})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.major.size': 8}) 
    mpl.rcParams.update({'xtick.major.width': 1.5}) 
    mpl.rcParams.update({'xtick.minor.size': 4}) 
    mpl.rcParams.update({'xtick.minor.width': 0.8}) 
    mpl.rcParams.update({'ytick.major.size': 8}) 
    mpl.rcParams.update({'ytick.major.width': 1.5}) 
    mpl.rcParams.update({'ytick.minor.size': 4}) 
    mpl.rcParams.update({'ytick.minor.width': 0.8}) 

    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 26}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 4})
    
    import matplotlib.pyplot as plt




def plot_layer_comparison(hlf_class, data, reference_class, reference_data, arg, show=False):
    """ plots showers of of data and reference next to each other, for comparison """
    num_layer = len(reference_class.relevantLayers)
    vmax = np.max(reference_data)
    layer_boundaries = np.unique(reference_class.bin_edges)
    for idx, layer_id in enumerate(reference_class.relevantLayers):
        plt.figure(figsize=(6, 4))
        reference_data_processed = reference_data\
            [:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        reference_class._DrawSingleLayer(reference_data_processed,
                                         idx, filename=None,
                                         title='Reference Layer '+str(layer_id),
                                         fig=plt.gcf(), subplot=(1, 2, 1), vmax=vmax,
                                         colbar='None')
        data_processed = data[:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        hlf_class._DrawSingleLayer(data_processed,
                                   idx, filename=None,
                                   title='Generated Layer '+str(layer_id),
                                   fig=plt.gcf(), subplot=(1, 2, 2), vmax=vmax, colbar='both')

        filename = os.path.join(arg.output_dir,
                                'Average_Layer_{}_dataset_{}.{}'.format(layer_id, arg.dataset, plt_ext))
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.close()


def SetGrid(ratio=True):
    fig = plt.figure(figsize=(10, 10))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def plot_Etot_Einc_discrete(hlf_class, reference_class, arg, ratio = False):
    """ plots Etot normalized to Einc histograms for each Einc in ds1 """
    # hardcode boundaries?
    bins = np.linspace(0.4, 1.4, 21)
    if("pion" in arg.dataset):
        bins = np.linspace(0., 2.0, 21)

    plt.figure(figsize=(10, 10))
    target_energies = 2**np.linspace(8, 23, 16)
    eps = 1.
    for i in range(len(target_energies)-1):
        if i > 3 and 'photons' in arg.dataset:
            bins = np.linspace(0.9, 1.1, 21)
        energy = target_energies[i]
        which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i] - eps) & \
                             (reference_class.Einc.squeeze() < target_energies[i] + eps)).squeeze()
        which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i] - eps) & \
                             (hlf_class.Einc.squeeze() < target_energies[i] + eps)).squeeze()
        ax = plt.subplot(4, 4, i+1)
        gen = hlf_class.GetEtot()[which_showers_hlf]
        ref = reference_class.GetEtot()[which_showers_ref]
        counts_ref, _, _ = ax.hist(reference_class.GetEtot()[which_showers_ref] /\
                                   reference_class.Einc.squeeze()[which_showers_ref],
                                   bins=bins, label=label2, density=True, facecolor = 'silver',
                                   histtype='stepfilled', alpha=1.0, linewidth=2.)
        counts_data, _, _ = ax.hist(hlf_class.GetEtot()[which_showers_hlf] /\
                                    hlf_class.Einc.squeeze()[which_showers_hlf], bins=bins,color = 'blue',
                                    label=label1, histtype='step', linewidth=3., alpha=1.,
                                    density=True)

        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)
        ax.text(0.95, 0.95, energy_label, ha='right', va='top',
                transform=ax.transAxes)
        ax.set_xlabel('Dep. Energy / Gen. Energy')
        ax.xaxis.set_label_coords(1., -0.15)
        ax.set_ylabel('counts')
        ax.yaxis.set_ticklabels([])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        h, l = ax.get_legend_handles_labels()
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Etot / Einc at E = {}: \n'.format(energy))
                f.write(str(seps))
                f.write('\n\n')
    ax = plt.subplot(4, 4, 16)
    ax.legend(h, l, loc='center', fontsize=20)
    ax.axis('off')
    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.{}'.format(arg.dataset, plt_ext))
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_Etot_Einc(hlf_class, reference_class, arg, ratio = False):
    """ plots Etot normalized to Einc histogram """

    if("pion" in arg.dataset):
        xmin, xmax = (0., 2.0)
    else:
        xmin, xmax = (0.5, 1.5)

    bins = np.linspace(xmin, xmax, 101)
    fig,gs = SetGrid(ratio) 
    ax0 = plt.subplot(gs[0])
    if(ratio):
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    counts_ref, _, _ = ax0.hist(reference_class.GetEtot() / reference_class.Einc.squeeze(),
                                bins=bins, label=label2, density=True, facecolor = 'silver',
                                histtype='stepfilled', alpha=1.0, linewidth=2.)
    counts_data, _, _ = ax0.hist(hlf_class.GetEtot() / hlf_class.Einc.squeeze(), bins=bins,color = 'blue',
                                 label=label1, histtype='step', linewidth=3., alpha=1.,
                                 density=True)

    if(ratio):
        eps = 1e-8
        h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

        ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
        ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
        ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

        xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
        ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
        plt.ylabel('Diff. (%)')
        plt.ylim([-50,50])

        plt_label = labels[arg.dataset]
        ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')

    ax0.set_ylabel("Arbitrary units")
    plt.xlim(xmin, xmax)
    #plt.xlabel(r'$E_{\mathrm{tot}} / E_{\mathrm{inc}}$')
    plt.xlabel('Dep. Energy / Gen. Energy')
    ax0.legend(fontsize=20)
    plt.margins(0.05, 0.2)
    plt.tight_layout()
    if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}.{}'.format(arg.dataset, plt_ext))
        plt.savefig(filename, dpi=300)
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()


def plot_E_layers(hlf_class, reference_class, arg, ratio = False):
    """ plots energy deposited in each layer """


    for key in hlf_class.GetElayers().keys():

        fig,gs = SetGrid(ratio) 
        ax0 = plt.subplot(gs[0])
        if(ratio):
            plt.xticks(fontsize=0)
            ax1 = plt.subplot(gs[1],sharex=ax0)

        if arg.x_scale == 'log':
            bins = np.logspace(np.log10(arg.min_energy),
                               np.log10(reference_class.GetElayers()[key].max()),
                               40)
        else:
            bins = 40
        counts_ref, bins, _ = ax0.hist(reference_class.GetElayers()[key], bins=bins, facecolor = 'silver',
                                       label=label2, density=True, histtype='stepfilled',
                                       alpha=1.0, linewidth=2.)
        counts_data, _, _ = ax0.hist(hlf_class.GetElayers()[key], label=label1, bins=bins, color = 'blue', 
                                     histtype='step', linewidth=3., alpha=1., density=True)

        if(ratio):
            eps = 1e-8
            h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

            ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
            ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
            ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

            xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
            ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
            plt.ylabel('Diff. (%)')
            plt.ylim([-50,50])

            plt_label = labels[arg.dataset]
            ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')


        ax0.set_ylabel("Arbitrary units")
        plt.xlabel("Layer {} Energy [MeV]".format(key))
        #plt.xlabel(r'$E$ [MeV]')
        ax0.set_yscale('log')
        plt.margins(0.05, 0.5)
        plt.xscale('log')
        ax0.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir, 'E_layer_{}_dataset_{}.{}'.format(
                key,
                arg.dataset, plt_ext))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of E layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('E layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECEtas(hlf_class, reference_class, arg, ratio = False):
    """ plots center of energy in eta """
    for key in hlf_class.GetECEtas().keys():

        fig,gs = SetGrid(ratio) 
        ax0 = plt.subplot(gs[0])
        if(ratio):
            plt.xticks(fontsize=0)
            ax1 = plt.subplot(gs[1],sharex=ax0)


        if arg.dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = ax0.hist(reference_class.GetECEtas()[key], bins=bins,facecolor = 'silver',
                                    label=label2, density=True, histtype='stepfilled',
                                    alpha=1.0, linewidth=2.)
        counts_data, _, _ = ax0.hist(hlf_class.GetECEtas()[key], label=label1, bins=bins, color = 'blue',
                                     histtype='step', linewidth=3., alpha=1., density=True)

        if(ratio):
            eps = 1e-8
            h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

            ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
            ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
            ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

            xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
            ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
            plt.ylabel('Diff. (%)')
            plt.ylim([-50,50])

            plt_label = labels[arg.dataset]
            ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')

        ax0.set_ylabel("Arbitrary units")
        plt.xlabel(r"Layer {} Center of Energy in $x$ [mm]".format(key))
        #plt.xlabel(r'[mm]')
        plt.xlim(*lim)
        ax0.set_yscale('log')
        plt.margins(0.05, 0.5)
        ax0.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir, 'ECEta_layer_{}_dataset_{}.{}'.format(key, arg.dataset, plt_ext))
            plt.savefig(filename, dpi=300)

        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of EC Eta layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('EC Eta layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECPhis(hlf_class, reference_class, arg, ratio = False):
    """ plots center of energy in phi """
    for key in hlf_class.GetECPhis().keys():

        fig,gs = SetGrid(ratio) 
        ax0 = plt.subplot(gs[0])
        if(ratio):
            plt.xticks(fontsize=0)
            ax1 = plt.subplot(gs[1],sharex=ax0)

        if arg.dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = ax0.hist(reference_class.GetECPhis()[key], bins=bins,facecolor = 'silver',
                                    label=label2, density=True, histtype='stepfilled',
                                    alpha=1.0, linewidth=2.)
        counts_data, _, _ = ax0.hist(hlf_class.GetECPhis()[key], label=label1, bins=bins,color = 'blue',
                                     histtype='step', linewidth=3., alpha=1., density=True)

        if(ratio):
            eps = 1e-8
            h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

            ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
            ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
            ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

            xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
            ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
            plt.ylabel('Diff. (%)')
            plt.ylim([-50,50])

            plt_label = labels[arg.dataset]
            ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')

        ax0.set_ylabel("Arbitrary units")
        plt.xlabel(r"Layer {} Center of Energy in $y$ [mm]".format(key))
        plt.xlim(*lim)
        ax0.set_yscale('log')
        plt.margins(0.05, 0.5)
        ax0.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECPhi_layer_{}_dataset_{}.{}'.format(key,
                                                                           arg.dataset, plt_ext))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of EC Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('EC Phi layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECWidthEtas(hlf_class, reference_class, arg, ratio = False):
    """ plots width of center of energy in eta """
    for key in hlf_class.GetWidthEtas().keys():

        fig,gs = SetGrid(ratio) 
        ax0 = plt.subplot(gs[0])
        if(ratio):
            plt.xticks(fontsize=0)
            ax1 = plt.subplot(gs[1],sharex=ax0)

        if arg.dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = ax0.hist(reference_class.GetWidthEtas()[key], bins=bins, facecolor = 'silver',
                                    label=label2, density=True, histtype='stepfilled',
                                    alpha=1.0, linewidth=2.)
        counts_data, _, _ = ax0.hist(hlf_class.GetWidthEtas()[key], label=label1, bins=bins, color = 'blue',
                                     histtype='step', linewidth=3., alpha=1., density=True)

        if(ratio):
            eps = 1e-8
            h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

            ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
            ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
            ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

            xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
            ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
            plt.ylabel('Diff. (%)')
            plt.ylim([-50,50])

            plt_label = labels[arg.dataset]
            ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')

        ax0.set_ylabel("Arbitrary units")
        plt.xlabel(r"Layer {} $x$ Width".format(key))
        plt.xlim(*lim)
        ax0.set_yscale('log')
        ax0.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthEta_layer_{}_dataset_{}.{}'.format(key,
                                                                              arg.dataset, plt_ext))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Eta layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Width Eta layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECWidthPhis(hlf_class, reference_class, arg, ratio = False):
    """ plots width of center of energy in phi """
    for key in hlf_class.GetWidthPhis().keys():

        fig,gs = SetGrid(ratio) 
        ax0 = plt.subplot(gs[0])
        if(ratio):
            plt.xticks(fontsize=0)
            ax1 = plt.subplot(gs[1],sharex=ax0)

        if arg.dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = ax0.hist(reference_class.GetWidthPhis()[key], bins=bins, facecolor = 'silver',
                                    label=label2, density=True, histtype='stepfilled',
                                    alpha=1.0, linewidth=2.)
        counts_data, _, _ = ax0.hist(hlf_class.GetWidthPhis()[key], label=label1, bins=bins, color = 'blue',
                                     histtype='step', linewidth=3., alpha=1., density=True)

        if(ratio):
            eps = 1e-8
            h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

            ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
            ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
            ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

            xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
            ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
            plt.ylabel('Diff. (%)')
            plt.ylim([-50,50])

            plt_label = labels[arg.dataset]
            ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')

        ax0.set_ylabel("Arbitrary units")
        plt.xlabel(r"Layer {} $y$ Width" .format(key))
        ax0.set_yscale('log')
        plt.xlim(*lim)
        ax0.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthPhi_layer_{}_dataset_{}.{}'.format(key,
                                                                              arg.dataset, plt_ext))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Width Phi layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_cell_dist(shower_arr, ref_shower_arr, arg, ratio = True):
    """ plots voxel energies across all layers """
    fig,gs = SetGrid(ratio) 
    ax0 = plt.subplot(gs[0])
    if(ratio):
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    if arg.x_scale == 'log':
        bins = np.logspace(np.log10(arg.min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50


    print('voxel range', np.amax(ref_shower_arr.flatten()), np.amin(ref_shower_arr.flatten()))

    eps = 1e-16
    counts_ref, _, _ = ax0.hist(ref_shower_arr.flatten(), bins=bins,facecolor = 'silver',
                                label=label2, density=True, histtype='stepfilled',
                                alpha=1.0, linewidth=2.)
    counts_data, _, _ = ax0.hist(shower_arr.flatten() + eps, label=label1, bins=bins, color = 'blue',
                                 histtype='step', linewidth=3., alpha=1., density=True)
    print(counts_ref[:10])

    if(ratio):
        eps = 1e-8
        h_ratio = 100. * (counts_data - counts_ref) / (counts_ref + eps)

        ax1.axhline(y=0.0, color='black', linestyle='-',linewidth=2)
        ax1.axhline(y=10, color='gray', linestyle='--',linewidth=2)
        ax1.axhline(y=-10, color='gray', linestyle='--',linewidth=2)

        xaxis = [(bins[i] + bins[i+1])/2.0 for i in range(len(bins)-1)]
        ax1.plot(xaxis,h_ratio,color='blue',linestyle='-',linewidth = 3)
        plt.ylabel('Diff. (%)')
        plt.ylim([-50,50])

        plt_label = labels[arg.dataset]
        ax0.set_title(plt_label, fontsize = 20, loc = 'right', style = 'italic')

    ax0.set_ylabel("Arbitrary units")
    plt.xlabel(r"Voxel Energy [MeV]")
    ax0.set_yscale('log')
    if arg.x_scale == 'log': plt.xscale('log')
    ax0.legend(fontsize=20)
    plt.tight_layout()
    if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(arg.output_dir,
                                'voxel_energy_dataset_{}.{}'.format(arg.dataset,plt_ext))
        plt.savefig(filename, dpi=300)
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of voxel distribution histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir,
                               'histogram_chi2_{}.txt'.format(arg.dataset)), 'a') as f:
            f.write('Voxel distribution: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()

def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()
