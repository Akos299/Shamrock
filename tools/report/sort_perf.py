from _testlib import *





import matplotlib.pyplot as plt
plt.style.use('custom_short_cycler.mplstyle')

import numpy as np


def standalone(json_lst : list, figure_folder : str) -> str:


    buf = r"\section{Shamalgs key pair sort}" + "\n\n"

    i = 0
    for report in json_lst:

        res = TestResults(i,report)

        sort_perf = res.get_test_instances("Benchmark","shamalgs/algorithm/details/bitonicSorts:benchmark")

        fileprefix = str(i)

        if len(sort_perf) == 1:


            fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(12,6))

            for s in sort_perf:

                for dataset in s.test_data:

                    n = dataset["dataset_name"]

                    vec_N = np.array(s.get_test_dataset(n,"Nobj"));
                    vec_T = np.array(s.get_test_dataset(n,"t_sort"));

                    if n.startswith("bitonic public"):
                        plt.plot(np.array(vec_N),(np.abs(vec_T)/vec_N), label = n,color='black',linewidth = 2)
                    elif n.startswith("bitonic updated"):
                        plt.plot(np.array(vec_N),(np.abs(vec_T)/vec_N), label = n,linewidth = 2,linestyle="--")
                    else:
                        plt.plot(np.array(vec_N),(np.abs(vec_T)/vec_N), label = n)

            #axs.set_title('Bitonic sort perf')

            axs.set_xscale('log')
            axs.set_yscale('log')

            axs.set_xlabel(r"$N$")

            axs.set_ylabel(r"$t_{\rm sort}/N$ (s)")

            axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs.grid()

            plt.tight_layout()

            plt.savefig(figure_folder+fileprefix+"sort_perf_comp.pdf")

            

            buf += res.get_config_str()
            buf +=  r"""

            \begin{figure}[ht!]
            \center
            \includegraphics[width=0.9\textwidth]{"""+ "figures/"+fileprefix+"sort_perf_comp.pdf" + r"""}
            \caption{TODO}
            \label{fig:fmm_prec}
            \end{figure}

            """
            i += 1

    if i == 0:
        return ""

    return buf


def stacked(json_lst : list, figure_folder : str):

    print(json_in)


def compared(json_lst : list, figure_folder : str):

    print(json_in)



from _test_reader import *
run(standalone, stacked,compared)