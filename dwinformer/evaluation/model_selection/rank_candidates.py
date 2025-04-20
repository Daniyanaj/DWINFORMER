#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from dwinformer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "dwinformerPlans"

    overwrite_plans = {
        'dwinformerTrainerV2_2': ["dwinformerPlans", "dwinformerPlansisoPatchesInVoxels"], # r
        'dwinformerTrainerV2': ["dwinformerPlansnonCT", "dwinformerPlansCT2", "dwinformerPlansallConv3x3",
                            "dwinformerPlansfixedisoPatchesInVoxels", "dwinformerPlanstargetSpacingForAnisoAxis",
                            "dwinformerPlanspoolBasedOnSpacing", "dwinformerPlansfixedisoPatchesInmm", "dwinformerPlansv2.1"],
        'dwinformerTrainerV2_warmup': ["dwinformerPlans", "dwinformerPlansv2.1", "dwinformerPlansv2.1_big", "dwinformerPlansv2.1_verybig"],
        'dwinformerTrainerV2_cycleAtEnd': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_cycleAtEnd2': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_reduceMomentumDuringTraining': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_graduallyTransitionFromCEToDice': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_independentScalePerAxis': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_Mish': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_Ranger_lr3en4': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_fp32': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_GN': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_momentum098': ["dwinformerPlans", "dwinformerPlansv2.1"],
        'dwinformerTrainerV2_momentum09': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_DP': ["dwinformerPlansv2.1_verybig"],
        'dwinformerTrainerV2_DDP': ["dwinformerPlansv2.1_verybig"],
        'dwinformerTrainerV2_FRN': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_resample33': ["dwinformerPlansv2.3"],
        'dwinformerTrainerV2_O2': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_ResencUNet': ["dwinformerPlans_FabiansResUNet_v2.1"],
        'dwinformerTrainerV2_DA2': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_allConv3x3': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_ForceBD': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_ForceSD': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_LReLU_slope_2en1': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_lReLU_convReLUIN': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_ReLU': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_ReLU_biasInSegOutput': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_ReLU_convReLUIN': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_lReLU_biasInSegOutput': ["dwinformerPlansv2.1"],
        #'dwinformerTrainerV2_Loss_MCC': ["dwinformerPlansv2.1"],
        #'dwinformerTrainerV2_Loss_MCCnoBG': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_Loss_DicewithBG': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_Loss_Dice_LR1en3': ["dwinformerPlansv2.1"],
        'dwinformerTrainerV2_Loss_Dice': ["dwinformerPlans", "dwinformerPlansv2.1"],
        'dwinformerTrainerV2_Loss_DicewithBG_LR1en3': ["dwinformerPlansv2.1"],
        # 'dwinformerTrainerV2_fp32': ["dwinformerPlansv2.1"],
        # 'dwinformerTrainerV2_fp32': ["dwinformerPlansv2.1"],
        # 'dwinformerTrainerV2_fp32': ["dwinformerPlansv2.1"],
        # 'dwinformerTrainerV2_fp32': ["dwinformerPlansv2.1"],
        # 'dwinformerTrainerV2_fp32': ["dwinformerPlansv2.1"],

    }

    trainers = ['dwinformerTrainer'] + ['dwinformerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'dwinformerTrainerNewCandidate24_2',
        'dwinformerTrainerNewCandidate24_3',
        'dwinformerTrainerNewCandidate26_2',
        'dwinformerTrainerNewCandidate27_2',
        'dwinformerTrainerNewCandidate23_always3DDA',
        'dwinformerTrainerNewCandidate23_corrInit',
        'dwinformerTrainerNewCandidate23_noOversampling',
        'dwinformerTrainerNewCandidate23_softDS',
        'dwinformerTrainerNewCandidate23_softDS2',
        'dwinformerTrainerNewCandidate23_softDS3',
        'dwinformerTrainerNewCandidate23_softDS4',
        'dwinformerTrainerNewCandidate23_2_fp16',
        'dwinformerTrainerNewCandidate23_2',
        'dwinformerTrainerVer2',
        'dwinformerTrainerV2_2',
        'dwinformerTrainerV2_3',
        'dwinformerTrainerV2_3_CE_GDL',
        'dwinformerTrainerV2_3_dcTopk10',
        'dwinformerTrainerV2_3_dcTopk20',
        'dwinformerTrainerV2_3_fp16',
        'dwinformerTrainerV2_3_softDS4',
        'dwinformerTrainerV2_3_softDS4_clean',
        'dwinformerTrainerV2_3_softDS4_clean_improvedDA',
        'dwinformerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'dwinformerTrainerV2_3_softDS4_radam',
        'dwinformerTrainerV2_3_softDS4_radam_lowerLR',

        'dwinformerTrainerV2_2_schedule',
        'dwinformerTrainerV2_2_schedule2',
        'dwinformerTrainerV2_2_clean',
        'dwinformerTrainerV2_2_clean_improvedDA_newElDef',

        'dwinformerTrainerV2_2_fixes', # running
        'dwinformerTrainerV2_BN', # running
        'dwinformerTrainerV2_noDeepSupervision', # running
        'dwinformerTrainerV2_softDeepSupervision', # running
        'dwinformerTrainerV2_noDataAugmentation', # running
        'dwinformerTrainerV2_Loss_CE', # running
        'dwinformerTrainerV2_Loss_CEGDL',
        'dwinformerTrainerV2_Loss_Dice',
        'dwinformerTrainerV2_Loss_DiceTopK10',
        'dwinformerTrainerV2_Loss_TopK10',
        'dwinformerTrainerV2_Adam', # running
        'dwinformerTrainerV2_Adam_dwinformerTrainerlr', # running
        'dwinformerTrainerV2_SGD_ReduceOnPlateau', # running
        'dwinformerTrainerV2_SGD_lr1en1', # running
        'dwinformerTrainerV2_SGD_lr1en3', # running
        'dwinformerTrainerV2_fixedNonlin', # running
        'dwinformerTrainerV2_GeLU', # running
        'dwinformerTrainerV2_3ConvPerStage',
        'dwinformerTrainerV2_NoNormalization',
        'dwinformerTrainerV2_Adam_ReduceOnPlateau',
        'dwinformerTrainerV2_fp16',
        'dwinformerTrainerV2', # see overwrite_plans
        'dwinformerTrainerV2_noMirroring',
        'dwinformerTrainerV2_momentum09',
        'dwinformerTrainerV2_momentum095',
        'dwinformerTrainerV2_momentum098',
        'dwinformerTrainerV2_warmup',
        'dwinformerTrainerV2_Loss_Dice_LR1en3',
        'dwinformerTrainerV2_NoNormalization_lr1en3',
        'dwinformerTrainerV2_Loss_Dice_squared',
        'dwinformerTrainerV2_newElDef',
        'dwinformerTrainerV2_fp32',
        'dwinformerTrainerV2_cycleAtEnd',
        'dwinformerTrainerV2_reduceMomentumDuringTraining',
        'dwinformerTrainerV2_graduallyTransitionFromCEToDice',
        'dwinformerTrainerV2_insaneDA',
        'dwinformerTrainerV2_independentScalePerAxis',
        'dwinformerTrainerV2_Mish',
        'dwinformerTrainerV2_Ranger_lr3en4',
        'dwinformerTrainerV2_cycleAtEnd2',
        'dwinformerTrainerV2_GN',
        'dwinformerTrainerV2_DP',
        'dwinformerTrainerV2_FRN',
        'dwinformerTrainerV2_resample33',
        'dwinformerTrainerV2_O2',
        'dwinformerTrainerV2_ResencUNet',
        'dwinformerTrainerV2_DA2',
        'dwinformerTrainerV2_allConv3x3',
        'dwinformerTrainerV2_ForceBD',
        'dwinformerTrainerV2_ForceSD',
        'dwinformerTrainerV2_ReLU',
        'dwinformerTrainerV2_LReLU_slope_2en1',
        'dwinformerTrainerV2_lReLU_convReLUIN',
        'dwinformerTrainerV2_ReLU_biasInSegOutput',
        'dwinformerTrainerV2_ReLU_convReLUIN',
        'dwinformerTrainerV2_lReLU_biasInSegOutput',
        'dwinformerTrainerV2_Loss_DicewithBG_LR1en3',
        #'dwinformerTrainerV2_Loss_MCCnoBG',
        'dwinformerTrainerV2_Loss_DicewithBG',
        # 'dwinformerTrainerV2_Loss_Dice_LR1en3',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
        # 'dwinformerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
