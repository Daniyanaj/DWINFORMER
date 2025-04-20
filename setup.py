from setuptools import setup, find_namespace_packages

setup(name='dwinformer',
      packages=find_namespace_packages(include=["dwinformer", "dwinformer.*"]),
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",dwinformer
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],dwinformer
      entry_points={
          'console_scripts': [
              'dwinformer_convert_decathlon_task = dwinformer.experiment_planning.dwinformer_convert_decathlon_task:main',
              'dwinformer_plan_and_preprocess = dwinformer.experiment_planning.dwinformer_plan_and_preprocess:main',
              'dwinformer_train = dwinformer.run.run_training:main',
              'dwinformer_predict = dwinformer.inference.predict_simple:main',
              'dwinformer_ensemble = dwinformer.inference.ensemble_predictions:main',
              'dwinformer_find_best_configuration = dwinformer.evaluation.model_selection.figure_out_what_to_submit:main',
              'dwinformer_print_available_pretrained_models = dwinformer.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'dwinformer_print_pretrained_model_info = dwinformer.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'dwinformer_download_pretrained_model = dwinformer.inference.pretrained_models.download_pretrained_model:download_by_name',
              'dwinformer_download_pretrained_model_by_url = dwinformer.inference.pretrained_models.download_pretrained_model:download_by_url',
              'dwinformer_determine_postprocessing = dwinformer.postprocessing.consolidate_postprocessing_simple:main',
              'dwinformer_export_model_to_zip = dwinformer.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'dwinformer_install_pretrained_model_from_zip = dwinformer.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'dwinformer_change_trainer_class = dwinformer.inference.change_trainer:main',
              'dwinformer_evaluate_folder = dwinformer.evaluation.evaluator:dwinformer_evaluate_folder',
              'dwinformer_plot_task_pngs = dwinformer.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      
      )
