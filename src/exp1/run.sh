# python3 exp1/unified_eval.py --collected_folder exp1/unified_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 1
# python3 exp1/unified_eval.py --collected_folder exp1/unified_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 10
# python3 exp1/unified_eval.py --collected_folder exp1/unified_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 30
# python3 exp1/unified_eval.py --collected_folder exp1/unified_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 60
# python3 exp1/unified_eval.py --collected_folder exp1/unified_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 120

# python3 exp1/opa_eval.py --collected_folder exp1/opa_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 1
# python3 exp1/opa_eval.py --collected_folder exp1/opa_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 10
# python3 exp1/opa_eval.py --collected_folder exp1/opa_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 30
# python3 exp1/opa_eval.py --collected_folder exp1/opa_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 60
# python3 exp1/opa_eval.py --collected_folder exp1/opa_saved_trials_inspection/perturb_collection --num_perturbs 10 --max_adaptation_time_sec 120

# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 1 --is_expert
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 10 --is_expert
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 30 --is_expert
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 60 --is_expert
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 120 --is_expert

# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 1
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 10
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 30
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 60
# python3 exp1/online_eval.py --num_perturbs 10 --max_adaptation_time_sec 120

# for fname in exp1/ferl_saved_trials_inspection/eval_perturbs_1_time_1.0/; do
#     echo $fname
#     python3 exp1/ferl_eval.py --exp_folder $fname
# done



# for fname in exp1/opa_saved_trials_inspection/eval_perturbs_*/; do
#     echo $fname
#     python3 exp1/eval_saved_traj.py --trials_folder $fname --perturb_folder exp1/opa_saved_trials_inspection/perturb_collection
# done

# for fname in exp1/opa_saved_trials_inspection/eval_perturbs_*/; do
#     echo $fname
#     python3 exp1/eval_saved_traj.py --trials_folder $fname --perturb_folder exp1/opa_saved_trials_inspection/perturb_collection
# done

# for fname in exp1/unified_saved_trials_inspection/eval_perturbs_*/; do
#     echo $fname
#     python3 exp1/eval_saved_traj.py --trials_folder $fname --perturb_folder exp1/unified_saved_trials_inspection/perturb_collection
# done

# for fname in exp1/ferl_saved_trials_inspection_final/eval_perturbs_*/; do
#     echo $fname
#     python3 exp1/eval_saved_traj.py --trials_folder $fname --perturb_folder exp1/ferl_saved_trials_inspection2/perturb_collection
# done

# for fname in ferl_saved_trials_inspection2/eval_perturbs_*/; do
#     echo $fname
#     python3 ferl_eval.py --exp_folder $fname
# done


# for fname in exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_10*/; do
#     echo $fname
#     python3 exp1/eval_saved_traj.py --trials_folder $fname --perturb_folder exp1/unified_saved_trials_inspection/perturb_collection
# done

# for fname in exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_*/; do
#     echo $fname
#     python3 exp1/eval_saved_traj.py --trials_folder $fname --perturb_folder exp1/unified_saved_trials_inspection/perturb_collection
# done
