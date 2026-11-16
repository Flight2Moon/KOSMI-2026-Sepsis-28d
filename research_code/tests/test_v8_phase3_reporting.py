from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.clinical_evaluation_v8 import (
    calibration_slope_intercept,
    decision_curve,
    select_threshold_by_policy,
)
from los_baseline.exp2_combined_v8 import build_exp2_combined_from_prediction_files
from los_baseline.exp3_reporting_v8 import build_exp3_report_from_stage2_predictions
from los_baseline.matrix_reporting_v8 import collect_matrix_rows


def _binary_frame(y, p, three_class, start=0):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pred=(p>=0.5).astype(int)
    return pd.DataFrame({
        'source_row_index':np.arange(start,start+len(y)),
        'subject_id':np.arange(1000+start,1000+start+len(y)),
        'y_true':y,
        'three_class_outcome':np.asarray(three_class,dtype=int),
        'positive_probability':p,
        'predicted_at_validation_threshold':pred,
        'predicted_at_0_5':pred,
    })


def test_threshold_policies_are_deterministic():
    y=np.array([0,0,0,1,1,1])
    p=np.array([0.05,0.15,0.35,0.45,0.75,0.95])
    for policy in ['fixed_0_5','youden','f1','sensitivity_target']:
        first=select_threshold_by_policy(y,p,policy=policy,sensitivity_target=0.67)
        second=select_threshold_by_policy(y,p,policy=policy,sensitivity_target=0.67)
        assert first.threshold == second.threshold
        assert 0 <= first.threshold <= 1
        assert first.selected_on_split == 'validation'


def test_perfect_calibration_has_slope_near_one_and_intercept_near_zero():
    # Construct repeated Bernoulli groups whose observed frequencies match p.
    p=np.repeat(np.array([0.1,0.2,0.5,0.8,0.9]),100)
    y=np.concatenate([
        np.array([1]*10+[0]*90),
        np.array([1]*20+[0]*80),
        np.array([1]*50+[0]*50),
        np.array([1]*80+[0]*20),
        np.array([1]*90+[0]*10),
    ])
    out=calibration_slope_intercept(y,p)
    assert abs(out['calibration_slope']-1.0) < 0.08
    assert abs(out['calibration_intercept']) < 0.08


def test_decision_curve_contains_reference_strategies():
    y=np.array([0,0,1,1])
    p=np.array([0.1,0.3,0.7,0.9])
    frame=decision_curve(y,p,thresholds=np.array([0.2,0.5]))
    assert list(frame.columns) == [
        'threshold','net_benefit_model','net_benefit_treat_all','net_benefit_treat_none','flag_rate'
    ]
    assert len(frame)==2
    assert (frame['net_benefit_treat_none']==0).all()


def test_sampled_exp2_combined_ranks_macro_auprc_and_writes_outputs(tmp_path: Path):
    y3=np.array([0,0,1,1,2,2,0,1,2])
    stage1_y=(y3==1).astype(int)
    stage2_y=(y3==2).astype(int)
    # Stage1 good and weak; Stage2 good and weak.
    p1_good=np.array([.05,.10,.85,.90,.20,.25,.10,.75,.30])
    p1_weak=np.array([.30,.35,.55,.60,.40,.45,.35,.52,.48])
    p2_good=np.array([.10,.15,.20,.25,.85,.90,.20,.25,.80])
    p2_weak=np.array([.35,.40,.45,.50,.60,.62,.40,.45,.58])
    s1={}; s2={}
    for name,p in [('good',p1_good),('weak',p1_weak)]:
        path=tmp_path/f's1_{name}.csv.gz'; _binary_frame(stage1_y,p,y3).to_csv(path,index=False,compression='gzip'); s1[name]=path
    for name,p in [('good',p2_good),('weak',p2_weak)]:
        path=tmp_path/f's2_{name}.csv.gz'; _binary_frame(stage2_y,p,y3).to_csv(path,index=False,compression='gzip'); s2[name]=path
    out=tmp_path/'combined'
    manifest=build_exp2_combined_from_prediction_files(
        run_id='r',strategy_id='T3_smote',stage1_prediction_paths=s1,stage2_prediction_paths=s2,
        output_dir=out,primary_metric='macro_auprc'
    )
    top=json.loads((out/'top_pair.json').read_text())['top_pair']
    assert top['stage1_model']=='good'
    assert top['stage2_model']=='good'
    assert manifest['selection_metric']=='macro_auprc'
    assert (out/'top_pair_predictions.csv.gz').is_file()
    assert (out/'clinical_validation/calibration_slope_intercept_ovr.csv').is_file()


def test_exp3_report_reuses_stage2_without_training(tmp_path: Path):
    y=np.array([0,0,0,1,1,1])
    three=np.where(y==1,2,0)
    paths={}
    for name,p in [('a',np.array([.1,.2,.3,.7,.8,.9])),('b',np.array([.2,.3,.4,.6,.7,.8]))]:
        model=tmp_path/name/'ensemble'/'validation'; model.mkdir(parents=True)
        pred=model/'predictions.csv.gz'; _binary_frame(y,p,three).to_csv(pred,index=False,compression='gzip')
        (model/'metrics.json').write_text(json.dumps({'auprc':float(1.0 if name=='a' else .9),'auroc':float(1.0 if name=='a' else .9),'brier_score':.1}))
        paths[name]=pred
    out=tmp_path/'exp3'
    payload=build_exp3_report_from_stage2_predictions(run_id='r',strategy_id='T4_undersampling',stage2_prediction_paths=paths,output_dir=out)
    assert payload['training_performed'] is False
    assert payload['reuse_stage2_artifacts'] is True
    assert payload['validation_reference_model']=='a'
    assert (out/'model_summary_validation.csv').is_file()
    assert (out/'models/a/clinical_validation/decision_curve.csv').is_file()


def test_matrix_collector_supports_not_requested_t4(tmp_path: Path):
    runs=tmp_path/'runs'; runs.mkdir()
    run=runs/'r1'; (run/'EXP1_flat_3class').mkdir(parents=True)
    pd.DataFrame([{'model_name':'xgboost','validation_macro_auprc':.5,'validation_macro_auroc':.7}]).to_csv(run/'EXP1_flat_3class/model_summary_validation.csv',index=False)
    manifest=[
        {'outcome_horizon_days':28,'strategy':'T1_baseline','enabled':True,'status':'completed','run_id':'r1'},
        {'outcome_horizon_days':28,'strategy':'T4_undersampling','enabled':False,'status':'not_requested','run_id':None},
    ]
    frame=collect_matrix_rows(runs_dir=runs,runs=manifest)
    assert len(frame)==1
    assert frame.iloc[0]['strategy']=='T1_baseline'
    assert int(frame.iloc[0]['outcome_horizon_days'])==28


def test_pipeline_stage2_materialization_scores_full_cohort(tmp_path: Path):
    from los_baseline.exp123_data import TaskDataBundle, TaskSplitData
    from los_baseline.pipeline_predictions_v8 import (
        materialize_member_ensemble_pipeline_predictions,
        materialize_standard_artifact_pipeline_predictions,
    )
    from los_baseline.task_spec import STAGE2_LONG_STAY_TASK
    from los_baseline.training_engine import train_and_evaluate_single_model

    def split_from_three(y3, start, name, survivors_only=False):
        y3=np.asarray(y3,dtype=int)
        mask=(y3!=1) if survivors_only else np.ones(len(y3),dtype=bool)
        selected=y3[mask]
        x=np.arange(len(y3),dtype=float)[mask]
        raw=pd.DataFrame({'f1':x,'f2':x*0.2+1.0})
        metadata=pd.DataFrame({
            'source_row_index':np.arange(start,start+len(y3))[mask],
            'subject_id':[f'p{v}' for v in np.arange(start,start+len(y3))[mask]],
            'three_class_outcome':selected,
        })
        labels=(selected==2).astype(int)
        return TaskSplitData(name,raw,labels,metadata)

    train_three=np.array([0]*18+[2]*8)
    valid_three=np.array([0,1,2,0,1,2,0,2,1,0,2,0])
    train=split_from_three(train_three,0,'train',survivors_only=True)
    survivor_valid=split_from_three(valid_three,100,'validation',survivors_only=True)
    full_valid=split_from_three(valid_three,100,'validation',survivors_only=False)
    bundle=TaskDataBundle(
        task=STAGE2_LONG_STAY_TASK,
        feature_names=('f1','f2'),
        splits={'train':train,'validation':survivor_valid},
        shared_split_hash='h',
    )
    project_config={
        'calibration':{'enabled':False},
        'data':{'patient_id_column':'subject_id'},
        'explainability':{},
    }
    model_dir=tmp_path/'member'
    train_and_evaluate_single_model(
        task_data=bundle,
        model_name='decision_tree',
        model_config={'parameters':{'max_depth':2},'training_policy':{'class_weight_mode':'none'}},
        output_dir=model_dir,
        seed=42,
        project_config=project_config,
    )
    standard=materialize_standard_artifact_pipeline_predictions(
        artifact_dir=model_dir/'artifact',
        full_validation=full_valid,
        task_kind='stage2',
        output_path=tmp_path/'standard.csv.gz',
    )
    standard_frame=pd.read_csv(standard)
    assert len(standard_frame)==len(valid_three)
    assert standard_frame['pipeline_inference_only'].all()
    assert standard_frame['three_class_outcome'].tolist()==valid_three.tolist()

    survivor_prediction=model_dir/'evaluation/validation/predictions.csv.gz'
    ensemble=materialize_member_ensemble_pipeline_predictions(
        member_dirs=[model_dir,model_dir],
        full_validation=full_valid,
        task_kind='stage2',
        survivor_validation_prediction_path=survivor_prediction,
        output_path=tmp_path/'ensemble.csv.gz',
    )
    ensemble_frame=pd.read_csv(ensemble)
    assert len(ensemble_frame)==len(valid_three)
    np.testing.assert_allclose(
        ensemble_frame['positive_probability'],
        standard_frame['positive_probability'],
        atol=1e-12,
    )


def test_matrix_collector_surfaces_t3_stage1_as_t2_reference(tmp_path: Path):
    runs=tmp_path/'runs'; runs.mkdir()
    t2=runs/'t2'; (t2/'EXP2_two_stage/stage1_mortality').mkdir(parents=True)
    pd.DataFrame([{'model_name':'xgboost','validation_auprc':.44,'validation_auroc':.81}]).to_csv(
        t2/'EXP2_two_stage/stage1_mortality/model_summary_validation.csv',index=False
    )
    t3=runs/'t3'; (t3/'TRACK3_smote').mkdir(parents=True)
    (t3/'TRACK3_smote/track3_manifest.json').write_text(json.dumps({'optimized_run_id':'t2'}))
    frame=collect_matrix_rows(runs_dir=runs,runs=[{
        'outcome_horizon_days':28,'strategy':'T3_smote','enabled':True,'status':'completed','run_id':'t3'
    }])
    row=frame[(frame['strategy']=='T3_smote') & (frame['experiment']=='EXP2_Stage1')].iloc[0]
    assert row['model_name']=='xgboost'
    assert row['source_strategy']=='T2_optuna_reference'
