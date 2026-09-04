import sys
sys.path.insert(0, "src")

from adapt.task import few_shot_examples, load_task, subsample_train


def test_load_task_shapes():
    task = load_task()
    assert len(task.labels) == 77
    assert len(task.train) == 10003
    assert len(task.test) == 3080


def test_subsample_is_stratified():
    task = load_task()
    sub = subsample_train(task, 0.05, seed=0)
    import collections
    counts = collections.Counter(sub["label"])
    # Every one of the 77 classes must still be present after cutting to 5%.
    assert len(counts) == 77, f"lost classes: only {len(counts)}/77 remain"
    assert 450 <= len(sub) <= 550


def test_subsample_full_fraction_is_identity():
    task = load_task()
    assert len(subsample_train(task, 1.0, seed=0)) == len(task.train)


def test_few_shot_examples_are_distinct_classes():
    task = load_task()
    ex = few_shot_examples(task, 4, seed=0)
    assert len(ex) == 4
    labels_used = {label for _, label in ex}
    assert len(labels_used) == 4, "few-shot examples should span distinct intents"
