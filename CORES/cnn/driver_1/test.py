#! /usr/bin/python3 

import numpy as np
import torch
import unittest
import copy
from easydict import EasyDict
import os


from cores_cnn_driver import (
    parse_and_validate_parameters,
    set_rng,
    build_network,
    build_datasets,
    base_parameters,
    train
)

from steves_utils.CORES.utils import (
    ALL_NODES,
    ALL_NODES_MINIMUM_1000_EXAMPLES,
    node_name_to_id
)

base_parameters = {}
base_parameters["experiment_name"] = "MANUAL CORES CNN"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"

base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337
base_parameters["desired_classes"] = ALL_NODES

base_parameters["source_domains"] = [1]
base_parameters["target_domains"] = [2,3,4,5]

base_parameters["num_examples_per_class_per_domain"]=100

base_parameters["batch_size"]=128

base_parameters["n_epoch"] = 25

base_parameters["patience"] = 10

base_parameters["criteria_for_best"] = "target"
base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False


base_parameters["x_net"] =     [
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 256]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*256, "out_features": 256}}, # 80 units per IQ pair
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm1d", "kargs": {"num_features":256}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": len(base_parameters["desired_classes"])}},
]

base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["RESULTS_DIR"] = "./results"
base_parameters["EXPERIMENT_JSON_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "experiment.json")
base_parameters["LOSS_CURVE_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "loss.png")
base_parameters["BEST_MODEL_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "best_model.pth")

def prep_datasets(p:EasyDict)->dict:
    """
    {
        "source": {
            "original": {"train":<data>, "val":<data>, "test":<data>}
            "processed": {"train":<data>, "val":<data>, "test":<data>}
        },
        "target": {
            "original": {"train":<data>, "val":<data>, "test":<data>}
            "processed": {"train":<data>, "val":<data>, "test":<data>}
        },
    }
    """
    torch.set_default_dtype(torch.float64)
    set_rng(p)
    datasets = build_datasets(p)

    return datasets

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

class Test_Datasets(unittest.TestCase):
    # @unittest.skip
    def test_correct_domains(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        for source, target in [
            ((1,2),(3,4,5)),
            ((1,),(2,3,4,5)),
            ((1,2,3,4,5),(1,2,3,4,5)),
        ]:

            params.source_domains = source
            params.target_domains = target

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            for ds in (datasets.source.original.values()):
                seen_domains = set()
                for x,y,u in ds:
                    seen_domains.add(u)
                self.assertEqual(
                    seen_domains, set(params.source_domains)
                )

            # target
            for ds in (datasets.target.original.values()):
                seen_domains = set()
                for x,y,u in ds:
                    seen_domains.add(u)
                self.assertEqual(
                    seen_domains, set(params.target_domains)
                )

    # @unittest.skip
    def test_correct_labels(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        for desired_classes in [
            ALL_NODES_MINIMUM_1000_EXAMPLES,
            ALL_NODES,
            list(set(ALL_NODES)-set(ALL_NODES_MINIMUM_1000_EXAMPLES))
        ]:

            params.desired_classes = desired_classes

            classes_as_ids = [node_name_to_id(y) for y in params.desired_classes]

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            for ds in (datasets.source.original.values()):
                seen_classes = set()
                for x,y,u in ds:
                    seen_classes.add(y)
                self.assertEqual(
                    seen_classes, set(classes_as_ids)
                )

            # target
            for ds in (datasets.target.original.values()):
                seen_classes = set()
                for x,y,u in ds:
                    seen_classes.add(y)
                self.assertEqual(
                    seen_classes, set(classes_as_ids)
                )

    # @unittest.skip
    def test_correct_example_count(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES


        for num_examples_per_class_per_domain in [
            100,
            200,
            1000
        ]:

            params.num_examples_per_class_per_domain = num_examples_per_class_per_domain

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            examples_by_domain = {}
            for ds in (datasets.source.original.values()):
                for x,y,u in ds:                   
                    if u not in examples_by_domain:
                        examples_by_domain[u] = set()
                    examples_by_domain[u].add(numpy_to_hash(x))
            

            for u, hashes in examples_by_domain.items():
                self.assertGreaterEqual(
                    len(hashes) / ( num_examples_per_class_per_domain * len(params.desired_classes) ),
                    0.9
                )
                self.assertLessEqual(
                    len(hashes) / ( num_examples_per_class_per_domain * len(params.desired_classes) ),
                    1.1
                )


            # target
            examples_by_domain = {}
            for ds in (datasets.target.original.values()):
                for x,y,u in ds:                   
                    if u not in examples_by_domain:
                        examples_by_domain[u] = set()
                    examples_by_domain[u].add(numpy_to_hash(x))
            

            for u, hashes in examples_by_domain.items():
                self.assertGreaterEqual(
                    len(hashes) / ( num_examples_per_class_per_domain * len(params.desired_classes) ),
                    0.9
                )
                self.assertLessEqual(
                    len(hashes) / ( num_examples_per_class_per_domain * len(params.desired_classes) ),
                    1.1
                )


    # @unittest.skip
    def test_sets_disjoint(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES

        for num_examples_per_class_per_domain in [
            100,
            500,
            1000,
        ]:
            params.num_examples_per_class_per_domain = num_examples_per_class_per_domain



            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            train_hashes = set()
            for x,y,u in datasets.source.original.train:                   
                train_hashes.add(numpy_to_hash(x))

            val_hashes = set()
            for x,y,u in datasets.source.original.val:                   
                train_hashes.add(numpy_to_hash(x))

            test_hashes = set()
            for x,y,u in datasets.source.original.test:                   
                train_hashes.add(numpy_to_hash(x))
            

            self.assertEqual( len(train_hashes.intersection(val_hashes)),  0 )
            self.assertEqual( len(train_hashes.intersection(test_hashes)), 0 )
            self.assertEqual( len(val_hashes.intersection(test_hashes)),   0 )


            # target
            train_hashes = set()
            for x,y,u in datasets.target.original.train:                   
                train_hashes.add(numpy_to_hash(x))

            val_hashes = set()
            for x,y,u in datasets.target.original.val:                   
                train_hashes.add(numpy_to_hash(x))

            test_hashes = set()
            for x,y,u in datasets.target.original.test:                   
                train_hashes.add(numpy_to_hash(x))
            

            self.assertEqual( len(train_hashes.intersection(val_hashes)),  0 )
            self.assertEqual( len(train_hashes.intersection(test_hashes)), 0 )
            self.assertEqual( len(val_hashes.intersection(test_hashes)),   0 )

    # @unittest.skip
    def test_train_randomizes_episodes_val_and_test_dont(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES

        NUM_ITERATIONS = 5

        for num_examples_per_class_per_domain  in [
            100,
            200,
            500
        ]:
            params.num_examples_per_class_per_domain = num_examples_per_class_per_domain

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            train = set()
            val   = set()
            test  = set()

            for _ in range(NUM_ITERATIONS):
                # source
                train_hashes = []
                for x,y in datasets.source.processed.train:                   
                    for h in [numpy_to_hash(x.numpy()) for x in x]: train_hashes.append(h)

                val_hashes = []
                for x,y in datasets.source.processed.val:                   
                    for h in [numpy_to_hash(x.numpy()) for x in x]: train_hashes.append(h)

                test_hashes = []
                for x,y in datasets.source.processed.test:                   
                    for h in [numpy_to_hash(x.numpy()) for x in x]: train_hashes.append(h)
                
                train.add(hash(tuple(train_hashes)))
                val.add(hash(tuple(val_hashes)))
                test.add(hash(tuple(test_hashes)))
            
            self.assertEqual(len(train), NUM_ITERATIONS)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(test), 1)


            train = set()
            val   = set()
            test  = set()

            for _ in range(NUM_ITERATIONS):
                # target
                train_hashes = []
                for x,y in datasets.target.processed.train:                   
                    for h in [numpy_to_hash(x.numpy()) for x in x]: train_hashes.append(h)

                val_hashes = []
                for x,y in datasets.target.processed.val:                   
                    for h in [numpy_to_hash(x.numpy()) for x in x]: train_hashes.append(h)

                test_hashes = []
                for x,y in datasets.target.processed.test:                   
                    for h in [numpy_to_hash(x.numpy()) for x in x]: train_hashes.append(h)
                
                train.add(hash(tuple(train_hashes)))
                val.add(hash(tuple(val_hashes)))
                test.add(hash(tuple(test_hashes)))
            
            self.assertEqual(len(train), NUM_ITERATIONS)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(test), 1)


    # @unittest.skip
    def test_iterator_changes_permutation(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES

        source_train = set()
        source_val = set()
        source_test = set()

        target_val = set()
        target_test = set()
        target_train = set()

        combos = [
            (1337, 420),
            (54321, 420),
            (12332546, 420),
        ]

        for seed, dataset_seed  in combos: 
            params.num_examples_per_class_per_domain = 100
            
            params.seed = seed
            params.dataset_seed = dataset_seed


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)

            # source
            train_hashes = []
            for x,y in datasets.source.processed.train:                   
                train_hashes.append(numpy_to_hash(x.numpy()))

            val_hashes = []
            for x,y in datasets.source.processed.val:
                val_hashes.append(numpy_to_hash(x.numpy()))

            test_hashes = []
            for x,y in datasets.source.processed.test:
                test_hashes.append(numpy_to_hash(x.numpy()))
            
            source_train.add(hash(tuple(train_hashes)))
            source_val.add(hash(tuple(val_hashes)))
            source_test.add(hash(tuple(test_hashes)))

            # target
            train_hashes = []
            for x,y in datasets.target.processed.train:                   
                train_hashes.append(numpy_to_hash(x.numpy()))

            val_hashes = []
            for x,y in datasets.target.processed.val:
                val_hashes.append(numpy_to_hash(x.numpy()))

            test_hashes = []
            for x,y in datasets.target.processed.test:
                test_hashes.append(numpy_to_hash(x.numpy()))
            
            target_train.add(hash(tuple(train_hashes)))
            target_val.add(hash(tuple(val_hashes)))
            target_test.add(hash(tuple(test_hashes)))


        self.assertEqual(  len(source_train), len(combos)  )
        self.assertEqual(  len(source_val), len(combos)    )
        self.assertEqual(  len(source_test), len(combos)   )
        self.assertEqual(  len(target_val), len(combos)    )
        self.assertEqual(  len(target_test), len(combos)   )
        self.assertEqual(  len(target_train), len(combos)  )


    # @unittest.skip
    def test_dataset_seed(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES

        source_train = set()
        source_val = set()
        source_test = set()

        target_val = set()
        target_test = set()
        target_train = set()

        combos = [
            (1337, 420),
            (54321, 420),
            (12332546, 420),
        ]

        for seed, dataset_seed  in combos:
            params.num_examples_per_class_per_domain = 100
            
            params.seed = seed
            params.dataset_seed = dataset_seed


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)

            # source
            train_hashes = []
            for x,y,u in datasets.source.original.train:                   
                train_hashes.append(numpy_to_hash(x))

            val_hashes = []
            for x,y,u in datasets.source.original.val:
                val_hashes.append(numpy_to_hash(x))

            test_hashes = []
            for x,y,u in datasets.source.original.test:
                test_hashes.append(numpy_to_hash(x))
            
            source_train.add(hash(tuple(train_hashes)))
            source_val.add(hash(tuple(val_hashes)))
            source_test.add(hash(tuple(test_hashes)))

            # target
            train_hashes = []
            for x,y,u in datasets.target.original.train:                   
                train_hashes.append(numpy_to_hash(x))

            val_hashes = []
            for x,y,u in datasets.target.original.val:
                val_hashes.append(numpy_to_hash(x))

            test_hashes = []
            for x,y,u in datasets.target.original.test:
                test_hashes.append(numpy_to_hash(x))
            
            target_train.add(hash(tuple(train_hashes)))
            target_val.add(hash(tuple(val_hashes)))
            target_test.add(hash(tuple(test_hashes)))


        self.assertEqual(  len(source_train),  1)
        self.assertEqual(  len(source_val),  1)
        self.assertEqual(  len(source_test),  1)
        self.assertEqual(  len(target_val),  1)
        self.assertEqual(  len(target_test),  1)
        self.assertEqual(  len(target_train),  1)

    # @unittest.skip
    def test_reproducability(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES

        NUM_ITERATIONS = 3

        all_hashes = set()

        params.num_examples_per_class_per_domain = 1000


        for _ in range(NUM_ITERATIONS):
            hashes = []

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            for ds in datasets.source.processed.values():
                for x,y in ds:
                    hashes.append(numpy_to_hash(x.numpy()))

            for ds in datasets.source.processed.values():
                for x,y in ds:
                    hashes.append(numpy_to_hash(x.numpy()))
            

            all_hashes.add(
                hash(tuple(hashes))
            )

        self.assertEqual(
            len(all_hashes), 1
        )
        print(all_hashes)


    # @unittest.skip
    def test_splits(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes = ALL_NODES_MINIMUM_1000_EXAMPLES

        for num_examples_per_class_per_domain in [
            100,
            200,
            500,
        ]:
            params.num_examples_per_class_per_domain = num_examples_per_class_per_domain

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            train_hashes = set()
            for X,Y in datasets.source.processed.train:                   
                for h in [numpy_to_hash(x.numpy()) for x in X]: train_hashes.add(h)

            val_hashes = set()
            for X,Y in datasets.source.processed.val:                   
                for h in [numpy_to_hash(x.numpy()) for x in X]: val_hashes.add(h)

            test_hashes = set()
            for X,Y in datasets.source.processed.test:                   
                for h in [numpy_to_hash(x.numpy()) for x in X]: test_hashes.add(h)
            
            total = len(train_hashes) + len(val_hashes) + len(test_hashes)
            self.assertAlmostEqual( len(train_hashes) / total, 0.7, places=1)
            self.assertAlmostEqual( len(val_hashes) / total, 0.15, places=1)
            self.assertAlmostEqual( len(test_hashes) / total, 0.15, places=1)


            # target
            train_hashes = set()
            for X,Y in datasets.target.processed.train:                   
                for h in [numpy_to_hash(x.numpy()) for x in X]: train_hashes.add(h)

            val_hashes = set()
            for X,Y in datasets.target.processed.val:                   
                for h in [numpy_to_hash(x.numpy()) for x in X]: val_hashes.add(h)

            test_hashes = set()
            for X,Y in datasets.target.processed.test:                   
                for h in [numpy_to_hash(x.numpy()) for x in X]: test_hashes.add(h)
            
            total = len(train_hashes) + len(val_hashes) + len(test_hashes)
            self.assertAlmostEqual( len(train_hashes) / total, 0.7, places=1)
            self.assertAlmostEqual( len(val_hashes) / total, 0.15, places=1)
            self.assertAlmostEqual( len(test_hashes) / total, 0.15, places=1)


import sys
if len(sys.argv) > 1:
    Test_Datasets().test_reproducability()
else:
    unittest.main()