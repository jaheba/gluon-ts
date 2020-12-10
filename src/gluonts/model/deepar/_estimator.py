# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from enum import Enum
from typing import List, Optional

import numpy as np
from mxnet.gluon import HybridBlock
from pydantic import BaseModel

from gluonts.core import ty
from gluonts.core.component import DType, validated
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    Transformation,
    VstackFeatures,
    InstanceSampler,
)
from gluonts.transform.feature import (
    DummyValueImputation,
    MissingValueImputation,
)

from ._network import DeepARPredictionNetwork, DeepARTrainingNetwork


class DropoutcellType(str, Enum):
    ZoneoutCell = "ZoneoutCell"
    RNNZoneoutCell = "RNNZoneoutCell"
    VariationalDropoutCell = "VariationalDropoutCell"
    VariationalZoneoutCell = "VariationalZoneoutCell"


class DeepAREstimator(BaseModel, GluonEstimator):
    """
    Construct a DeepAR estimator.

    This implements an RNN-based model, close to the one described in
    [SFG17]_.

    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's DeepAR Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html>`_.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_layers
        Number of RNN layers (default: 2)
    num_cells
        Number of RNN cells for each layer (default: 40)
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    dropoutcell_type
        Type of dropout cells to use
        (available: 'ZoneoutCell', 'RNNZoneoutCell', 'VariationalDropoutCell' or 'VariationalZoneoutCell';
        default: 'ZoneoutCell')
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data
        (default: False)
    use_feat_static_cat
        Whether to use the ``feat_static_cat`` field from the data
        (default: False)
    use_feat_static_real
        Whether to use the ``feat_static_real`` field from the data
        (default: False)
    cardinality
        Number of values of each categorical feature.
        This must be set if ``use_feat_static_cat == True`` (default: None)
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput())
    scaling
        Whether to automatically scale the target values (default: true)
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100)
    imputation_method
        One of the methods from ImputationStrategy
    train_sampler
        Controls the sampling of windows during training.
    alpha
        The scaling coefficient of the activation regularization
    beta
        The scaling coefficient of the temporal activation regularization
    """

    class Config:
        arbitrary_types_allowed = True

    freq: str
    prediction_length: ty.PositiveInt
    alpha: ty.PositiveFloat0 = 0.0
    beta: ty.PositiveFloat0 = 0.0
    cardinality: List[ty.PositiveInt] = [1]
    cell_type: str = "lstm"
    context_length: Optional[ty.PositiveInt] = None
    distr_output: DistributionOutput = StudentTOutput()
    dropout_rate: ty.PositiveFloat0 = 0.1
    dropoutcell_type: DropoutcellType = DropoutcellType.ZoneoutCell
    dtype: DType = np.float32
    embedding_dimension: Optional[List[ty.PositiveInt]] = None
    imputation_method: Optional[MissingValueImputation] = None
    lags_seq: Optional[List[int]] = None
    num_cells: ty.PositiveInt = 40
    num_layers: ty.PositiveInt = 2
    num_parallel_samples: ty.PositiveInt = 100
    scaling: bool = True
    time_features: Optional[List[TimeFeature]] = None
    train_sampler: InstanceSampler = ExpectedNumInstanceSampler(1.0)
    trainer: Trainer = Trainer()
    use_feat_dynamic_real: bool = False
    use_feat_static_cat: bool = False
    use_feat_static_real: bool = False

    def __init__(self, **fields):
        BaseModel.__init__(self, **fields)

        if self.context_length is None:
            self.context_length = self.prediction_length

        if self.embedding_dimension is None:
            self.embedding_dimension = [
                min(50, (cat + 1) // 2) for cat in self.cardinality
            ]

        if self.lags_seq is None:
            self.lags_seq = get_lags_for_frequency(freq_str=self.freq)

        if self.time_features is None:
            self.time_features = time_features_from_frequency_str(self.freq)

        if self.imputation_method is None:
            self.imputation_method = DummyValueImputation(
                self.distr_output.value_in_support
            )

    @property
    def lead_time(self):
        return 0

    @property
    def history_length(self):
        return self.context_length + max(self.lags_seq)

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            "use_feat_static_cat": bool(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                    imputation_method=self.imputation_method,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                    dtype=self.dtype,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=self.train_sampler,
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                    dummy_value=self.distr_output.value_in_support,
                ),
            ]
        )

    def create_training_network(self) -> DeepARTrainingNetwork:
        return DeepARTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropoutcell_type=self.dropoutcell_type,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
            alpha=self.alpha,
            beta=self.beta,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = DeepARPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropoutcell_type=self.dropoutcell_type,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
        )
