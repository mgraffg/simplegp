# Copyright 2013 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
__version__ = "0.2.6"

from .simplega import SimpleGA
from .simplegp import GP, GPRPropU, GPPDE, GPwRestart, GPMAE
from .forest import GPForest, SubTreeXO
from .classification import Classification, ClassificationPDE
from .time_series import TimeSeries, RTimeSeries
from .recursiveGP import RecursiveGP, RGP
from .eval import Eval
from .Rprop_mod import RPROP
from utils import Pool, VerifyOutput


__all__ = [SimpleGA, TimeSeries, GP, GPPDE,
           GPRPropU, GPForest, SubTreeXO, Classification,
           RecursiveGP, RGP, RTimeSeries, GPwRestart, GPMAE,
           Eval, RPROP, VerifyOutput, ClassificationPDE,
           Pool]




















