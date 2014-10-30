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
__version__ = '0.6.4'

from .simplega import SimpleGA, BestNotFound
from .simplegp import GP, GPwRestart, GPMAE
from .gppde import GPPDE
from .forest import GPForest, SubTreeXO, SubTreeXOPDE
from .classification import Classification, ClassificationPDE
from .time_series import TimeSeries
from .recursiveGP import RecursiveGP, RGP
from .eval import Eval
from .Rprop import RPROP
from utils import Pool, VerifyOutput


__all__ = [SimpleGA, TimeSeries, GP, GPPDE,
           GPForest, SubTreeXO, SubTreeXOPDE, Classification,
           RecursiveGP, RGP, GPwRestart, GPMAE,
           Eval, RPROP, VerifyOutput, ClassificationPDE,
           Pool, BestNotFound]




















