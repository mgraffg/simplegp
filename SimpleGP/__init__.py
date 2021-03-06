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
__version__ = '0.11.5'

from .simplega import SimpleGA, BestNotFound
from .simplegp import GP, GPwRestart, GPMAE, GPS
from .generational import Generational, GenerationalPDE
from .gppde import GPPDE
from .forest import GPForest, GPForestPDE, SubTreeXO, SubTreeXOPDE
from .classification import Classification, ClassificationPDE
from .time_series import TimeSeries
from .recursiveGP import RecursiveGP, RGP
from .eval import Eval
from .Rprop import RPROP
from .elm import ELM, ELMPDE
from .PrGP import PrGP, GSGP
from .sparse_array import SparseArray, SparseEval
from .gppg import SparseGPPG
from .utils import Pool, VerifyOutput
from .egp import EGPS
from .bayes import Bayes, IBayes, AdaBayes
from .root import RootGP


__all__ = [SimpleGA, TimeSeries, GP, GPPDE,
           GPForest, SubTreeXO, SubTreeXOPDE, Classification,
           RecursiveGP, RGP, GPwRestart, GPMAE,
           Eval, RPROP, VerifyOutput, ClassificationPDE,
           Pool, BestNotFound, Generational, GenerationalPDE,
           ELM, ELMPDE, GPForestPDE, PrGP, GSGP, SparseEval,
           SparseArray, SparseGPPG, GPS, EGPS, Bayes, IBayes, AdaBayes, RootGP]




















