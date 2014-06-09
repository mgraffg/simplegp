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
import numpy as np
from SimpleGP import VerifyOutput


def test_verify_output():
    v = VerifyOutput()
    np.random.seed(0)
    a = np.arange(1000)
    b = a*2.5 + 12 + np.random.normal(0, scale=1.0, size=a.shape[0])
    flag = v.verify(b[:-24], b[-24:])
    assert flag
    assert v.verify(b[:-190], b[-190:])
    try:
        v.verify(b[:-191], b[-191:])
        assert False
    except Exception:
        assert True
    b = 2.5*a**3 + 12 + np.random.normal(0, scale=1.0, size=a.shape[0])
    flag = v.verify(b[:-190], b[-190:])
    print v.value_ts, v.value, v.value/v.value_ts
    assert flag
