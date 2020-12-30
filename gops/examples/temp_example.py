"""

for dubug

"""

import os
import sys





def test(**kwargs):
    print(kwargs['obs_dim'])
    temp = kwargs['obs_dime']






if __name__ == "__main__":
    # kw = {'buffer_name':'replay_buffer','obs_dim':4, 'act_dim':3, 'max_size':1000 }
    # buffer = create_buffer(**kw)

    module_path = os.path.dirname(os.path.realpath(__file__))
    print(os.path.realpath(__file__))
    print(module_path)
    print(os.path.dirname(module_path))
    # add to path
    sys.path.append(module_path)
