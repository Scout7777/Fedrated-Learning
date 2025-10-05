from datasets import load_dataset
import tenseal
from fontTools.misc.eexec import encrypt

context = tenseal.context(tenseal.SCHEME_TYPE.CKKS,
                      poly_modulus_degree=8192,
                      coeff_mod_bit_sizes=[60,40,40,60]
                          )
context.generate_galois_keys()
context.global_scale = 2**40

if context.is_private(): print('private!')
if context.is_public(): print('public!')

test1 = [1,2,3,4,5]
test2 = [1,1,1,1,1]
encrypted1 = tenseal.ckks_vector(context,test1)
encrypted2 = tenseal.ckks_vector(context,test2)
encrypted3 = encrypted2 + encrypted1
print(encrypted3.decrypt())


secret_key = context.secret_key()
context.make_context_public()

if context.is_private(): print('private!')
if context.is_public(): print('public!')
test1 = [1,2,3,4,5]
test2 = [1,1,1,1,1]
encrypted1 = tenseal.ckks_vector(context,test1)
encrypted2 = tenseal.ckks_vector(context,test2)


encrypted3 = encrypted2 + encrypted1
print('encrypted3:%s',encrypted3)

