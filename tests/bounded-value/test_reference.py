import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from pokemon_bounded import reference as ref
from pokemon_learner.core import distribution


def fixture():
    rng = np.random.default_rng(6173)
    shapes = {'w': (323,32), 'b': (32,), 'pw': (32,4), 'pb': (4,), 'vw': (32,), 'vb': ()}
    params = {k: np.asarray(rng.normal(0, .08, s), np.float32) for k,s in shapes.items()}
    features = rng.normal(0, .3, (12,323)).astype(np.float32)
    mask = rng.random((12,4)) > .4; mask[:,0] = True
    valid = np.ones(12, bool); valid[-2:] = False
    actions = np.zeros(12, np.int32)
    logits, _ = ref.forward(params, features)
    old = distribution(logits, mask)[1][:,0] + np.linspace(-.5,.5,12,dtype=np.float32)
    adv = rng.normal(size=12).astype(np.float32)
    returns = np.linspace(0,1,12,dtype=np.float32)
    return params, features, mask, actions, old, adv, returns, valid


class ReferenceTests(unittest.TestCase):
    def test_stable_sigmoid(self):
        with np.errstate(over='raise', invalid='raise'):
            y = ref.sigmoid(np.array([-1000., -2., 0., 2., 1000.], np.float32))
        np.testing.assert_allclose(y, [0., .11920292, .5, .88079708, 1.], atol=1e-7)
        self.assertEqual(y.dtype, np.float32)

    def test_initial_half(self):
        p,x,*_ = fixture(); p['vw'].fill(0); p['vb'][...] = 0
        np.testing.assert_array_equal(ref.forward(p,x)[1], np.full(12,.5,np.float32))

    def test_all_parameter_directional_derivatives(self):
        p,*data = fixture(); metrics,grad = ref.loss_grad(p,*data)
        self.assertTrue(np.isfinite(metrics['loss']))
        rng = np.random.default_rng(321)
        for name in p:
            direction = np.asarray(rng.normal(size=p[name].shape),np.float32)
            direction /= max(float(np.linalg.norm(direction)),1e-8)
            epsilon = np.float32(.005)
            plus = {k:v.copy() for k,v in p.items()}; minus = {k:v.copy() for k,v in p.items()}
            plus[name] += epsilon*direction; minus[name] -= epsilon*direction
            actual = (ref.loss_grad(plus,*data)[0]['loss'] - ref.loss_grad(minus,*data)[0]['loss'])/(2*epsilon)
            expected = float(np.sum(grad[name]*direction))
            self.assertAlmostEqual(actual, expected, delta=8e-5, msg=name)
        self.assertGreater(float(np.linalg.norm(grad['vw'])),1e-4)
        self.assertGreater(float(np.linalg.norm(grad['w'])),1e-4)

    def test_padding_empty_and_immutability(self):
        args=list(fixture()); p,x,mask,actions,old,adv,ret,valid=args
        before={k:v.copy() for k,v in p.items()}
        expected=ref.loss_grad(*args)
        for a in (x,old,adv,ret): a[~valid]=np.nan
        mask[~valid]=False;actions[~valid]=-1
        actual=ref.loss_grad(*args)
        self.assertEqual(actual[0],expected[0])
        for k in p:
            np.testing.assert_array_equal(actual[1][k],expected[1][k])
            np.testing.assert_array_equal(p[k],before[k])
        valid[:]=False
        metrics,grad=ref.loss_grad(*args)
        self.assertTrue(all(v==0 for v in metrics.values()))
        for g in grad.values():np.testing.assert_array_equal(g,np.zeros_like(g))

    def test_floor_domain_and_target_rejection(self):
        args=list(fixture());p=args[0];p['pb'][0]=20
        with self.assertRaises(ValueError):ref.loss_grad(*args)
        args=list(fixture());args[6][0]=1.1
        with self.assertRaises(ValueError):ref.loss_grad(*args)


if __name__=='__main__':unittest.main()
