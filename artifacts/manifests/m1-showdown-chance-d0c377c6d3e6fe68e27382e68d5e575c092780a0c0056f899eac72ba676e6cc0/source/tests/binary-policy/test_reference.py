"""Independent arithmetic/analytic oracle checks; no third-party test framework."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.reference import (dyadic_threshold, word_selects_rare,
    scalar_policy, selected_logp_derivative, MINIMUM_LEGAL_MASS)
from pokemon_binary.policy import parts, distribution_kernel, threshold_words


class ReferenceTests(unittest.TestCase):
    def test_hand_computable_words_and_validation(self):
        cases=[(np.float32(.5),1<<63),(np.float32(.25),1<<62),
               (MINIMUM_LEGAL_MASS,1<<40),
               (np.nextafter(np.float32(.5),np.float32(0)),(1<<63)-(1<<39)),
               (np.nextafter(MINIMUM_LEGAL_MASS,np.float32(1)),(1<<40)+(1<<17))]
        for p,t in cases:
            h,l=dyadic_threshold(p);self.assertEqual((h<<32)|l,t)
            for w in (t-1,t,t+1):self.assertEqual(word_selects_rare(w>>32,w&0xffffffff,p),w<t)
        for p in (.25,np.float64(.25),np.float32(np.nan),np.float32(0),np.float32(.6)):
            with self.assertRaises((TypeError,ValueError)):dyadic_threshold(p)
        for word in (True,np.uint32(1),-1,2**32):
            with self.assertRaises((TypeError,ValueError)):word_selects_rare(word,0,np.float32(.25))

    def test_numpy_policy_and_analytic_derivative_against_jax(self):
        # Include both signs, the nondifferentiable floor's declared branch, and tie.
        x=np.array([[d,0.] for d in [-100,-27,-17,-16.6,-16,-10,-1,0,1,10,16,16.6,17,27,100]],np.float32)
        mask=np.ones_like(x,bool)
        p,lp,ra,r=map(np.asarray,parts(x,mask))
        for i,row in enumerate(x):
            ref=scalar_policy(row,mask[i])
            np.testing.assert_allclose(p[i],ref.probabilities,atol=2e-7,rtol=2e-6)
            np.testing.assert_allclose(lp[i],ref.logp,atol=2e-6,rtol=2e-6)
            self.assertEqual(ra[i],ref.rare_index)
            self.assertEqual(tuple(map(int,threshold_words(jnp.array([r[i]]))[0])),dyadic_threshold(np.float32(r[i])))
            for action in (0,1):
                grad=np.asarray(jax.grad(lambda l: distribution_kernel(l,mask[i:i+1])[1][0,action])(row[None,:]))[0]
                np.testing.assert_allclose(grad,selected_logp_derivative(row,mask[i],action),atol=2e-7,rtol=3e-6)
        for mask in (np.array([True,False]),np.array([False,True]),np.array([False,False])):
            row=np.array([2,-1],np.float32);ref=scalar_policy(row,mask)
            a,b=distribution_kernel(row[None,:],mask[None,:])
            np.testing.assert_array_equal(a[0],ref.probabilities)
            np.testing.assert_array_equal(b[0],ref.logp)
        for row,mask in (([0,1],np.ones(2,bool)),(np.zeros(3,np.float32),np.ones(2,bool)),
                         (np.zeros(2,np.float32),np.ones(2,np.int32)),(np.array([np.nan,0],np.float32),np.ones(2,bool))):
            with self.assertRaises((TypeError,ValueError)):scalar_policy(row,mask)

    def test_analytic_finite_difference(self):
        for d in (-8,-2,0,2,8):
            for action in (0,1):
                x=np.array([d,0],np.float32);mask=np.ones(2,bool)
                g=selected_logp_derivative(x,mask,action)
                for col in (0,1):
                    plus=x.copy();minus=x.copy();plus[col]+=.01;minus[col]-=.01
                    fd=(float(scalar_policy(plus,mask).logp[action])-float(scalar_policy(minus,mask).logp[action]))/.02
                    self.assertAlmostEqual(float(g[col]),fd,delta=5e-5)
                self.assertEqual(float(g.sum()),0)

if __name__=='__main__':unittest.main()
