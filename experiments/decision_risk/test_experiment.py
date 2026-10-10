"""Scientific contract checks; synthetic signals here are tests, not evidence."""
import unittest

import numpy as np

from run import estimate_features, metrics, terms
from selector import select


class IdentityTests(unittest.TestCase):
    def test_known_constructive_interaction_changes_threshold_decision(self):
        x=np.array([[[1.,-1.,1.,-1.],[2.,-2.,2.,-2.]]])
        a=np.array([[[1.,-1.,1.,-1.],[0.,0.,0.,0.]]])
        mixed,px,pa=terms(np.eye(2),x,a,np.array([[1.]]))
        np.testing.assert_allclose(px,[[1.,4.]])
        np.testing.assert_allclose(pa,[[1.,0.]])
        np.testing.assert_allclose(mixed,[[[[4.,4.]]]])
        # A frozen threshold on first-component power: clean is left (0),
        # full is right (1), no interaction is left (0).
        self.assertFalse(px[0,0]>3)
        self.assertTrue(mixed[0,0,0,0]>3)
        self.assertFalse((px+pa)[0,0]>3)

    def test_orthogonal_time_series_have_no_cross_contribution(self):
        x=np.array([[[1.,-1.,1.,-1.],[2.,-2.,2.,-2.]]])
        a=np.array([[[1.,1.,-1.,-1.],[1.,1.,-1.,-1.]]])
        p,px,pa=terms(np.eye(2),x,a,np.array([[.7]]))
        np.testing.assert_allclose(p[0,0,0],px[0]+.49*pa[0])
        np.testing.assert_allclose(p[0,0,0],np.mean((x[0]+.7*a[0])**2,axis=-1))

    def test_zero_contamination_is_exact_clean_control(self):
        rng=np.random.default_rng(72)
        x=rng.normal(size=(4,3,23));a=rng.normal(size=(5,3,23));w=rng.normal(size=(2,3))
        p,px,_=terms(w,x,a,np.zeros((5,2)))
        np.testing.assert_allclose(p,np.broadcast_to(px[:,None,None],p.shape))


class SelectionTests(unittest.TestCase):
    def test_decoder_agreement_never_changes_command(self):
        original=np.array([1.,-2.,1.,-2.]);augmented=np.array([2.,-1.,-1.,1.])
        chosen,use=select(original,augmented,np.zeros((4,3)),1.)
        np.testing.assert_array_equal(chosen>0,[True,False,False,True])
        np.testing.assert_array_equal(use,[False,False,True,True])

    def test_constant_original_choice_preserves_all_commands(self):
        original=np.array([1.,-2.]);augmented=-original
        chosen,use=select(original,augmented,np.zeros((2,3)),0.)
        np.testing.assert_array_equal(chosen,original)
        self.assertFalse(use.any())

    def test_equal_coverage_counts_and_harm_denominator(self):
        m=metrics(np.array([5.,4.,3.,2.]),np.array([False,True,True,False]),
            np.array([False,True,False,False]),np.array([True,True,False,True]),.5)
        self.assertEqual(m['accepted'],2)
        self.assertEqual(m['risk'],.5)
        self.assertEqual(m['harm_given_clean_correct'],.5)
        self.assertEqual(m['correct_rejected'],1)

    def test_fixed_threshold_does_not_force_test_coverage(self):
        m=metrics(np.array([5.,4.,3.,2.]),np.array([False,True,True,False]),
            np.array([False,True,False,False]),np.array([True,True,False,True]),.9,threshold=4.5)
        self.assertEqual(m['actual_coverage'],.25)
        self.assertEqual(m['risk'],0.)

    def test_overestimated_artifact_is_exposed_as_clipping(self):
        class LDA:
            coef_=np.array([[1.,-1.]])
            intercept_=np.array([0.])
        f,raw,pa=estimate_features(np.array([[1.,2.]]),np.array([[100.,100.]]),
            LDA(),np.ones(2),np.zeros(2),np.array([.01,.01]))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[0,-1],1.)
        self.assertGreater(float(pa.sum()),100.)


if __name__=='__main__':unittest.main()
