import copy
import unittest
from pokemon_gradient_audit.seal import validate_records
from pokemon_gradient_audit.campaign import PLAN
from pokemon_diversity.common import read


class SealTests(unittest.TestCase):
    def test_completion_and_four_corruption_negatives(self):
        plan=read(PLAN)
        manifest={'sources':{'checked.py':'sha'},'runtime':{'backend':'cpu'}}
        report={'status':'passed','plan':plan,'counts':plan['expected_counts'],'input_hash':'input'}
        verify={'status':'passed','cohorts':18,'input_hash':'input'}
        qualify={'status':'passed','tests_passed':13,**manifest}
        exits=[{'exit_code':0}]
        good=[report,verify,qualify,manifest,exits]
        validate_records(*good)
        for which in range(4):
            bad=copy.deepcopy(good)
            if which==0:bad[0]['counts']['cohorts']=17
            if which==1:bad[1]['input_hash']='wrong'
            if which==2:bad[2]['sources']={'other.py':'sha'}
            if which==3:bad[4][0]['exit_code']=1
            with self.assertRaises(ValueError):validate_records(*bad)


if __name__=='__main__':unittest.main()
