import pytest

import src.observers.subscribers as subscribers

class TestEarlyStoper:
    increasing1 = [1,2,3]
    increasing2 = [1,2,3,4,5,6]
    increasing3 = [100,1,0,4,5,6]

    descreasing1 = [3,2,1]
    descreasing2 = [3,100,4,3,2,1]

    mix1 = [1,100,2]
    mix2 = [2,100,1] 

    def new_stoper(self, lookback):
        return subscribers.EarlyStoper(lookback=lookback)

    def check_for_error(self, stoper, xs):
        for x in xs: stoper.update(x) 

    def test_inc_one(self):
        with pytest.raises(subscribers._EarlyStopperError):
            self.check_for_error(self.new_stoper(3), self.increasing1)

    def test_inc_two(self):
        with pytest.raises(subscribers._EarlyStopperError):
            self.check_for_error(self.new_stoper(3), self.increasing2)
    
    def test_inc_three(self):
         with pytest.raises(subscribers._EarlyStopperError):
            self.check_for_error(self.new_stoper(3), self.increasing3)

    def test_dec_one(self):
        self.check_for_error(self.new_stoper(3), self.descreasing1)
        assert True
    
    def test_dec_two(self):
        self.check_for_error(self.new_stoper(3), self.descreasing2)
        assert True

    def test_mix_one(self):
        self.check_for_error(self.new_stoper(3), self.mix1)
        assert True

    def test_mix_two(self):
        self.check_for_error(self.new_stoper(3), self.mix2)
        assert True