from RE11_gocard import gocard
class pensioner(gocard):
    def __init__(self,amount,discount):
        super().__init__(amount)
        self._d=discount
    def rides(self,amount):
        discounted=amount*(1-(self._d/100))
        self._x-=discounted
        self._transactions.append(discounted)
        return self._x,self._transactions
class vip(gocard):
    def __init__(self,amount,discount):
        super().__init__(amount)
        self._d=discount
        self._discountvip=False
        self._ridecount=0
    def rides(self,amount):
        discounted=amount*(1-(self._d/100))
        furtherdiscount=discounted*(1-(self._d/100))
        if self._discountvip:
            self.x-=furtherdiscount
            self._transactions.append(-furtherdiscount)
        else:
            self._x-=discounted
            self._transactions.append(-discounted)
        if self._ridecount%10==0:
            self._discountvip=True
        if self._ridecount%15==0:
            self._discountvip=False
            self._ridecount=0
class regular(gocard):
    def __init__(self,amount,discount):
        super().__init__(amount)
        self._d=discount
        self._ridecount=0
    def rides(self,amount):
        discounted=amount*(1-(self._d/100))
        if 10<self._ridecount<=15:
            self._x-=discounted
            self._transactions.append(-discounted)
        if self._ridecount==15:
            self._ridecount=0
        else:
            self._x-=amount
            self._transactions.append(-amount)