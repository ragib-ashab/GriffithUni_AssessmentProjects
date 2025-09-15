class gocard:
    """Keeps record of a go card.
    x is the balance on the card
    amount is the transactions taking place
    positive for topups and negatives for rides"""
    def __init__(self,x):
        self._x=x
        self._transactions=[x]
    def rides(self,amount):
        self._x-=amount
        self._transactions.append(-amount)
    def topup(self,amount):
        self._x+=amount
        self._transactions.append(amount)
    def balance(self):
        return self._x
    def statement(self):
        print("Statement")
        print(f"{"Event":15s}{"Amount($)":>11s}{"Balance($)":>11s}")
        opening=self._transactions[0]
        print(f"{"Opening":15s}{"":>11s}{opening:>11.2f}")
        for i in self._transactions[1:]:
            opening+=i
            if i<0:
                print(f"{"Ride":15s}{abs(i):>11.2f}{opening:>11.2f}")
            else:
                print(f"{"Topup":15s}{i:>11.2f}{opening:>11.2f}")
        print(f"{"Final balance":15s}{"":>11}{self._x:>11.2f}")