def transactions(x):
    for key,value in x.items():
        closing=x["main"]+x["deposit"]-x["withdraw"]
    return closing
bank={}
customerno=1
customer=input(f"Customer{customerno} Opening balance of account: ")
while customer!="":
    bank[f"Customer {customerno}"]={"main": float(customer), "deposit": 0, "withdraw": 0}
    customerno+=1
    customer=input(f"Customer{customerno} Opening balance of account: ")
print(bank)
for key,values in bank.items():
    x=int(input(f"{key} Amount deposited or withdrew (negative value) "))
    try:
        while x!=0:
            if x>0:
                bank[key]["deposit"]+=x
            elif x<0:
                bank[key]["withdraw"]+=abs(x)
            x=int(input(f"{key} Amount deposited or withdrew (negative value) "))
    except:
        print("Invalid value! Try entering an integer.")
        x=int(input(f"{key} Amount deposited or withdrew (negative value) "))
for key,value in bank.items():
    print(f"{key}:")
    print(f"Opeing balance: {bank[key]["main"]}")
    print(f"Deposits: {bank[key]["deposit"]}")
    print(f"Withdrawals: {bank[key]["withdraw"]}")
    print(f"Closing balance: {transactions(bank[key])}")
    print()