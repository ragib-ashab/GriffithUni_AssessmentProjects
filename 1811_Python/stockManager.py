stock={}
print("Welcome to stock management app.")
xyz=input("Press: 1. To manage inventory 2. To manage stock 3. Buy items ")
while xyz!="0":
    if xyz=="1":
        stocker=input("What items you wish to store ").upper()
        while stocker!="":
            stock[stocker]={}
            stocker=input("What items you wish to store ").upper()
        print(stock)
        items=input("What items would u wish to add variants to? ").upper()
        while items!="":
            if items in stock:
                variant=input(f"What kind of {items} you want to add ").upper()
                while variant!="":
                    stock[items][variant]=0
                    variant=input(f"What kind of {items} you want to add ").upper()
                items=input("What items would u wish to add variants to? ").upper()
            else:
                print(f"{items} not in stock. ")
    if not stock:
        print("No inventory is set. Press 1 to set inventory")
    else:
        if xyz=="2":
            print(stock)
            stocking=input("What items would u wish to add stocks to? ").upper()
            while stocking!="":
                if stocking in stock:
                    x=input(f"What kind of {stocking} you want to add stocks to? ").upper()
                    while x!="":
                        if x in stock[stocking]:
                            amount=int(input(f"Amount of {x}{stocking} you want to add? "))
                            stock[stocking][x]=amount
                            x=input(f"What kind of {stocking} you want to add stocks to? ").upper()
                    stocking=input("What items would u wish to add variants to? ").upper()
                else:
                    print(f"{stocking} not in stock")
                    stocking=input("What items would u wish to add stocks to? ").upper()
        print(stock)
        if xyz=="3":
            reqitem=input("What item do you want to buy? ").upper()
            while reqitem!="":
                if reqitem in stock:
                    req=int(input(f"How many {reqitem} do you want to buy? "))
                    while req!=0:
                        choice=input(f"What kind of {reqitem}(colour/size) would you like? ").upper()
                        if choice in stock[reqitem]:
                            if stock[reqitem][choice]==0:
                                print(f"All {choice} {reqitem} are out of stock")
                            elif stock[reqitem][choice]<req:
                                yn=input(f"Only {req} are in {choice} {reqitem}. Do you want to buy all of them? ").upper()
                                if yn=="YES" or yn=="Y":
                                    req=stock[reqitem][choice]
                                    stock[reqitem][choice]=0
                                    print(f"{choice} {reqitem} needs to be restocked")
                                elif yn=="NO" or yn=="N":
                                    req=0
                            else:
                                stock[reqitem][choice]-=req
                        YN=input(f"Any other {reqitem}? ").upper()
                        if YN=="YES" or YN=="Y":
                            choice=input(f"What kind of {reqitem}(colour/size) would you like? ").upper()
                        elif YN=="N" or YN=="NO":
                            break
                reqitem=input("Any other item do you want to buy? ").upper()
            for item,variants in stock.items():
                for variant,inventory in variants.items():
                    print(f"Number of {variant} {items} in stock {inventory}")
    xyz=input("Press: 1. To manage inventory 2. To manage stock 3. Buy items ")
print(stock)