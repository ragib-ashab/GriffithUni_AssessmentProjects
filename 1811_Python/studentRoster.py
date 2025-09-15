def avgmrk(x):
    avgmrk=[]
    for student,courses in roster.items():
        totalmarks=0
        for course,mark in courses.items():
            totalmarks+=mark
        average=totalmarks/len(courses)
        avgmrk.append(average)
    return avgmrk
def avgcrs(x):
    avgcrs=[]
    crsmrk={}
    for student,courses in roster.items():
        for course,mark in courses.items():
            if course not in crsmrk:
                crsmrk[course]={"total": mark, "count": 1}
            else:
                crsmrk[course]["total"]+=mark
                crsmrk[course]["count"]+=1
    for course,data in crsmrk.items():
        average=data["total"]/data["count"]
        avgcrs.append(average)
    return avgcrs
roster={}
try:
    n=int(input("Number of students "))
    #For manual input and varying course for different students
    # for i in range(1,n+1):
    #     subno=int(input(f"Number of subjects for student {i} "))
    #     roster[f"student {i}"]={}
    #     for j in range(subno):
    #         name=input("Name of the subject: ")
    #         marks=int(input(f"Marks of the {name} for student {i}: "))
    #         roster[f"student {i}"][name]=marks
    for i in range(1,n+1):
            subject={}
            marks=input(f"Student {i} (courses 1-4): ").split()
            for j in range(len(marks)):
                marks[j]=float(marks[j])
                subject[f"course {j+1}"]=marks[j]
            roster[i]=subject
    print()
    print(roster)
    highestavg=max(avgmrk(roster))
    print(f"The highest average mark of students: {highestavg}")
    highestcrsavg=max(avgcrs(roster))
    print(f"The highest average mark of courses: {highestcrsavg}")
except:
            print("Invalid input! Try again.")
            n = int(input("Number of students you want to calculate for? (Enter 0 to exit): "))
print("Exiting program.")