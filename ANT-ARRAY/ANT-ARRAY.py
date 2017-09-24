import matplotlib.pyplot as plt
import math
#circle1 = plt.Circle((0, 0), 1, color='black',linewidth=1,fill=False)

fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
#strangth one point SP;
SP = (70,0)
c = plt.Circle(SP, 1, color='red')
ax.add_artist(c)
#circle center point
CP = (0,0)

#DST between SP and CP
def getDST():
    return math.sqrt((SP[0]-CP[0])**2 + (SP[1]-CP[1])**2) 

for i in range(1,130):
    for j in range(-20,20):
        CP = (0,j)
        c = plt.Circle(CP, i+getDST()%1, color='black',linewidth=0.1,fill=False)
        ax.add_artist(c)
        
ax.set_xlim((-100, 100))
ax.set_ylim((-120, 120))
plt.show()
