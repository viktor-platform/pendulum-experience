from viktor import ViktorController
from viktor.core import (
    progress_message,
    File,
)

from viktor.parametrization import (
    ViktorParametrization,
    BooleanField,
    NumberField,
    LineBreak,
    Step,
    Text,
)

from pathlib import Path
from viktor.views import (
    ImageResult,
    ImageView,
    WebResult,
    WebView,

)

import matplotlib.animation as animation
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from gekko import GEKKO
import numpy as np
import math


def moonGrav(params, **kwargs):
    if params.doubleStep.gravity:
        grav = 1.62
    else:
        grav = 9.81
    return grav



class Parametrization(ViktorParametrization):
    singleStep = Step("Single Pendulum", views=['singlePendulum',  "whats_next"])
    singleStep.text1 = Text(
        """
## Welcome to the Pendulum Experience!

In this app, two examples of state-space control systems are displayed. As a user, you may change the weight and 
target time for the pendulum to reach the objective.  
In the second part of the pendulum experience, the challenge is to balance 
a double pendulum upright as fast as possible. 
        """
    )


    singleStep.targetTime = NumberField(
        "Target time",
        variant='slider',
        default=6.2,
        step=0.2,
        min=5.0,
        max=10.0,
        flex=50
    )

    singleStep.lb1 = LineBreak()
    singleStep.weight = NumberField(
        "weight",
        min=1,
        default=1,
        max=10,
        flex=50,
    )

    singleStep.text2 = Text(
        """
## About this app
The app is inspired and based on the code of  [Prof John Hedengren](https://www.linkedin.com/in/hedengren/) and the [Gekko Optimisation Suit](https://gekko.readthedocs.io/en/latest/).
The user interface is built in [VIKTOR](https://www.viktor.ai/) and the github repository can be found [here.](https://github.com/viktor-platform/pendulum-experience) 
        """
    )

    doubleStep = Step("Double Pendulum", views=['doublePendulum', "whats_next"])
    doubleStep.text3 = Text(
        """
# Double Pendulum
In this part of the app, you can change the weights, choose an end location
and even apply moon gravity!        
        """

    )
    doubleStep.massOne = NumberField(
        "Mass one",
        default=50,
        min=30,
        max=70,
        flex=50,
    )
    doubleStep.lb2 = LineBreak()
    doubleStep.massTwo = NumberField(
        "Mass two",
        default=50,
        min=30,
        max=70,
        flex=50,
    )
    doubleStep.lb3 = LineBreak()
    doubleStep.endx = NumberField(
        "final position",
        min=-1.5,
        max=1.5,
        step=0.1,
        default=0.0,
        flex=50,
    )

    doubleStep.lb4 = LineBreak()
    doubleStep.gravity = BooleanField(
        "Moon Gravity (1.62 m/s^2)",
        default=False,
        flex=50,
    )

    doubleStep.text4 = Text(
        """
## About this app
The app is inspired and based on the code of  [Prof John Hedengren](https://www.linkedin.com/in/hedengren/) and the [Gekko Optimisation Suit](https://gekko.readthedocs.io/en/latest/).
The user interface is built in [VIKTOR](https://www.viktor.ai/) and the github repository can be found [here.](https://github.com/viktor-platform/pendulum-experience) 
        """
    )

class Controller(ViktorController):
    label = 'My Entity Type'
    parametrization = Parametrization
    @ImageView("Single Pendulum", duration_guess=10)
    def singlePendulum(self, params, **kwargs):
        # result=generateModel(params)

        model = GEKKO()
        weight = params.singleStep.weight
        model.time = np.linspace(0,8,100)
        end_loc = int(100.0*params.singleStep.targetTime/8.0)

        #Parameters
        m1a = model.Param(value=10)
        m2a = model.Param(value=weight)
        final = np.zeros(len(model.time))
        for i in range(len(model.time)):
            if model.time[i] < 6.2:
                final[i] = 0
        else:
            final[i] = 1
        final = model.Param(value=final)

        #MV
        ua = model.Var(value=0)
        progress_message(message='building state space model...')
        #State Variables
        theta_a = model.Var(value=0)
        qa = model.Var(value=0)
        ya = model.Var(value=-1)
        va = model.Var(value=0)
        epsilon = model.Intermediate(m2a/(m1a+m2a))
        
        #define the state space
        model.Equation(ya.dt() == va)
        model.Equation(va.dt() == -epsilon*theta_a + ua)
        model.Equation(theta_a.dt() == qa)
        model.Equation(qa.dt() == theta_a -ua)
        
        #Definine the Objectives
        #Make all the state variables be zero at time >= 6.2

        progress_message(message='Defining objective equations...')

        model.Obj(final*ya**2)
        model.Obj(final*va**2)
        model.Obj(final*theta_a**2)
        model.Obj(final*qa**2)

        model.fix(ya,pos=end_loc,val=0.0)
        model.fix(va,pos=end_loc,val=0.0)
        model.fix(theta_a,pos=end_loc,val=0.0)
        model.fix(qa,pos=end_loc,val=0.0)
        #Try to minimize change of MV over all horizon
        model.Obj(0.001*ua**2)

        model.options.IMODE = 6 #MPC
        model.solve() #(disp=False)

        plt.figure(figsize=(12,10))
        x1 = ya.value
        y1 = np.zeros(len(model.time))

        x2 = 1*np.sin(theta_a.value)+x1
        x2b = 1.05*np.sin(theta_a.value)+x1
        y2 = 1*np.cos(theta_a.value)-y1
        y2b = 1.05*np.cos(theta_a.value)-y1

        fig = plt.figure(figsize=(8,6.4))
        ax = fig.add_subplot(111,autoscale_on=False,\
                            xlim=(-1.5,0.5),ylim=(-0.4,1.2))
        ax.set_xlabel('position')
        ax.get_yaxis().set_visible(False)

        crane_rail, = ax.plot([-1.5,0.5],[-0.2,-0.2],'k-',lw=4)
        start, = ax.plot([-1,-1],[-1.5,1.5],'k:',lw=2)
        objective, = ax.plot([0,0],[-0.5,1.5],'k:',lw=2)
        mass1, = ax.plot([],[],linestyle='None',marker='s',\
                        markersize=40,markeredgecolor='k',\
                        color='orange',markeredgewidth=2)
        mass2, = ax.plot([],[],linestyle='None',marker='o',\
                        markersize=20,markeredgecolor='k',\
                        color='orange',markeredgewidth=2)
        line, = ax.plot([],[],'o-',color='orange',lw=4,\
                        markersize=6,markeredgecolor='k',\
                        markerfacecolor='k')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)
        start_text = ax.text(-1.06,-0.3,'start',ha='right')
        end_text = ax.text(0.06,-0.3,'objective',ha='left')

        def init():
            mass1.set_data([],[])
            mass2.set_data([],[])
            line.set_data([],[])
            time_text.set_text('')
            return line, mass1, mass2, time_text

        def animate(i):
            mass1.set_data([x1[i]],[y1[i]-0.1])
            mass2.set_data([x2b[i]],[y2b[i]])
            line.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            time_text.set_text(time_template % model.time[i])
            return line, mass1, mass2, time_text

        progress_message(message='generating the results...')

        ani_a = animation.FuncAnimation(fig, animate, \
                np.arange(1,len(model.time)), \
                interval=83,blit=False,init_func=init)
        
        plt.plot()
        progress_message(message='performing dynamic calculations.....')

        tempFile = NamedTemporaryFile(suffix='.gif', delete=False, mode='wb')
        ani_a.save(tempFile.name, writer='imagemagick')
        tempFile.close()
        path = Path(tempFile.name)
        return ImageResult(File.from_path(path))
    


    @ImageView("Double Pendulum", duration_guess=10)
    def doublePendulum(self, params, **kwargs):
        #Defining a model
        m = GEKKO(remote=True)
        #Define initial and final conditions and limits
        pi = math.pi;
        x0 = 0; xdot0 = 0
        q10 = pi; q1dot0 = 0 #0=vertical, pi=inverted
        q20 = pi; q2dot0 = 0 #0=vertical, pi=inverted
        xf = params.doubleStep.endx; xdotf = 0
        q1f = 0; q1dotf = 0
        q2f = 0; q2dotf = 0
        xmin = -2; xmax = 2
        umin = -10; umax = 10

        #Defining the time parameter (0, 1)
        N = 100
        t = np.linspace(0,1,N)
        m.time = t

        #Final time
        TF = m.FV(12,lb=2,ub=25); TF.STATUS = 1
        end_loc = len(m.time)-1
        final = np.zeros(len(m.time))
        for i in range(N):
            if i >=(N-2):
                final[i] = 1000

        final = m.Param(value=final)

        #Parameters
        progress_message(message='building state space model...')

        mc = m.Param(value=1) #cart mass
        m1 = m.Param(value=params.doubleStep.massOne*0.002) #link 1 mass
        m2 = m.Param(value=params.doubleStep.massTwo*0.002) #link 2 mass
        L1 = m.Param(value=.5) #link 1 length
        LC1 = m.Param(value=.25)  #link 1 CM pos
        L2 = m.Param(value=.5) #link 1 length
        LC2 = m.Param(value=.25) #link 1 CM pos
        I1 = m.Param(value=.01) #link 1 MOI
        I2 = m.Param(value=.01) #link 2 MOI
        g = m.Const(value=moonGrav(params)) #gravity
        Bc = m.Const(value=.5) #cart friction
        B1 = m.Const(value=.001) #link 1 friction
        B2 = m.Const(value=.001) #link 2 friction

        #MV
        u = m.MV(lb=umin,ub=umax); u.STATUS = 1

        #State Variables
        x, xdot, q1, q1dot, q2, q2dot = m.Array(m.Var, 6)

        x.value = x0; xdot.value = xdot0
        q1.value = q10; q1dot.value = q1dot0
        q2.value = q20; q2dot.value = q2dot0
        x.LOWER = xmin; x.UPPER = xmax

        #Intermediates
        h1 = m.Intermediate(mc + m1 + m2)
        h2 = m.Intermediate(m1*LC1 + m2*L1)
        h3 = m.Intermediate(m2*LC2)
        h4 = m.Intermediate(m1*LC1**2 + m2*L1**2 + I1)
        h5 = m.Intermediate(m2*LC2*L1)
        h6 = m.Intermediate(m2*LC2**2 + I2)
        h7 = m.Intermediate(m1*LC1*g + m2*L1*g)
        h8 = m.Intermediate(m2*LC2*g)

        M = np.array([[h1, h2*m.cos(q1), h3*m.cos(q2)],
                    [h2*m.cos(q1), h4, h5*m.cos(q1-q2)],
                    [h3*m.cos(q2), h5*m.cos(q1-q2), h6]])
        C = np.array([[Bc, -h2*q1dot*m.sin(q1), -h3*q2dot*m.sin(q2)],
                    [0, B1+B2, h5*q2dot*m.sin(q1-q2)-B2],
                    [0, -h5*q1dot*m.sin(q1-q2)-B2, B2]])

        G = np.array([0, -h7*m.sin(q1), -h8*m.sin(q2)])
        U = np.array([u, 0, 0])
        DQ = np.array([xdot, q1dot, q2dot])
        CDQ = C@DQ
        b = np.array([xdot.dt()/TF, q1dot.dt()/TF, q2dot.dt()/TF])
        Mb = M@b

        progress_message(message='Defining equations...')

        #Defining the State Space Model
        m.Equations([(Mb[i] == U[i] - CDQ[i] - G[i]) for i in range(3)])
        m.Equation(x.dt()/TF == xdot)
        m.Equation(q1.dt()/TF == q1dot)
        m.Equation(q2.dt()/TF == q2dot)

        m.Obj(final*(x-xf)**2)
        m.Obj(final*(xdot-xdotf)**2)
        m.Obj(final*(q1-q1f)**2)
        m.Obj(final*(q1dot-q1dotf)**2)
        m.Obj(final*(q2-q2f)**2)
        m.Obj(final*(q2dot-q2dotf)**2)

        #Try to minimize final time
        m.Obj(TF)

        m.options.IMODE = 6 #MPC
        m.options.SOLVER = 3 #IPOPT
        m.solve()

        m.time = np.multiply(TF, m.time)

        plt.close('all')
        progress_message(message='generating the results...')

        x1 = x.value
        y1 = np.zeros(len(m.time))

        x2 = L1.value*np.sin(q1.value)+x1
        x2b = (1.05*L1.value[0])*np.sin(q1.value)+x1
        y2 = L1.value[0]*np.cos(q1.value)+y1
        y2b = (1.05*L1.value[0])*np.cos(q1.value)+y1

        x3 = L2.value[0]*np.sin(q2.value)+x2
        x3b = (1.05*L2.value[0])*np.sin(q2.value)+x2
        y3 = L2.value[0]*np.cos(q2.value)+y2
        y3b = (1.05*L2.value[0])*np.cos(q2.value)+y2

        fig = plt.figure(figsize=(8,6.4))
        ax = fig.add_subplot(111,autoscale_on=False,\
                            xlim=(-2.5,2.5),ylim=(-2.5,2.5))
        ax.set_xlabel('position')
        ax.get_yaxis().set_visible(False)

        crane_rail, = ax.plot([-2.5,2.5],[-0.2,-0.2],'k-',lw=4)
        start, = ax.plot([-1.5,-1.5],[-1.5,1.5],'k:',lw=2)
        objective, = ax.plot([1.5,1.5],[-1.5,1.5],'k:',lw=2)
        mass1, = ax.plot([],[],linestyle='None',marker='s',\
                        markersize=40,markeredgecolor='k',\
                        color='orange',markeredgewidth=2)
        mass2, = ax.plot([],[],linestyle='None',marker='o',\
                        markersize=20,markeredgecolor='k',\
                        color='orange',markeredgewidth=2)
        mass3, = ax.plot([],[],linestyle='None',marker='o',\
                        markersize=20,markeredgecolor='k',\
                        color='orange',markeredgewidth=2)
        line12, = ax.plot([],[],'o-',color='black',lw=4,\
                        markersize=6,markeredgecolor='k',\
                        markerfacecolor='k')
        line23, = ax.plot([],[],'o-',color='black',lw=4,\
                        markersize=6,markeredgecolor='k',\
                        markerfacecolor='k')

        time_template = 'time = %.1fs'
        time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)

        def init():
            mass1.set_data([],[])
            mass2.set_data([],[])
            mass3.set_data([],[])
            line12.set_data([],[])
            line23.set_data([],[])
            time_text.set_text('')
            return line12, line23, mass1, mass2, mass3, time_text

        def animate(i):
            mass1.set_data([x1[i]], [y1[i]-0.1])
            mass2.set_data([x2[i]], [y2[i]])
            mass3.set_data([x3[i]], [y3[i]])
            line12.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            line23.set_data([x2[i],x3[i]],[y2[i],y3[i]])
            time_text.set_text(time_template % m.time[i])
            return line12, line23, mass1, mass2, mass3, time_text
        progress_message(message='generating visualisation...')

        ani_a = animation.FuncAnimation(fig, animate, \
                np.arange(len(m.time)), \
                interval=83,init_func=init) 
        
        tempFile = NamedTemporaryFile(suffix='.gif', delete=False, mode='wb')
        ani_a.save(tempFile.name, writer='Pillow')
        tempFile.close()
        path = Path(tempFile.name)
        return ImageResult(File.from_path(path))
    
    @WebView("What's next?", duration_guess=1)
    def whats_next(self, params, **kwargs):
        """Initiates the process of rendering the "What's next" tab."""
        html_path = Path(__file__).parent / "next_step.html"
        with html_path.open(encoding="utf-8") as _file:
            html_string = _file.read()
        return WebResult(html=html_string)
    