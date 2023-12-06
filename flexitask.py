import collections as col
import streamlit as st
import random
import datetime as dt
from ortools.sat.python import cp_model
import math
import time
import plotly_express as px
import pandas as pd

color_map = {'NAVY': '#001f3f', 'BLUE': '#0074D9',
             'AQUA': '#7FDBFF', 'TEAL': '#39CCCC',
             'PURPLE': '#B10DC9',
             'FUCHSIA': '#F012BE',
             'MAROON': '#85144b',
             'RED': '#DB7093',
             'ORANGE': '#FF851B',
             'YELLOW': '#FFDC00',
             'GREEN': '#2ECC40',
             'LIME': '#01FF70'}

color_setup_matrix = {'NAVY': {'NAVY': 0,
                               'BLUE': 5,
                               'AQUA': 4,
                               'TEAL': 4,
                               'PURPLE': 6,
                               'FUCHSIA': 7,
                               'MAROON': 4,
                               'RED': 4,
                               'ORANGE': 6,
                               'YELLOW': 8,
                               'GREEN': 5,
                               'LIME': 5},
                      'BLUE': {'NAVY': 3,
                               'BLUE': 0,
                               'AQUA': 6,
                               'TEAL': 8,
                               'PURPLE': 6,
                               'FUCHSIA': 5,
                               'MAROON': 4,
                               'RED': 4,
                               'ORANGE': 8,
                               'YELLOW': 8,
                               'GREEN': 6,
                               'LIME': 4},
                      'AQUA': {'NAVY': 5,
                               'BLUE': 8,
                               'AQUA': 0,
                               'TEAL': 3,
                               'PURPLE': 4,
                               'FUCHSIA': 6,
                               'MAROON': 7,
                               'RED': 5,
                               'ORANGE': 4,
                               'YELLOW': 5,
                               'GREEN': 7,
                               'LIME': 6},
                      'TEAL': {'NAVY': 4,
                               'BLUE': 8,
                               'AQUA': 3,
                               'TEAL': 0,
                               'PURPLE': 3,
                               'FUCHSIA': 7,
                               'MAROON': 4,
                               'RED': 6,
                               'ORANGE': 3,
                               'YELLOW': 8,
                               'GREEN': 8,
                               'LIME': 5},
                      'PURPLE': {'NAVY': 5,
                                 'BLUE': 8,
                                 'AQUA': 6,
                                 'TEAL': 3,
                                 'PURPLE': 0,
                                 'FUCHSIA': 8,
                                 'MAROON': 5,
                                 'RED': 7,
                                 'ORANGE': 3,
                                 'YELLOW': 4,
                                 'GREEN': 3,
                                 'LIME': 7},
                      'FUCHSIA': {'NAVY': 5,
                                  'BLUE': 4,
                                  'AQUA': 4,
                                  'TEAL': 5,
                                  'PURPLE': 5,
                                  'FUCHSIA': 0,
                                  'MAROON': 4,
                                  'RED': 5,
                                  'ORANGE': 7,
                                  'YELLOW': 7,
                                  'GREEN': 6,
                                  'LIME': 7},
                      'MAROON': {'NAVY': 7,
                                 'BLUE': 5,
                                 'AQUA': 4,
                                 'TEAL': 7,
                                 'PURPLE': 8,
                                 'FUCHSIA': 3,
                                 'MAROON': 0,
                                 'RED': 4,
                                 'ORANGE': 8,
                                 'YELLOW': 4,
                                 'GREEN': 7,
                                 'LIME': 8},
                      'RED': {'NAVY': 7,
                              'BLUE': 2,
                              'AQUA': 4,
                              'TEAL': 4,
                              'PURPLE': 7,
                              'FUCHSIA': 6,
                              'MAROON': 8,
                              'RED': 0,
                              'ORANGE': 4,
                              'YELLOW': 8,
                              'GREEN': 4,
                              'LIME': 4},
                      'ORANGE': {'NAVY': 4,
                                 'BLUE': 4,
                                 'AQUA': 7,
                                 'TEAL': 3,
                                 'PURPLE': 7,
                                 'FUCHSIA': 6,
                                 'MAROON': 6,
                                 'RED': 4,
                                 'ORANGE': 0,
                                 'YELLOW': 5,
                                 'GREEN': 6,
                                 'LIME': 6},
                      'YELLOW': {'NAVY': 6,
                                 'BLUE': 8,
                                 'AQUA': 7,
                                 'TEAL': 7,
                                 'PURPLE': 7,
                                 'FUCHSIA': 4,
                                 'MAROON': 6,
                                 'RED': 5,
                                 'ORANGE': 4,
                                 'YELLOW': 0,
                                 'GREEN': 3,
                                 'LIME': 6},
                      'GREEN': {'NAVY': 4,
                                'BLUE': 3,
                                'AQUA': 7,
                                'TEAL': 7,
                                'PURPLE': 7,
                                'FUCHSIA': 4,
                                'MAROON': 3,
                                'RED': 3,
                                'ORANGE': 5,
                                'YELLOW': 3,
                                'GREEN': 0,
                                'LIME': 6},
                      'LIME': {'NAVY': 5,
                               'BLUE': 3,
                               'AQUA': 5,
                               'TEAL': 3,
                               'PURPLE': 4,
                               'FUCHSIA': 3,
                               'MAROON': 4,
                               'RED': 6,
                               'ORANGE': 3,
                               'YELLOW': 4,
                               'GREEN': 7,
                               'LIME': 0}}

colors = ['RED', 'YELLOW', 'BLUE', 'GREEN']
durations = [3, 5, 6, 8, 10, 2]

intermediate_solutions = [{'makespan': 1000, 'solno': -1}]


class SolutionHandler(cp_model.CpSolverSolutionCallback):
    def __init__(self, makespan, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__makespan = makespan
        self.__solution_count = 0
        self.__solution_limit = limit
        self.__datapoints = [1]  # {'makespan': 1000, 'solno': -1}]
        # self.fig2 = px.scatter(intermediate_solutions,
        #                        x="solno", y="makespan")
        # self.chart = st.plotly_chart(self.fig2, use_container_width=True)
        self.chart = st.line_chart([])  # self.__datapoints)

    def on_solution_callback(self):
        self.__solution_count += 1
        print('Obj = ' + str(self.ObjectiveValue()),
              'Solution No.'+str(self.__solution_count), self.WallTime())
        self.__datapoints.append(
            {'makespan': self.ObjectiveValue(), 'solno': self.__solution_count})
        # {'makespan': self.Value(self.__makespan), 'solno': self.__solution_count})
        # self.Value(self.__makespan)])
        self.chart.add_rows([self.ObjectiveValue()])
        # print()
        if self.__solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

        # self.chart.add_rows(intermediate_solutions)
        # st.table(self.__datapoints)
        # , color="Color",  color_discrete_map=color_map, title="Best Solution")  # text="Task",

    def solution_count(self):
        return self.__solution_count

    def datapoints(self):
        return self.__datapoints


st.set_page_config(layout="wide")
model = cp_model.CpModel()

max_tasks = 100
max_res = 10
max_sol = 1000

task = col.namedtuple('task', 'name color duration start end')
restask = col.namedtuple('restask', 'name res start end presense interval')
restasksol = col.namedtuple('restasksol', 'name res start end presense')

tasks = col.defaultdict(list)
restasks = col.defaultdict(list)

form = st.sidebar.form(key="input-form")
n_task = form.slider("Tasks", min_value=2, max_value=max_tasks, value=5)
n_res = form.slider("Resources", min_value=1, max_value=max_res, value=2)
n_sol = form.slider("Max Solution", min_value=1,
                    max_value=max_sol, value=50)
time_limit = form.slider("Time limit (sec)",
                         min_value=10, max_value=600, value=30)
horizon = form.slider("Horizon", min_value=0, max_value=1000, value=200)
change_over_limit = form.slider(
    "Setup limit", min_value=0, max_value=100, value=30)
# define variables
for i in range(0, n_task):
    # task related variable
    tasks[i] = task('T-'+str(i), colors[i % len(colors)], durations[i % len(durations)],
                    model.NewIntVar(0, horizon, 'T-'+str(i)+'-start'),
                    model.NewIntVar(0, horizon, 'T-'+str(i)+'-end'))

    selected1 = 0
    count = 0
    # task-resource related variables
    for r in range(0, n_res):
        selected1 = (1-selected1)*random.randint(0, 1)
        # start
        res_task_start = model.NewIntVar(
            0, horizon, 'R-'+str(r)+'-T-'+str(i)+'-start')
        # end
        res_task_end = model.NewIntVar(
            0, horizon, 'R-'+str(r)+'-T-'+str(i)+'-end')

        # presense
        res_task_presense = model.NewBoolVar(
            'R-'+str(r)+'-T-'+str(i)+'-presense')

        # interval
        res_task_interval = model.NewOptionalIntervalVar(
            res_task_start, tasks[i].duration, res_task_end, res_task_presense, 'R-'+str(r)+'-T-'+str(i)+'-interval')

        # restask
        restasks[r, i] = restask('R-'+str(r)+'-'+tasks[i].name, r, res_task_start,
                                 res_task_end, res_task_presense, res_task_interval)
        #print('res_no', str(r), 'i=', str(i), type(restasks[r, i]))

    # task can only be executed at one resource
    model.Add(cp_model.LinearExpr.Sum(
        [restasks[r, i].presense for r in range(0, n_res)]) == 1)

    # # task start should be start of res task
    # model.AddMaxEquality(
    #     tasks[i].start, [restasks[r, i].start for r in range(0, n_res)])
    # # # task end should be end of res task
    # model.AddMaxEquality(
    #     tasks[i].end, [restasks[r, i].end for r in range(0, n_res)])

res_positions = {}
# resource constraints
res_changeovers = col.defaultdict(list)
for r in range(0, n_res):
    # no overlap
    model.AddNoOverlap([restasks[r, t].interval for t in range(0, n_task)])

    arcs = col.defaultdict(list)
    # # exclusion implication
    for t in range(n_task):
        model.Add(restasks[r, t].start == 0).OnlyEnforceIf(
            restasks[r, t].presense.Not())
        model.Add(restasks[r, t].end == 0).OnlyEnforceIf(
            restasks[r, t].presense.Not())
        model.Add(restasks[r, t].start == tasks[t].start).OnlyEnforceIf(
            restasks[r, t].presense)
        model.Add(restasks[r, t].end == tasks[t].end).OnlyEnforceIf(
            restasks[r, t].presense)

        arcs[-1, t] = model.NewBoolVar('R-'+str(r)+'T-'+str(t)+'First')
        model.Add(restasks[r, t].presense == 1).OnlyEnforceIf(arcs[-1, t])
        # model.AddImplication(arcs[-1, t], restasks[r, t].presense)

        arcs[t, -1] = model.NewBoolVar('R-'+str(r)+'T-'+str(t)+'Last')
        model.Add(restasks[r, t].presense == 1).OnlyEnforceIf(arcs[t, -1])
        # model.AddImplication(arcs[-1, t], restasks[r, t].presense)

        # disjunctive constraints
        for t2 in range(n_task):
            if(t != t2):
                pred = model.NewBoolVar(
                    'R-'+str(r)+'-T-'+str(t)+'->T-'+str(t2))
                arcs[t, t2] = pred
                offset = color_setup_matrix[tasks[t].color][tasks[t2].color]
                if(offset > 0):
                    res_changeovers[r, t, t2] = pred
                model.Add(restasks[r, t2].start >=
                          restasks[r, t].end + offset).OnlyEnforceIf(pred)
                # model.Add(restasks[r, t].presense == 1).OnlyEnforceIf(pred)
                # model.Add(restasks[r, t2].presense == 1).OnlyEnforceIf(pred)

                model.AddImplication(pred, restasks[r, t].presense)
                model.AddImplication(pred, restasks[r, t2].presense)
                # model.AddBoolAnd(
                #     [restasks[r, t].presense, restasks[r, t2].presense]).OnlyEnforceIf(pred)

    res_positions[r] = arcs
    # for each pair atmost one disjunction is allowed
    for t in range(n_task):
        support = []
        support.append(arcs[t, -1])
        recieve = []
        recieve.append(arcs[-1, t])
        next = -1
        for t2 in range(n_task):
            if(t != t2):
                if(t > t2 and tasks[t].color == tasks[t2].color):
                    model.Add(arcs[t, t2] == 0)
                    model.Add(restasks[r, t].start >= restasks[r, t2].start)

                support.append(arcs[t, t2])
                recieve.append(arcs[t2, t])
                # either t supports t2 or t2 supports t
                model.Add(arcs[t, t2] + arcs[t2, t] <= 1)
        # next same
        # for each resource task: exactly one supporter and one receiver
        model.Add(cp_model.LinearExpr.Sum(support) ==
                  1).OnlyEnforceIf(restasks[r, t].presense)
        model.Add(cp_model.LinearExpr.Sum(recieve) ==
                  1).OnlyEnforceIf(restasks[r, t].presense)

        # model.Add(cp_model.LinearExpr.Sum(support) - cp_model.LinearExpr.Sum(recieve)
        #           == 0)

    # only one at position 1
    model.Add(cp_model.LinearExpr.Sum(
        [arcs[t, -1] for t in range(n_task)]) <= 1)

    model.Add(cp_model.LinearExpr.Sum(
        [arcs[-1, t] for t in range(n_task)]) <= 1)


# objective
min_duration = sum([tasks[i].duration for i in range(n_task)])
min_makespan = math.floor(min_duration/n_res)
print("min makespan = ", min_makespan)
makespan = model.NewIntVar(min_makespan, horizon, 'makespan')
model.AddMaxEquality(makespan, [tasks[i].end for i in range(n_task)])

totalChangeover = model.NewIntVar(0, change_over_limit, 'Changeover')
model.Add(cp_model.LinearExpr.Sum(
    [res_changeovers[r, t, t2]*color_setup_matrix[tasks[t].color][tasks[t2].color]for r, t, t2 in res_changeovers]) == totalChangeover)

# totalSelected = model.NewIntVar(0, n_task, 'tasks')
# model.Add(cp_model.LinearExpr.Sum([restasks[r, i].presense for r in range(
#     0, n_res) for i in range(n_task)]) == totalSelected)

# model.Maximize(totalSelected)
# model.Minimize(totalChangeover+makespan)

model.Minimize(makespan)

# totalStart = model.NewIntVar(0, horizon, 'StartPoints')
# model.Add(sum([tasks[t].start for t in range(n_task)]) == totalStart)
# model.Minimize(totalStart)  # + 10*totalChangeover)


def displaySolution(solver, status, start_time, datapoints):

    # st.header(str(solver.StatusName(status)) + '---' + str(solver.Value(makespan)) +
    #           '---'+str(min_makespan) + "--" + str(time.time() - start_time) + '-'+str(solver.Value(totalChangeover)))

    start_schedule = dt.datetime.now().replace(hour=0, minute=0, second=0)
    sol = []
    for r in range(n_res):
        for t in range(n_task):
            if solver.Value(restasks[r, t].presense):
                sol.append({
                    'task': restasks[r, t].name,
                    'Resource': 'res-'+str(r),
                    'Start': start_schedule + dt.timedelta(hours=solver.Value(tasks[t].start)),
                    'End': start_schedule + dt.timedelta(hours=solver.Value(restasks[r, t].end)),
                    'Color': tasks[t].color})

    st.header('Status = ' + str(solver.StatusName(status)))
    st.text('Makespan = ' + str(solver.Value(makespan)))
    st.text('Solver Time = ' + str(time.time() - start_time))
    st.text('Total Setup = '+str(solver.Value(totalChangeover)))
    # st.table(sol)
    # resource timeline chart
    # st.header("Resource Schedule:")
    df = pd.DataFrame(sol)
    fig = px.timeline(df,  x_start="Start", x_end="End",
                      y="Resource", color="Color",  text="task", color_discrete_map=color_map, title="Best Solution")  # text="task",
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

    # # , color="Color",  color_discrete_map=color_map, title="Best Solution")  # text="Task",
    # fig2 = px.scatter(datapoints,  x="solno", y="makespan")

    # st.plotly_chart(fig2, use_container_width=True)

    # for r in range(n_res):
    #     st.write("Resource-"+str(r))
    #     for a in res_positions[r]:
    #         from_t = a[0]
    #         to_t = a[1]
    #         if(solver.Value(res_positions[r][from_t, to_t])):
    #             st.write(res_positions[r][from_t, to_t].Name(
    #             ) + ' Value = ' + str(solver.Value(res_positions[r][from_t, to_t])))

    # for r, t, t2 in res_changeovers:
    #     var = res_changeovers[r, t, t2]
    #     offset = color_setup_matrix[tasks[t].color][tasks[t2].color]
    #     if solver.Value(var) == 1:
    #         st.write(
    #             var.Name()+' ['+str(solver.Value(var))+']- Offset-' + str(offset))


def solve():
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.num_search_workers = 16
    solver.interleave_search = True
    solver.interleave_batch_size = 5
    solver.use_main_interval_for_tasks = True
    solver.log_search_progress = True
    solver.log_to_stdout = True
    start_time = time.time()

    sol_handler = SolutionHandler(makespan, n_sol)
    # status = solver.Solve(model)
    status = solver.SolveWithSolutionCallback(model, sol_handler)
    st.write("Status = " + solver.StatusName(status=status) + '-'+str(status))
    if(status > 0):
        displaySolution(solver, status, start_time, sol_handler.datapoints())


submit = form.form_submit_button(label="Solve")

if(submit):
    solve()
else:
    st.title("Task assignment problem play ground")
    st.header("Problem Statement")
    st.text("Tasks: number of tasks to allocate to resources.")
    st.text("Each task will be assigned a random duration and a color (4 possible colors are assigned).")
    st.text("Optimizer assigns tasks to resources to minimize overall makespan.")
    st.text("Resources: number available resources where tasks can be assigned.")
    st.text("When two tasks are assigned to a resource next to each other a setup time is allocated based on the colors of tasks.")
    st.header("Solver tuning")
    st.text("Time limit: limit runtime of solver")
    st.text("Horizon: bound for makespan.")
    st.text("Setup limit: total setup time bound")
