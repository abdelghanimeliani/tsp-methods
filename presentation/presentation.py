from random import randint
from manim import *
from manim_presentation import Slide
import numpy as np


class PFA(Scene):
    alpha, beta = np.random.uniform(1, 2, 2)
    r = np.random.random((2, 4))
    np.random.seed(76)
    def construct(self):
        # parameters
        pop_size = 10
        cost = lambda x, y: x ** 2 *.3 + y ** 2 *.7 + 1

        # setup environment
        grid = NumberPlane()
        self.play(Write(grid))
        self.wait()
        x_min, x_max, _ = grid.x_range
        y_min, y_max, _ = grid.y_range
        
        # population
        population = VGroup(
            *[
                Dot(grid.c2p(x, y), .05, color=RED) for x, y in 
                zip(np.random.uniform(x_min, x_max, pop_size), np.random.uniform(y_min, y_max, pop_size))
            ]
        )
        self.play(Create(population))
        self.wait()
        
        path_finder = sorted(population, key=lambda p: cost(p.get_x(), p.get_y()))[0]
        center = Dot(
            grid.c2p(center_of_mass([p.get_x() for p in population]), center_of_mass([p.get_y() for p in population])),
            .05, 
            color=YELLOW
        )

        self.play(Indicate(path_finder, color=GREEN))
        self.play(path_finder.animate.set_color(GREEN))
        self.wait()
        self.play(Create(center))
        self.wait()
        chosen = population[-1]
        to_finder = Arrow(chosen, chosen.get_center()*.9+path_finder.get_center()*.1, buff=0, color=GREEN)
        to_center = Arrow(chosen, chosen.get_center()*.9+center.get_center()*.1, buff=0, color=YELLOW)
        self.play(GrowArrow(to_finder))
        self.wait()
        self.play(GrowArrow(to_center))
        self.wait()

class PFAAnimation(Scene):
    def construct(self):
        
        alpha, beta = np.random.uniform(1, 2, 2)
        r = np.random.random((4, 3))
        u = np.random.uniform(-1, 1, (3, 3))
        np.random.seed(76)
        
        # parameters
        pop_size = 10
        cost = lambda x, y: x ** 2 *.3 + y ** 2 *.7 + 1
        n_iter = 20
        
        # setup environment
        grid = NumberPlane()

        x_min, x_max, _ = grid.x_range
        y_min, y_max, _ = grid.y_range
        
        # population
        population = VGroup(
            *[
                Dot(grid.c2p(x, y), .05, color=RED) for x, y in 
                zip(np.random.uniform(x_min, x_max, pop_size), np.random.uniform(y_min, y_max, pop_size))
            ]
        )
        self.add(population, grid)
        path_finder = old_finder = path_finder = sorted(population, key=lambda p: cost(p.get_x(), p.get_y()))[0]
        for k in range(n_iter):
            self.play(path_finder.animate.set_color(GREEN))
            A = u[2] * np.exp(-k*2/n_iter)
            next_finder = path_finder.shift(r[3]*2*(path_finder.get_center()-old_finder.get_center())+A)
            if cost(next_finder.get_x(), next_finder.get_y()) < cost(path_finder.get_x(), path_finder.get_y()):
                self.play(Transform(old_finder, path_finder))
                self.play(Transform(path_finder, next_finder))
            

            new_pop = []
            for p in population:
                if p == path_finder:
                    continue
                j = randint(0, pop_size-1)
                epsilon = (1 - k/n_iter) * u[1] * np.linalg.norm(p.get_center()-population[j].get_center())
                new_pop.append(
                    p.shift(
                        alpha*r[1]*(p.get_center()-population[j].get_center())+
                        beta*r[2]*(path_finder.get_center()-p.get_center())+epsilon
                    )
                )
                next_finder = sorted(new_pop, key=lambda p: cost(p.get_x(), p.get_y()))[0]
