import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame(columns=['Generation', 'Fitness']) 

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Could not open or find the image')
        exit(0)
    return image

def generate_circle(num_points):  
    points = []
    for i in range(num_points):   
        angle = 2 * np.pi * i / num_points         
        x = int(image.shape[1] / 2 + image.shape[1] / 2 * np.cos(angle))    
        y = int(image.shape[0] / 2 + image.shape[0] / 2 * np.sin(angle))
        points.append((x, y))
    return points

def fitness_function(individual): 
    binary_image = np.ones_like(image) * 255
    for i in range(len(individual) - 1):
        cv2.line(binary_image, points[individual[i]], points[individual[i + 1]], 0, 1)  
    cv2.line(binary_image, points[individual[-1]], points[individual[0]], 0, 1) 
    difference = np.abs(binary_image - image) 
    return np.mean(difference)  

def generate_image(individual):
    img = np.ones_like(image) * 255  
    for i in range(1, len(individual)):  
        cv2.line(img, points[individual[i - 1]], points[individual[i]], 0, 1) 
    return img  

def generate_initial_population(num_population, num_points):
    population = []  
    for i in range(num_population): 
        individual = list(range(num_points)) 
        random.shuffle(individual) 
        population.append(individual) 
    return population   

def crossover(parent1, parent2): 
    child = [-1] * len(parent1)  
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(0, len(parent1) - 1)
    if start > end:
        start, end = end, start
    for i in range(start, end + 1):
        child[i] = parent1[i]
    j = 0
    for i in range(len(parent2)):
        if child[i] == -1:
            while parent2[j] in child:
                j += 1  
            child[i] = parent2[j]
    return child 

def mutate(individual):
    if random.random() < mutation_rate:
        i = random.randint(0, len(individual) - 1)  
        j = random.randint(0, len(individual) - 1)
        individual[i], individual[j] = individual[j], individual[i]
    return individual 

def save_image(image, path):
    cv2.imwrite(path, image)
    
def display_results():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(best_img, cmap='gray')
    plt.title('Generated Image')
    plt.axis('off')

    plt.suptitle(f'Mutation Rate: {mutation_rate}, Population Size: {num_population}, Generations: {num_generations}')
    plt.show()
    
    plt.plot(results['Generation'], results['Fitness'])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Values')
    plt.text(0.95, 0.95, f'Mutation Rate: {mutation_rate}\nPopulation Size: {num_population}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
    
def genetic_algorithm(num_generations, num_population, num_points):
    population = generate_initial_population(num_population, num_points)
    for i in range(num_generations):
        population = sorted(population, key=lambda x: fitness_function(x))
        new_population = population[:num_population // 2]
        while len(new_population) < num_population:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child) 
            new_population.append(child)
        population = new_population 
        fitness = fitness_function(population[0])
        results.loc[i] = [i, fitness] 
        print('Generation:', i, 'Fitness:', fitness_function(population[0]))
        cv2.imshow('Generation', generate_image(population[0]))
        cv2.waitKey(1)
    return population[0]

mutation_rate = 0.2
num_generations = 500
num_points = 360
num_population = 600

image_path = 'Image.jpg'
image = load_image(image_path)
points = generate_circle(num_points)
best_individual = genetic_algorithm(num_generations, num_population, num_points)
best_img = generate_image(best_individual)

save_image(best_img, image_path.split('.')[0] + '_results.jpg')

display_results()