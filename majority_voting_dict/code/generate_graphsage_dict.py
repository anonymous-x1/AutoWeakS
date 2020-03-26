'''
graphsage
'''
from setting import setting
import numpy as np
import heapq

global num_course
global num_train_job

num_course = 1951
num_train_job = 566
def read_vector_order(args):
    vector_order_dict = {}
    with open(args.vector_order, 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split(' ')
            vector_order_dict[int(arr[0])] = len(vector_order_dict)
            line = f.readline()
    return vector_order_dict

def cos_sim(c, j):
    '''
    calculate the cosine similarity between two vecotrs
    '''
    vector_a = np.mat(c)
    vector_b = np.mat(j)

    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def generate_dict(vector, args, output_file):
    course_list = range(num_course)
    vector_order_list = read_vector_order(args)

    with open(output_file, 'w') as writer:
        for job_index in range(num_train_job):
            writer.write(str(job_index)+' ')
            job_ebd = vector[  vector_order_list[job_index] ]
            predictions = []
            for course_index in range(num_course):
                course_ebd = vector[ vector_order_list [course_index] ]
                predictions.append(cos_sim(course_ebd, job_ebd))
            map_course_score = {course_list[t]: predictions[t] for t in range(len(course_list))}
            ranklist = heapq.nlargest(num_course, map_course_score, key=map_course_score.get)
            # print(len(ranklist))
            # print(ranklist)
            for item in ranklist:
                writer.write(str(item)+' ')
            writer.write('\n')
if __name__ == '__main__':
    args = setting()
    # job_list = load_job_list(args.job_list)
    # course_list = load_course_list(args.course_list)
    vector = np.load(args.vector_path)
    print(vector.shape)
    generate_dict(vector, args, 'graphsage_dict')



