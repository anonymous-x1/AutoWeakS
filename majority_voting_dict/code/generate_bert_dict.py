import numpy as np
import heapq
global num_course
global num_train_job

num_course = 1951
num_train_job = 566

bert_pretrain_job_npy_path = '/Users/jeremyhao/Desktop/revise_paper/bert_pretrain/bert_pretrain_job.npy'
bert_pretrain_course_npy_path = '/Users/jeremyhao/Desktop/revise_paper/bert_pretrain/bert_pretrain_course.npy'

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
'''
bert pretrain
'''
def generate_dict(output_file):
    course_list = range(num_course)
    job_vector = np.load(bert_pretrain_job_npy_path)
    course_vector = np.load(bert_pretrain_course_npy_path)
    with open(output_file, 'w') as writer:
        for job_index in range(num_train_job):
            writer.write(str(job_index)+' ')
            job_ebd = job_vector[job_index]
            predictions = []
            for course_index in range(num_course):
                course_ebd = course_vector[course_index]
                predictions.append(cos_sim(course_ebd, job_ebd))
            map_course_score = {course_list[t]: predictions[t] for t in range(len(course_list))}
            ranklist = heapq.nlargest(num_course, map_course_score, key=map_course_score.get)
            for item in ranklist:
                writer.write(str(item)+' ')
            writer.write('\n')

