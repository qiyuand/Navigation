from params import parse_all_args
from train import Task


def main():
    args = parse_all_args()
    task = Task(args)
    task.train()



if __name__ == '__main__':
    main()