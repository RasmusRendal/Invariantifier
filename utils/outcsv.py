
def get_csv(name):
    x = []
    y = []
    with open(name, 'r') as f:
        content = f.readlines()
        vals = content[0].split(",")
        for i in vals[1:]:
            y.append([])
        for line in content[1:]:
            vals = line.split(',')
            i = int(vals[0])
            x.append(i)
            for i in range(1, len(vals)):
                y[i-1].append(float(vals[i]))
    return x, y

