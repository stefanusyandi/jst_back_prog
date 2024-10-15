import numpy as np
import copy
import tabulate

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

miu = 0.1
bobot_1 = np.matrix([
    [15.41, 2.61],
    [-15.28, -0.59],
    [-10.79, 1.90]
])

bobot_2 = np.matrix([
    [0.78],
    [0.52],
    [-0.09]
])

def feed_to_hidden_layer(inputtan) -> np.matrix:
    inputtan = copy.deepcopy(inputtan)
    inputtan = np.hstack((inputtan, np.matrix([[1]])))
    
    hasil = (inputtan * bobot_1)

    return hasil

def feed_to_output(inputtan) -> np.matrix:
    inputtan = copy.deepcopy(inputtan)
    inputtan = np.hstack((inputtan, np.matrix([[1]])))
    # print(inputtan)

    hasil = (inputtan * bobot_2)

    return hasil

def update_weight_2(target : float, miu : float, hasil : float, inputtan : np.matrix):
    inputtan = copy.deepcopy(inputtan)
    
    inputtan = np.hstack((inputtan, np.matrix([[1]])))
    # print(inputtan)
    error = hasil - target
    result =  bobot_2 - miu * error * (1 - hasil) * inputtan.T
    # print()
    # # print(bobot_2)
    # print()
    return result


def update_weight_1(target : float, miu : float, hasil : float, inputtan : np.matrix):
    inputtan = copy.deepcopy(inputtan)
    
    inputtan = np.hstack((inputtan, np.matrix([[1]])))
    # print(inputtan)
    error = hasil - target
    result = bobot_1 - miu * error * (1 - hasil) * inputtan.T
    # print(bobot_2)
    return result

def print_bobot_1(matrix):
    print("\t\t-- BOBOT 1 --")
    matrix_as_array = matrix.A  # Converts to a NumPy array
    list_from_array : list= matrix_as_array.tolist()  # Converts to a list

    list_from_array.insert(0, ["  ", "N1", "N2"])
    list_from_array[1].insert(0, "X1")
    list_from_array[2].insert(0, "X2")
    list_from_array[3].insert(0, "1")
    print(tabulate.tabulate(list_from_array ))

def print_bobot_2(matrix):
    print("\t\t-- BOBOT 2 --")
    matrix_as_array = matrix.A  # Converts to a NumPy array
    list_from_array : list= matrix_as_array.tolist()  # Converts to a list

    list_from_array.insert(0, ["  ", "Output"])
    list_from_array[1].insert(0, "N1")
    list_from_array[2].insert(0, "N2")
    list_from_array[3].insert(0, "N0")
    print(tabulate.tabulate(list_from_array ))

def main() :
    global bobot_1
    global bobot_2
    data_inputtan = [
        [0.1, 0.35, 0.3],
        [0.2, 0.2, 0.3],
        [0.6, 0.55, 0.55],
        [0.35, 0.35, 0.50],
        [0.2, 0.25, 0.45],
        [0.3, 0.25, 0.35],
        [0.6, 0.45, 0.35],
        [0.20, 0.30, 0.35],
        [0.55, 0.45, 0.50],
        [0.45, 0.90, 0.80],
        [0.45, 0.70, 0.80],
        [0.50, 0.82, 0.70]
    ]
    print(f"miu = {miu}")
    print_bobot_1(bobot_1)
    print_bobot_2(bobot_2)

    for i in range(len(data_inputtan)):
        print("============ INPUT - " + str(i + 1) + " ===================")

        x1 = data_inputtan[i][0]
        x2 = data_inputtan[i][1]
        target = data_inputtan[i][2]
        print(f"X1 = {x1}")
        print(f"X2 = {x2}")
        print(f"target = {target}")
        inputtan = [x1, x2]
        inputtan = np.matrix([
            inputtan
        ])

        output_ke_hidden_layer = feed_to_hidden_layer(inputtan)
        print(f"Output dari input awal ke hidden layer: ")
        print(f"N1 = {output_ke_hidden_layer[0,0]} \nN2 = {output_ke_hidden_layer[0,1]}")
        # print(output_ke_hidden_layer[0,0])
        print(f"dimasukkan ke fungsi aktivasi sigmoid menjadi: N1 = {sigmoid(output_ke_hidden_layer)[0, 1]}, N2 = {sigmoid(output_ke_hidden_layer)[0, 1]}")
        print()
        output_final = feed_to_output(
            sigmoid(output_ke_hidden_layer)
        )

        print(f"Masukkan ke output layer dan akan menghasilkan hasil = {output_final[0, 0]}")
        print(f"Masukkan ke fungsi aktivasi, dan hasil akhirnya adalah = {sigmoid(output_final[0,0])}")

        weight2 = update_weight_2(target, miu, output_final[0, 0],inputtan)
        bobot_2 = weight2
        
        print()

        weight1 = update_weight_1(target, miu, output_final[0, 0],inputtan)
        bobot_1 = weight1

        print("Ubah bobot sesuai dengan algoritma Back Propagation maka akan mendapatan: ")
        print_bobot_1(bobot_1)
        print_bobot_2(bobot_2)
if __name__ == "__main__":
    main()




