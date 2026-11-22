import pennylane as qml

# 量子比特 0、1、2、3 依次对应图像块的左上、右上、左下、右下。
def circuit_RegionShape_Ry_4qubits(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')

    for wire in range(4):
        qml.RY(weights[wire], wires=wire)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 3])

    for wire in range(4):
        qml.RY(weights[4 + wire], wires=wire)

    single_z = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    neighbor_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
    joint_zz = [
        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
        for i, j in neighbor_pairs
    ]
    return single_z + joint_zz

def circuit_DirectionalEdge_Ry_4qubits(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')

    for wire in range(4):
        qml.RY(weights[wire], wires=wire)

    qml.CRY(weights[4], wires=[0, 1])
    qml.CRY(weights[4], wires=[2, 3])
    qml.CRY(weights[5], wires=[0, 2])
    qml.CRY(weights[5], wires=[1, 3])

    for wire in range(4):
        qml.RY(weights[6 + wire], wires=wire)

    single_z = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    neighbor_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
    joint_zz = [
        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
        for i, j in neighbor_pairs
    ]
    return single_z + joint_zz

def circuit_Texture_Ry_4qubits(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')

    for wire in range(4):
        qml.RY(weights[wire], wires=wire)

    qml.CRY(weights[4], wires=[0, 1])
    qml.CRY(weights[4], wires=[2, 3])
    qml.CRY(weights[5], wires=[0, 2])
    qml.CRY(weights[5], wires=[1, 3])
    qml.CRY(weights[6], wires=[0, 3])
    qml.CRY(weights[7], wires=[1, 2])

    for wire in range(4):
        qml.RY(weights[8 + wire], wires=wire)

    single_z = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    pairwise_zz = [
        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs
    ]
    return single_z + pairwise_zz

def circuit_RGBCrossChannel_4qubits(inputs, weights):
    # 输入按 R、G、B 通道分组，每组含四个空间位置。
    for wire in range(4):
        qml.RY(inputs[..., wire], wires=wire)

    qml.CRY(weights[0], wires=[0, 1])
    qml.CRY(weights[0], wires=[2, 3])

    for wire in range(4):
        qml.RX(inputs[..., 4 + wire], wires=wire)

    qml.CRY(weights[1], wires=[0, 2])
    qml.CRY(weights[1], wires=[1, 3])

    for wire in range(4):
        qml.RZ(inputs[..., 8 + wire], wires=wire)

    # 读出前旋转使 B 通道的相位变化可由 Z/ZZ 测量捕获。
    for wire in range(4):
        qml.RX(weights[2 + wire], wires=wire)
        qml.RY(weights[6 + wire], wires=wire)

    single_z = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    joint_zz = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs]
    return single_z + joint_zz

def circuit_CornerCurvature_4qubits(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation="Y")
    for q in range(4):
        qml.RY(weights[q], wires=q)

    qml.CRY(weights[4], wires=[0, 1])
    qml.CRY(weights[4], wires=[2, 3])
    qml.CRY(weights[5], wires=[0, 2])
    qml.CRY(weights[5], wires=[1, 3])

    qml.CRZ(weights[6], wires=[0, 3])
    qml.CRZ(weights[7], wires=[1, 2])
    for q in range(4):
        qml.RX(weights[8 + q], wires=q)

    single_z = [qml.expval(qml.PauliZ(q)) for q in range(4)]
    pairs = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
    joint_zz = [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in pairs]
    return single_z + joint_zz
