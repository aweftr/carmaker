class tree():
    global ptfer
    global px_per_unit

    def __init__(self,start_pos):
        self.head = Node(start_pos)
        self.node_append(self.head)

    def search_Node(initial,node):
        for i in range(4):
            if i == initial:
                pass
            else:
                if node.state[i] == 1:
                    x=node.pos[0]
                    y=node.pos[1]
                    if i == 1:
                        while True:
                            x = x+1
                            if Node([x,y]).rnode:
                                node._l_son =
