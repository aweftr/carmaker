class tree():
    global ptfer
    global px_per_unit

    def __init__(self,start_pos):
        self.head = Node(start_pos)
        self.node_append(self.head)

    def search_Node(initial,node,finalpos):
        for i in range(4):
            if i == initial:
                pass
            else:
                if node.state[i] == 0:
                    x=node.pos[0]
                    y=node.pos[1]

                    if i == 0:
                        while True:
                            y = y-1
                            tmp = Node([x,y])
                            if x==finalpos[0] and y==finalpos[1]:
                                node._l_son = tmp
                            elif tmp.rnode:
                                node._l_son = tmp
                                search_Node(2,tmp)
                                break
                            elif tmp.state[0]==1:
                                break
                    elif i==1:
                        while True:
                            x = x+1
                            tmp = Node([x,y])
                            if x==finalpos[0] and y==finalpos[1]:
                                if node._l_son != None:
                                    node._l_son = tmp
                                else:
                                     node._r_son = tmp
                            elif tmp.rnode:
                                if node._l_son != None:
                                    node._l_son = tmp
                                else:
                                     node._r_son = tmp
                                search_Node(3,tmp)
                                break
                            elif tmp.state[1]==1:
                                break
                    elif i==2:
                        while True:
                            y = y+1
                            tmp = Node([x,y])
                            if x==finalpos[0] and y==finalpos[1]:
                                if node._l_son != None:
                                    node._l_son = tmp
                                else:
                                     node._r_son = tmp
                            elif tmp.rnode:
                                if node._l_son != None:
                                    node._l_son = tmp
                                else:
                                    node._r_son = tmp
                                search_Node(0,tmp)
                                break
                            elif tmp.state[2]==1:
                                break
                    elif i==3:
                        while True:
                            x = x-1
                            tmp = Node([x,y])
                            if x==finalpos[0] and y==finalpos[1]:
                                if node._l_son != None:
                                    node._l_son = tmp
                                else:
                                     node._r_son = tmp
                            elif tmp.rnode:
                                if node._l_son != None:
                                    node._l_son = tmp
                                else:
                                    node._r_son = tmp
                                search_Node(1,tmp)
                                break
                            elif tmp.state[3]==1:
                                break
