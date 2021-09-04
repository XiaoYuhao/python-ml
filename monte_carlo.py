import random

'''
赌博问题

给定一个赌徒初始本钱，其每轮赌博输赢的概率确定，赢一轮得1单位金钱，输一轮输1单位金钱。
输光身上所有钱之后，出局。

结论就是，随着轮数的增加，基本上都破产被收割了。原因是庄家的资金是无穷的。
'''


def game():
    init_money = 100                             # 本钱
    people_num = 1000                              # 参与人数
    round_num = 100000                            # 参与轮数
    result = [] 
    for i in range(0, people_num):      
        money = init_money
        out_round = 0
        for j in range(1, round_num+1):
            p = random.randint(0, 1)        # 输赢的概率为0.5
            if p == 0:
                money += 1                  
            else:
                money -= 1
            out_round = j
            if money == 0:                  # 本钱输光就出局
                break
        result.append((i, out_round, money))
    
    win_people = 0
    lose_people = 0
    out_people = 0
    for res in result:
        print(res)
        if res[1] < round_num:
            out_people += 1
        elif res[2] >= init_money:
            win_people += 1
        else:
            lose_people += 1
    
    print(f"win people  : {win_people}")
    print(f"lose people : {lose_people}")
    print(f"out people  : {out_people}")

'''
生育问题

某些家庭在重男轻女的思想下（当然这是非常不好的思想），会采取这样的生育策略：

- 若生下的是男孩，则停止；
- 若生下的是女孩，则继续，直到生下男孩为止。

这也是为什么现在很多家庭由一个弟弟和多个姐姐组成的原因。
并且有些人认为，这样会导致男女比例不均衡问题的出现，女生的比例会比男生要高一些，
然而在实际统计数据中，男生比例要高于女生。

假设生下男孩和女孩的概率均等。

试问会这种策略会影响男女比例吗？

不会。生育问题是典型的稳态马尔可夫过程，下一次生育不受上一次生育的影响。根据马氏过程
的特性，你知道历史无需考虑历史路径，最终的平衡概率只取决于每一步的概率。
'''

def birth():
    parent_num = 1000000
    male_num = 0
    female_num = 0

    for i in range(1, parent_num+1):
        while True:
            p = random.randint(0, 1)            # 生男生女概率均等，为0.5
            if p == 0:
                male_num += 1                   # 生下男孩
                break
            else:
                female_num += 1                 # 生下女孩
    
    print(f"male num   : {male_num}")
    print(f"female num : {female_num}")


if __name__ == '__main__':
    game()
    #birth()