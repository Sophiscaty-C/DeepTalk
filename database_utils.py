from utils import  *
import sqlite3

def create_database(group, member, message, Type_dict):
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    create_sql="CREATE TABLE IF NOT EXISTS "
    temp=message.split(' ')
    for i in range(1, len(temp)):
        if i==1:
            create_sql+=temp[i]+" (ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
            Type_dict[len(Type_dict)]=temp[i]
        elif i<len(temp)-1:
            create_sql+=temp[i]+" TEXT, "
        else:
            create_sql += temp[i] + " TEXT);"
    set_dict("./database/" + str(group.id) + "/Type_dict.json", Type_dict)
    cursor.execute(create_sql)
    connect.commit()
    connect.close()
    return temp[1]

def insert_database(group, member, t, message, Type_dict):
    t = str(t)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    col_name=get_col_name(cursor, Type_dict[t])
    temp = message.split(' ')
    insert_sql="INSERT INTO "+Type_dict[t]+" ("
    for i in range(1, len(temp)):
        if i<len(temp)-1:
            insert_sql+=col_name[i]+", "
        else:
            insert_sql += col_name[i] + ") VALUES ("
    for i in range(1, len(temp)):
        if i<len(temp)-1:
            insert_sql+="'"+temp[i]+"', "
        else:
            insert_sql +="'"+temp[i]+"')"
    cursor.execute(insert_sql)
    connect.commit()
    connect.close()

def select_database(group, member, t, Type_dict, message="", page=1):
    p, l = t
    p = str(p)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    col_name=get_col_name(cursor, Type_dict[p])
    st=""
    for i in col_name:
        st+=i+" "
    st+="\n"
    select_sql=""
    if len(l)==0:
        select_sql+="SELECT * FROM "+Type_dict[p]
    elif len(l)!=0 and message=="":
        select_sql += "SELECT * FROM " + Type_dict[p] + " WHERE "
        for i in range(len(col_name)):
            for j in range(len(l)):
                if i<len(col_name)-1 and j<len(l)-1:
                    select_sql += col_name[i] + " LIKE '%" + l[j] + "%' OR "
                else:
                    select_sql += col_name[i] + " LIKE '%" + l[j] + "%'"
    elif message!="":
        select_sql+="SELECT * FROM "+Type_dict[p]+" WHERE "
        for i in range(len(col_name)):
            if i < len(col_name) - 1:
                select_sql+=col_name[i]+" LIKE '%"+message+"%' OR "
            else:
                select_sql += col_name[i] + " LIKE '%" + message + "%'"
    c=cursor.execute(select_sql)
    count=0
    for row in c:
        count+=1
        if (page - 1)*10 < count <= page*10:
            for col in range(len(col_name)):
                st+=str(row[col])+" "
            st+='\n'
        else:
            continue
    connect.close()
    if count%10==0:
        pagecount=int(count/10)
    else:
        pagecount=int(count/10)+1
    return st, pagecount

def update_database(group, member, t, message, Type_dict):
    p, l = t
    p = str(p)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    temp = message.split(' ')
    update_sql=""
    if temp[1].isdigit():
        id = temp[1]
        for i in range(2, len(temp), 2):
            update_sql += "UPDATE " + Type_dict[p] + " SET "+temp[i]+"='"+temp[i+1]+"' WHERE ID=" + id
    else:
        id1 = temp[1].split('-')[0]
        id2 = temp[1].split('-')[1]
        for i in range(2, len(temp), 2):
            update_sql += "UPDATE " + Type_dict[p] + " SET "+temp[i]+"='"+temp[i+1]+"' WHERE ID>=" + id1 + " AND ID<=" + id2
    cursor.execute(update_sql)
    connect.commit()
    connect.close()
def delete_database(group, member, t, Type_dict, message=""):
    p, l = t
    p = str(p)
    s = "./database/" + str(group.id) + "/" + str(member.id) + ".db"
    connect = sqlite3.connect(s)
    cursor = connect.cursor()
    col_name = get_col_name(cursor, Type_dict[p])
    if message=="":
        select_sql=""
        if len(l) == 0:
            select_sql += "SELECT ID FROM " + Type_dict[p]
        elif len(l) != 0 and message == "":
            select_sql += "SELECT ID FROM " + Type_dict[p] + " WHERE "
            for i in range(len(col_name)):
                for j in range(len(l)):
                    if i < len(col_name) - 1 and j < len(l) - 1:
                        select_sql += col_name[i] + " LIKE '%" + l[j] + "%' OR "
                    else:
                        select_sql += col_name[i] + " LIKE '%" + l[j] + "%'"
        delete_sql="DELETE FROM "+Type_dict[p]+" WHERE ID IN (SELECT ID FROM(" +select_sql+") AS TMP)"
    else:
        temp=message.split(' ')[1]
        if temp.isdigit():
            id=temp
            delete_sql="DELETE FROM "+Type_dict[p]+" WHERE ID="+id
        else:
            id1=temp.split('-')[0]
            id2=temp.split('-')[1]
            delete_sql = "DELETE FROM " + Type_dict[p] + " WHERE ID>=" + id1+" AND ID<="+id2
    cursor.execute(delete_sql)
    connect.commit()
    connect.close()