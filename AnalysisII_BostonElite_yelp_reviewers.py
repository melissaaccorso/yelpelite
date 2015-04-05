# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 17:17:54 2013

@author: tdong
"""

def createMacro(RESTLINK,NEW_PAGE=1,EXTRACT=1,REV_NUM=0):
    macro = 'SET !TIMEOUT_PAGE 10' + '\n'
    macro += 'TAB T=1'+'\n'
    macro += 'SET !TIMEOUT_STEP 0' + '\n'
    if NEW_PAGE == 1:
        macro += 'URL GOTO='+RESTLINK+'?sort_by=elites_desc&start='+str(REV_NUM)+'\n'
        macro += 'WAIT SECONDS=5'+'\n'
    else:
        if EXTRACT==1: ##Check to make sure valid page
            macro += 'TAG POS=1 TYPE=DIV ATTR=CLASS:*reviewer-details EXTRACT=HTM'+'\n'            
        if EXTRACT==2: ##extract data
            macro += 'TAG POS=1 TYPE=DIV ATTR=ID:bizReviewsContent EXTRACT=HTM'+'\n'     

    return macro

def main(directory):
    import os
    import io
    from bs4 import BeautifulSoup
    import win32com.client
    import sys
    import re
    
    os.chdir(directory)    
    w = win32com.client.Dispatch('imacros')
    w.iimOpen('-fx',1,10)
    

    
    infile = io.open('yelp_restauraunts_by_ngh_final20131002.csv','r')
    inline = infile.readline()
    inline = infile.readline() ##start reading from line 2
    inline = inline.replace('\n','')    
    inline = inline.replace('"','')
    in_array = inline.split(',')
    
    RESTLINK = in_array[0]

    outfile = open('yelp_reviewers_by_ngh_20131003.csv','w')
    outfile.write('Restaurant Link,User Name,User Link,User City,User State,User Friends,Review Date,Review Rating'+'\n')
    REV_NUM=0            
    
    while len(RESTLINK)>0:
        macro = createMacro(RESTLINK,1,0,REV_NUM)
        flag = w.iimPlayCode(macro)        
        macro = createMacro(RESTLINK,0,1,REV_NUM)
        flag = w.iimPlayCode(macro)
        valid_page = w.iimGetLastExtract(1)
        
        while valid_page!='#EANF#':                      
            macro = createMacro(RESTLINK,0,2,REV_NUM)
            flag = w.iimPlayCode(macro)
            page_data = w.iimGetLastExtract(1)
#            print page_data
            
            rpage_soup = BeautifulSoup(page_data)    

#            print rpage_soup.prettify()
                    
            for result in rpage_soup.find_all("li",{"itemprop":"review"}):         
                data = {}
                user_data = result.find("li",{"class":"user-name"})        
                user_data = user_data.find("a")
                user_link = user_data.get("href")
                user_name = user_data.get_text()
                
                data['Username']=user_name
                data['User Link']=user_link        
                
                user_location = result.find("p",{"class":"reviewer_info"}).get_text()
                uloc_arr = user_location.split(',')
                try:
                    
                    data['City']=uloc_arr[0]                    
                    data['State']=uloc_arr[1]
                except IndexError:
                    data['City']=user_location
                    data['State']='N/A'
            
                user_friends = result.find("li",{"class": re.compile('friend-count.*')})        
                user_friends=user_friends.get_text().decode()
                user_friends = user_friends.replace('friends','')
                user_friends = user_friends.replace('friend','')    
                user_friends = user_friends.replace(' ','')    
                data['Friends']=user_friends 
                
                review_rating = result.find("meta",{"itemprop":"ratingValue"}).get("content")        
            
                review_date = result.find("meta",{"itemprop":"datePublished"}).get("content")
            
                data['Rating']=review_rating
                data['Date']=review_date
            
                write_data = RESTLINK+','+data['Username']+','+data['User Link']+','+data['City']+','+data['State']+','+data['Friends']+','+data['Date']+','+data['Rating']+'\n'
                outfile.write(write_data)
                REV_NUM+=1

            macro = createMacro(RESTLINK,1,0,REV_NUM)
            flag = w.iimPlayCode(macro)        
            macro = createMacro(RESTLINK,0,1,REV_NUM)
            flag = w.iimPlayCode(macro)
            valid_page = w.iimGetLastExtract(1)

        inline = infile.readline() ##start reading from line 2
        inline = inline.replace('\n','')    
        inline = inline.replace('"','')
        in_array = inline.split(',')
        RESTLINK = in_array[0]
        REV_NUM=0
         

main('D:\\Yelp')