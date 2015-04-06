# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:11:53 2013

@author: tdong
"""

def createMacro(NEW_PAGE=1,R_NUM=1,REST_NUM=0,NGH='MA:Boston::Allston/Brighton'):
    macro = 'SET !TIMEOUT_PAGE 10' + '\n'
    macro += 'TAB T=1'+'\n'
    macro += 'SET !TIMEOUT_STEP 0' + '\n'
    if NEW_PAGE == 1:
        macro += 'URL GOTO=http://www.yelp.com/search?find_desc=restaurants&find_loc=Boston%2C+MA&ns=1#start='+str(REST_NUM)+'&l=p:'+NGH+'\n'
        macro += 'WAIT SECONDS=5'+'\n'
        macro += 'TAG POS=1 TYPE=H3 ATTR=CLASS:search-result-title EXTRACT=TXT'+'\n'
    else:
        macro += 'TAG POS=1 TYPE=UL ATTR=CLASS:ylist* EXTRACT=HTM'+'\n'
        macro += 'WAIT SECONDS=2'+'\n'
    return macro

def getNeighborhoodList(directory):
    import os
    import io
    from bs4 import BeautifulSoup
    import win32com.client
    import sys
    import re
    
    os.chdir(directory)    

#    w = win32com.client.Dispatch('imacros')
#    w.iimOpen('-fx',1,10)    
    infile = open('neighborhoodstoparse.txt','r')
    outfile = io.open('ngh_list.csv','w',encoding='utf-16')
    
    ngh_content = infile.readline()
    ngh_soup = BeautifulSoup(ngh_content)
    for ngh_item in ngh_soup.find_all("input", {"name": "place"}):
        outfile.write(ngh_item.get('value')+u'\n')
    
    
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
#    
    outfile = io.open('yelp_restaurants_by_ngh.csv','w',encoding='utf-16')
    outfile.write(u'Restaurant Link,Restaurant Name,Neighborhood,Restaurant Address,City,State,Phone,Price,Star Rating,Number of Reviews,Category 1,Category 2,Category 3'+u'\n')    
#
    REST_NUM=0
    NEW_PAGE=1
    R_NUM=1    
    
    ngh_file = io.open('ngh_list_remainder.csv','r',encoding='utf-16')

    NGH = ngh_file.readline()
    NGH = NGH.replace('\n','')        

    NEW_PAGE=1
    macro = createMacro(NEW_PAGE,R_NUM,REST_NUM,NGH)
    flag = w.iimPlayCode(macro) 
        


    while len(NGH)>0:  
        print NGH
        while w.iimGetLastExtract(1)!='#EANF#':
            print w.iimGetLastExtract(1)
            NEW_PAGE=0
            macro = createMacro(NEW_PAGE,R_NUM,REST_NUM,NGH)
            flag = w.iimPlayCode(macro) 
            results_content = w.iimGetLastExtract(1)
            print 'new page'
            results_soup = BeautifulSoup(results_content)
    
            for result_soup in results_soup.find_all("div",{"class": re.compile('search-result.*')}):                
##                print result_soup.prettify()
                result = {}
                
                try:
                    link = result_soup.find("a", {"class": "biz-name"})
                    new_link = 'http://www.yelp.com'    
                    new_link = new_link + link.get('href')
                except AttributeError:
                    new_link = ''    
                result['link']=new_link                    
                    
                try:    
                    rest_name = link.get_text() 
                    rest_name = rest_name.replace(',','')                    
                except AttributeError:
                    rest_name = ''
                result['rest_name']=rest_name
                
                try:
                    rating = result_soup.find("i",{"class": re.compile('star-img.*')})                
                    new_rating = rating.get('title')
                    new_rating = new_rating.replace('star rating','')
                    new_rating = new_rating.strip()    
                except AttributeError:
                    new_rating = ''
                result['rating']=new_rating
                
                try:                    
                    num_review = result_soup.find("span",{"class": re.compile('review-count.*')})
                    review_num = num_review.get_text()
                    review_num = review_num.replace('reviews','')
                    review_num = review_num.replace('review','')                    
                    review_num = review_num.strip()
                except AttributeError:
                    review_num= ''                    
                result['num_reviews']=review_num
    
                try: 
                    price = result_soup.find("span",{"class": re.compile('business-attribute.*')})
                    new_price = price.get_text()
                except AttributeError:
                    new_price = ''
                result['price']=new_price
                
                try:
                    category = result_soup.find("span",{"class": "category-str-list"})
                    cats = category.get_text()
                    cats = cats.replace('\n','')
                    cats = cats.strip()
                    cat_arr = cats.split(',')   
                    text = ''
                    for item in cat_arr:
                        temp = item.strip()
                        text = text+temp+','          
                    array_len = len(text)-1
                    text = text[:array_len]
                except AttributeError:
                    text = ''
                result['category']=text
    
                
                try:
                    neighborhood = result_soup.find("span",{"class": "neighborhood-str-list"})
                    nghs = neighborhood.get_text()
                    nghs = nghs.replace('\n','')
                    nghs = nghs.strip()
                    nghs_arr = nghs.split(',')   
                    text = ''
                    for item in nghs_arr:
                        temp = item.strip()
                        text = text+temp+','        
                        array_len = len(text)-1
                        text = text[:array_len]
                except:
                    text = ''
                result['neighborhood']=text
            
                try:
                    address = result_soup.find("address")
                    new_address = ''
                    for e in address.recursiveChildGenerator():
                        if isinstance(e, basestring):
                            new_address += e.strip()
                        elif e.name == 'br':
                            new_address += '\n'
                    add_array = new_address.split('\n')
                    new_address=add_array[0]
                    new_address=new_address.replace(',','')
                    cit_array = add_array[1].split(',')
                    new_city=cit_array[0].strip()
                    new_state=cit_array[1].strip()
                except AttributeError:
                    new_address = new_address
                    new_city = 'N/A'
                    new_state = 'N/A'
                except IndexError:
                    new_address = new_address
                    new_city = 'N/A'
                    new_state = 'N/A'
                result['address']=new_address
                result['city']=new_city
                result['state']=new_state
                    
                try:
                    phone = result_soup.find("span",{"class": "biz-phone"})
                    phone_text = phone.get_text()
                    phone_text =phone_text.strip()
                except AttributeError:
                    phone_text = ''
                result['phone']=phone_text          
    
                write_data =result['link']+','+result['rest_name']+','+result['neighborhood']+','+result['address']+','+result['city']+','+result['state']+','+result['phone']+','+result['price']+','+result['rating']+','+result['num_reviews']+','+result['category']+'\n' 
                
                outfile.write(unicode(write_data))
                print write_data
                R_NUM+=1
                REST_NUM+=1                

            ##Separate restaurant and new page counter
            NEW_PAGE=1
            macro = createMacro(NEW_PAGE,R_NUM,REST_NUM,NGH)
            flag = w.iimPlayCode(macro) 
      
        
        REST_NUM=0
        R_NUM=1
        NEW_PAGE=1
        NGH = ngh_file.readline()
        NGH = NGH.replace('\n','')    
        macro = createMacro(NEW_PAGE,R_NUM,REST_NUM,NGH)
        flag = w.iimPlayCode(macro)    

    outfile.close()
    
main('D:\\Yelp')
getNeighborhoodList('D:\\Yelp')