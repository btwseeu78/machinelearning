# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:21:37 2017

@author: Apu
"""
import sys
T = int(input())
i = 0
while i < T:
        i = i+1
        N= int(input())
        inp = list(map(int,input().split(" ")))
        if(len(inp) != N):
            raise ValueError()
        counter = 0
        val = 0
        for i in inp :
            if ( (i * 2**counter) >= 2**(counter+1)):
                 temp = ((i * 2**counter) // (2**(counter+1)))
                 inp[counter] = 0
                 if((counter + 1) == len(inp)):
                     
                     inp.append(temp)
                     
                 else:
                    inp[(counter+1)] = inp[(counter+1)] + temp
                 counter = counter + 1
            else:
                if((counter) != (len(inp) + 1 )):
                    counter = counter + 1;
                else:
                    break
        
        
        for i in inp:
            val =  i + val
        print(val)
            
           
           
    
