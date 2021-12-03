# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 12:24:09 2021

@author: MyPC
"""
age = int(input()) 
gender = int(input()) 
score = int(input())

class HUMAN : 
  def __init__(self, age, gender):
    self.age = age 
    self.gender = gender
    self.age >0 and self.age <= 160 
    self.gender = "Male" or self.gender = "Female" 

  def eat (selff) : 
    self.age = "eat" 
  def sleep (self) : 
    self.ape = "sleep" 
  def run (self) : 
    self.age = "run" 

class STUDENT (HUMAN) : 
  def __init__ (self, age, gender, score) : 
    self.age > 18 and self.age < 28 
    self.score = score 
  def check_score(self) : 
    if score > 3.8 and score < 4: 
      return "A+"
    if score > 3.3 and score < 3.5: 
      return "A" 
    if score > 3 and score < 3.2: 
     return "B+"
    if score > 2.6 and score < 2.9: 
     return "B"   
    if score > 1.8 and score < 2.5: 
     return "C"
    if score > 1.8:
     return "D"