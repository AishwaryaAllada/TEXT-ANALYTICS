import sys
from test import create_classifer
import csv


if __name__ == "__main__":
  lab = sys.argv[1]
  tr_sw = sys.argv[2]
  val_sw = sys.argv[3]
  test_sw = sys.argv[4]
  tr_withoutSW = sys.argv[5]
  val_withoutSW = sys.argv[6]
  test_withoutSW = sys.argv[7]
  labs = []
  with open(lab, "r") as f:
    for line in f:
      labs.append(line)

  labelss = labs[: len(labs) - 1] 
  labelss = [x.replace('\n', '') for x in labelss]

  train_labels = labelss[:640000]
  test_labels = labelss[640000:720000]
  val_labels = labelss[720000:]

 
  with open(tr_sw, 'r') as f:
    file = csv.reader(f)
    train_WithSW = list(file)

  train_WithSW_xx = [' '.join(x) for x in train_WithSW]

  with open(val_sw, 'r') as f:
    file = csv.reader(f)
    val_WithSW = list(file)

  val_WithSW_xx = [' '.join(x) for x in val_WithSW]

  with open(test_sw, 'r') as f:
    file = csv.reader(f)
    test_WithSW = list(file)

  test_WithSW_xx = [' '.join(x) for x in test_WithSW]

  with open(tr_withoutSW, 'r') as f:
    file = csv.reader(f)
    train_WithoutSW = list(file)

  train_WithoutSW_xx = [' '.join(x) for x in train_WithoutSW]

  with open(val_withoutSW, 'r') as f:
    file = csv.reader(f)
    val_WithoutSW = list(file)

  val_WithoutSW_xx = [' '.join(x) for x in val_WithoutSW]

  with open(test_withoutSW, 'r') as f:
    file = csv.reader(f)
    test_WithoutSW = list(file)

  test_WithoutSW_xx = [' '.join(x) for x in test_WithoutSW]



create_classifer((1,1),train_WithSW_xx,train_labels,val_WithSW_xx,val_labels,test_WithSW_xx,test_labels) #uni - with
create_classifer((2,2),train_WithSW_xx,train_labels,val_WithSW_xx,val_labels,test_WithSW_xx,test_labels) #bi - with
create_classifer((1,2),train_WithSW_xx,train_labels,val_WithSW_xx,val_labels,test_WithSW_xx,test_labels) #uni+bi - with
create_classifer((1,1),train_WithoutSW_xx,train_labels,val_WithoutSW_xx,val_labels,test_WithoutSW_xx,test_labels) #uni - without
create_classifer((2,2),train_WithoutSW_xx,train_labels,val_WithoutSW_xx,val_labels,test_WithoutSW_xx,test_labels) #bi - without
create_classifer((1,2),train_WithoutSW_xx,train_labels,val_WithoutSW_xx,val_labels,test_WithoutSW_xx,test_labels) #uni+bi - without