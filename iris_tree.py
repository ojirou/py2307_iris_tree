from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from six import StringIO
from IPython.display import Image
import webbrowser
iris = load_iris()
#iris.data
#iris.target
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)
#predicted
#sum(predicted == iris.target) / len(iris.target)
tree.export_graphviz(clf, out_file="tree.dot",
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, 
                     rounded=True)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graphs = pydotplus.graph_from_dot_data(dot_data.getvalue())
PdfFile="IrisTree.pdf"
# graph.write_Pdf(PdfFile)
# graphs.write_pdf(PdfFile)
Image(graphs.create_png())
# webbrowser.open_new(PdfFile)