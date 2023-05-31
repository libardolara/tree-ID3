
import pandas as pd
import numpy as np
import base as pp


class Node():
    '''
        Class for a Node in a Decision Tree
        A node has a Value and Childens
        A Child in a tupple of a condition(string) and a Node
    '''

    def __init__(self, value, parent ) -> None:
        '''
            Function construct a Node using a value
                value:      a Value (class for classification or the mean)
        '''
        self.value = value
        self.parent = parent
        self.children = []
        self.tagged = False
    
    def isLeaf(self):
        '''
            Function to determine if this Node is a Leaf
            Returns True of False if its a Leaf
        '''
        return len(self.children)==0
    
    def addChildren(self, condition, child):
        '''
            Function to add a children to this node
                condition:      a string defining the condition to reach the child
                child:          a Node.
        '''
        self.children.append((condition, child))
    
    def evaluate(self, x):
        '''
            Function to add a children to this node
                x:      a data point
            Returns the value of the sub-tree starting from this node
        '''
        if self.isLeaf():
            return self.value # Leaf return value
        else:
            feature_name = self.value
            for condition, child in self.children: # Evaluate each child
                feature_value = x[feature_name]
                if type(feature_value)==str or type(feature_value)==object:
                    if eval(f'"{feature_value}"{condition}'): # Evaluate condition for discrete values
                        return child.evaluate(x)
                else:
                    if eval(f'{feature_value}{condition}'): # Evaluate condition for continous values
                        return child.evaluate(x)
            return child.evaluate(x)
    
    def printTree(self, level=1):
        '''
            Function to print the sub-tree starting from this node
                level:      current level for printing
            Returns the subtree with the conditions and nodes
        '''
        if self.isLeaf():
            return f' -> {self.value}'
        else:
            space = '    '
            pntr = f' -> {self.value}\n'
            for condition, child in self.children:
                pntr += space*(level) +condition
                pntr += child.printTree(level+3) + '\n'
            return pntr
    
    def flat_tree(self):
        '''
            Function to flath the sub-tree into an array
            Returns all the nodes of the subtree in an array
        '''
        flaten = [] 
        for condition, a_child in self.children: 
            a_flaten = a_child.flat_tree() # Flat a child. Children first
            flaten.extend(a_flaten) 
        flaten.append(self) # Parent Last
        return flaten

    def prune_child(self,  child):
        '''
            Function to prune a child of the current node
                child:      A Node childe to be removed
            Returns child removed
        '''
        if not self.isLeaf():
            for index, (a_condition, a_child) in enumerate(self.children):
                if child == a_child:
                    return self.children.pop(index)
        
    def __repr__(self):
        '''
            Function to cast the Node as a String
            Returns a string with the Nodes value
        '''
        return self.value
    
    def __str__(self) -> str:
        '''
            Function to cast the Node as a String
            Returns a string with the Nodes value
        '''
        return self.value
        
class DecisionTree(pp.Model):
    '''
        Base class for a Decision Tree 
    '''

    def split_continuos(feature, target):
        '''
            Function to split a continuos feature near the middle of the sorted feature.
                feature:     a feature Series
                target:      a target Series
            Returns the value to split less or equal to
        '''
        sorted = np.array(np.argsort(feature)) # Sort the feature
        split_half = int(np.floor(len(feature)/2)) # Starting at the mid point
        split = split_half
        found_split = False
        index = 1
        while not found_split and index < 4: 
            if split-1 > 0: # Making sure the index is with in the range
                prev_i = sorted[split-1]
            else:
                prev_i = sorted[split]
            split_i = sorted[split]
            if split+1 < len(sorted): # Making sure the index is with in the range
                pos_i = sorted[split+1]
            else:
                pos_i = sorted[split]
            if target.iloc[prev_i]!= target.iloc[split_i]: # Split with the previous point
                return feature.iloc[prev_i]
            elif target.iloc[pos_i]!= target.iloc[split_i]: # Split with the posterior point
                return feature.iloc[split_i]
            else:
                split += (-1)**(index+1)*2*index # Sequence to review the the surroundings of the midpoint
                index += 1
                if split >= len(sorted):
                    split = len(sorted)-1
                if split < 0:
                    split = 0
        return feature.iloc[sorted[split_half]] # Default to mid point

    def flat_tree(self):
        '''
            Function to flat the decision tree into an array
            Returns all the nodes of the decision tree in an array
        '''
        return self.root.flat_tree()
    
    def pruneClassification(classTree, kwargs):
        '''
            Function to prune a Classification Decision tree
                classTree:     a Classification Decision Tree Model
                kwargs:        a dictionary with parameters to use
            Returns the Classification Tree pruned
        '''
        params = kwargs.copy()
        task = 'classification'
        metric = 'accuracy'
        dataset = params.pop('ds')
        verbose = params.pop('verbose', False) # Optional parameter for testing
        recursion = params.pop('recursion', True) # Optional parameter for testing
        X_val = dataset[4]
        y_val = dataset[5]
        flat_dt = classTree.flat_tree()
        init_score = pp.score(task, classTree.transform(X_val), y_val,metric)
        pruned = False
        index = 0
        while  index < len(flat_dt):
            node = flat_dt[index]
            if node.parent != None and not node.isLeaf():
                poped_child = node.parent.prune_child(node)
                pp.verboseprint(verbose,'Candidate',poped_child)
                # ADD A REPLACEMENT WITH THE TRAINING INSTANCES COVERED BY SUBTREE
                replacement = Node(node.value_counts.index[0], node.parent)
                replacement.value_counts = node.value_counts
                node.parent.addChildren(poped_child[0], replacement)
                # EVALUATE PERFROMANCE OF PRUNED
                _score = pp.score(task, classTree.transform(X_val), y_val, metric) 
                d_score = _score - init_score
                pp.verboseprint(verbose,'Difference without',d_score)
                if d_score < 0: 
                    # IF WORSE REMOVE REPLACEMENT AND LEAVE ORIGINAL NODE
                    pp.verboseprint(verbose,'Since its worse put back')
                    node.parent.prune_child(replacement)
                    node.parent.addChildren(poped_child[0], poped_child[1]) # 0 for condition 1 for node
                else:
                    pruned = True
                    DecisionTree.cleanPruned(flat_dt, poped_child[1].flat_tree())
                    index -= 1
                    init_score = _score # Update the score. Finding the max 
                    pp.verboseprint(verbose,'Since its not worse, prune')
            index += 1
        if pruned and recursion:
            # RETRY PRUNING
            return DecisionTree.pruneClassification(classTree, kwargs)
        return classTree
    
    def cleanPruned(flat_dt,pruned):
        for node in pruned:
            for index, flat_node in enumerate(flat_dt):
                if flat_node == node:
                    flat_dt.pop(index)
                    break
        return flat_dt

    def pruneRegression(rTree, kwargs):
        '''
            Function to prune a Regression Decision tree
                classTree:     a Regression Decision Tree Model
                kwargs:        a dictionary with parameters to use
            Returns the Regression Tree pruned
        '''
        params = kwargs.copy()
        task = 'regression'
        metric = 'mse'
        dataset = params.pop('ds')
        X_val = dataset[4]
        y_val = dataset[5]
        flat_dt = rTree.flat_tree()
        init_score = pp.score(task, rTree.transform(X_val), y_val,metric)
        pruned = False
        index = 0
        while  index < len(flat_dt):
            node = flat_dt[index]
            if node.parent != None and not node.isLeaf():
                poped_child = node.parent.prune_child(node)
                # ADD A REPLACEMENT WITH THE TRAINING INSTANCES COVERED BY SUBTREE
                replacement = Node(node.trimmed_value, node.parent)
                replacement.trimmed_value = node.trimmed_value
                replacement.trimmed_weight = node.trimmed_weight
                node.parent.addChildren(poped_child[0], replacement)
                # EVALUATE PERFROMANCE OF PRUNED
                _score = pp.score(task, rTree.transform(X_val), y_val, metric) 
                d_score = init_score - _score 
                if d_score < 0: 
                    # IF WORSE REMOVE REPLACEMENT AND LEAVE ORIGINAL NODE
                    node.parent.prune_child(replacement)
                    node.parent.addChildren(poped_child[0], poped_child[1]) # 0 for condition 1 for Node
                else:
                    pruned = True
                    DecisionTree.cleanPruned(flat_dt, poped_child[1].flat_tree())
                    index -= 1
                    init_score = _score # Update the score. Finding the minimum 
            index += 1
        if pruned:
            # RETRY PRUNING
            return DecisionTree.pruneRegression(rTree, kwargs)
        return rTree
    
class DecisionTreeClassifier(DecisionTree):
    '''
        Class for a Decision Tree Classifier
    '''

    def fit(self, X, y, params):
        '''
        Function to fit the model to the data
            X:          a Pandas Dataframe with the features
            y:          a Pandas Dataframe with the target
            params:     a Dictionary with the parameters to be used in the model
        Fits the model to the data
        '''
        super().fit(X, y, params)
        self.root = DecisionTreeClassifier.build_tree(X, y)
    
    def transform(self, X):
        '''
        Function to transform a set of data using the trained model
            X:          a Pandas Dataframe with the features to transform
        Returns the transformation of the features
        '''
        return X.apply(self.root.evaluate, axis=1)

    def entropy(partition_indexes, target):
        '''
            Function to calculate the entropy of a partition
                partition_indexes:     array with the indexes of the partiton
                target:      a target Series
            Returns the entropy
        '''
        Nm = len(partition_indexes) # Base length of the partition
        classes = target[partition_indexes].value_counts() # Count the classes ocurrances.
        pmi = classes/Nm # The ratio of classes in the partiton
        return -sum(pmi*np.log2(pmi))     

    def gainRatio(initial_e, feature, target):
        '''
            Function to calculate the Gain ration
                initial_e:      Initial entropy of the feature
                feature:        a feature Series
                target:         a target Series
            Returns the entropy
        '''
        Nm = len(feature)
        expeted_e = 0
        iv = 0
        if feature.dtypes == 'object': # Handle categorical variables
            # Expected Entropy
            divisions = feature.value_counts() # Dividing the feature in partitions
            for _value, _count in divisions.iteritems():
                pmj = _count/Nm # The ratio of a value in the partiton
                partition = feature[feature==_value].index # Indexes of the partition
                Ij = DecisionTreeClassifier.entropy(partition,target) # Entropy of the partition
                expeted_e += pmj*Ij
            # IV
            Dpj = divisions/Nm
            iv = -sum(Dpj*np.log2(Dpj)) 
        else: # Handle continuos variables
            # Expected Entropy
            split_value = DecisionTreeClassifier.split_continuos(feature,target) # Find the value to split the feature
            split_a = feature[feature <= split_value] # Less or equal than the split value
            split_b = feature[feature > split_value] # Greater than the split value
            pmj_a = len(split_a)/Nm #The ratio of one partiton
            pmj_b = len(split_b)/Nm #The ratio of one partiton
            Ij_a = DecisionTreeClassifier.entropy(split_a.index,target) # Entropy of a partition
            Ij_b = DecisionTreeClassifier.entropy(split_b.index,target) # Entropy of a partition
            expeted_e += pmj_a*Ij_a
            expeted_e += pmj_b*Ij_b
            # IV
            if pmj_a != 0:
                iv += pmj_a*np.log2(pmj_a)
            if pmj_b != 0:
                iv += pmj_b*np.log2(pmj_b)
            iv = -iv
            if iv==0:
                iv = 1 # Avoid dividing by zero
        try:
            response = (initial_e - expeted_e)/iv
        except:
            response = (initial_e - expeted_e)
        return response

    def build_tree(features, target, parent=None, condition = None):
        '''
            Function to build a tree for classification
                features:        a feature Data Frame
                target:         a target Series
                parent:         the parent Node, default None
                condition:      the condition to the parent node
            Returns the the root node with the Tree built
        '''
        if len(features) == 0:
            return parent

        if len(features.columns) == 0: # No more features. Done building
            child = Node(target.value_counts().index[0], parent) # Plurality vote
            child.value_counts = target.value_counts() # Save the count of classes
            parent.addChildren(condition , child)
            return parent
        
        initial_e = DecisionTreeClassifier.entropy(features.index,target) # Intial Entropy
        max_gainr = 0
        max_colm = None
        if initial_e == 0: # Pure feature. Done building
            child = Node(target.value_counts().index[0], parent) # Plurality vote
            child.value_counts = target.value_counts() # Save the count of classes
            parent.addChildren(condition , child)
            return parent
        # SEARCH FOR THE MAX GAIN RATIO
        for column in features.columns:
            feature = features[column]
            gainr = DecisionTreeClassifier.gainRatio(initial_e, feature, target)
            if gainr > max_gainr:
                max_gainr = gainr
                max_colm = column
        if max_colm is None: # No Gain
            max_colm = column # Pick any column
        if parent is None: # Starting with the root
            parent = Node(max_colm, None)
            parent.value_counts = target.value_counts() # Saving the prediction at each level
            new_parent = parent
        else:
            child = Node(max_colm, parent)
            child.value_counts = target.value_counts() # Save the count of classes
            parent.addChildren(condition , child) # Add child with condition
            new_parent = child
        feature = features[max_colm]
        new_features = features[[x for x in features.columns if x!=max_colm]] # Remove chosen feature from the next tree building
        if feature.dtypes == 'object': # Discrete variale
            branches = feature.value_counts()
            for branch, count in branches.iteritems(): # Build a tree for each branch of the chosen feature
                partition = feature==branch
                DecisionTreeClassifier.build_tree(new_features[partition], target[partition], new_parent, f'=="{branch}"' )
        else: # Continous variale
            # Build 2 trees for the binary split of the chosen feature
            split_value = DecisionTreeClassifier.split_continuos(feature,target)
            split_a = feature <= split_value
            split_b = feature > split_value
            DecisionTreeClassifier.build_tree(new_features[split_a], target[split_a], new_parent, f'<={split_value}' )
            DecisionTreeClassifier.build_tree(new_features[split_b], target[split_b], new_parent, f'>{split_value}' )
        return parent
    
class DecisionTreeRegressor(DecisionTree):

    def fit(self, X, y, params):
        '''
        Function to fit the model to the data
            X:          a Pandas Dataframe with the features
            y:          a Pandas Dataframe with the target
            params:     a Dictionary with the parameters to be used in the model
        Fits the model to the data
        '''
        super().fit(X, y, params)
        self.root = DecisionTreeRegressor.build_tree(X, y)
    
    def transform(self, X):
        '''
        Function to transform a set of data using the trained model
            X:          a Pandas Dataframe with the features to transform
        Returns the transformation of the features
        '''
        return X.apply(self.root.evaluate, axis=1)

    def squared_error(ground_truth, predicted):
        '''
        Function to calculate the squared error of two values
            ground_truth:      The ground truth value
            predicted:         The predicted value
        Returns the squared error
        '''
        return sum((ground_truth - predicted)**2)

    def Err(feature, target):
        '''
        Function to calculate the error from a feature and possible partitions
            feature:      a Series with the feature
            target:       a Series with the target
        Returns the squared error
        '''
        Nm = len(feature)
        error = 0
        if feature.dtypes == 'object': # Handle categorical variables
            # Minimize the Squared Error
            divisions = feature.value_counts() # Dividing the feature in branches
            for _value, _count in divisions.iteritems():
                partition = feature[feature==_value].index # Indexes of the partition
                se = DecisionTreeRegressor.squared_error(target[partition], target[partition].mean())
                error += se
        else: # Handle continuos variables
            # Minimize the Squared Error
            split_value = DecisionTreeRegressor.split_continuos(feature) # Find the value to split the feature
            split_a = feature < split_value # Less or equal than the split value
            split_b = feature >= split_value # Greater than the split value

            se_a = DecisionTreeRegressor.squared_error(target[split_a], target[split_a].mean()) # Squared Error of a branch
            se_b = DecisionTreeRegressor.squared_error(target[split_b], target[split_b].mean()) # Squared Error of a branch
            error += se_a
            error += se_b
        return error/Nm
    
    def split_continuos(feature):
        '''
            Function to split a continuos feature
                feature:     a feature Series
            Returns the value to split less or equal to
        '''
        sorted = np.array(np.argsort(feature)) # Sort the feature
        split_half = int(np.floor(len(feature)/2)) # Starting at the mid point
        return feature.iloc[sorted[split_half]] # Default to mid point
    
    def build_tree(features, target, parent=None, condition = None):
        '''
            Function to build a tree for regression
                features:        a feature Data Frame
                target:         a target Series
                parent:         the parent Node, default None
                condition:      the condition to the parent node
            Returns the the root node with the Tree built
        '''
        if len(features) == 0:
            return parent

        if len(features.columns) == 0: # No more features. Done building
            child = Node(target.mean(), parent) # Mean
            child.trimmed_value = target.mean() # Save the trimm value
            child.trimmed_weight = len(target) # Weight of the value
            parent.addChildren(condition , child)
            return parent
        
        min_error = target.max()**2
        min_colm = None
        # SEARCH FOR THE MIN ERROR
        for column in features.columns:
            feature = features[column]
            mean_se = DecisionTreeRegressor.Err(feature,target)
            if min_error > mean_se:
                min_error = mean_se
                min_colm = column
        if min_colm is None: # No Error?
            min_colm = column # Pick any column
        if parent is None: # Startig with the root
            parent = Node(min_colm, None)
            # Saving the prediction at each level
            parent.trimmed_value = target.mean() # Save the trimm value
            parent.trimmed_weight = len(target) # Weight of the value
            new_parent = parent
        else:
            child = Node(min_colm, parent)
            child.trimmed_value = target.mean() # Save the trimm value
            child.trimmed_weight = len(target) # Weight of the value
            parent.addChildren(condition , child) # Add child with condition
            new_parent = child
        feature = features[min_colm]
        new_features = features[[x for x in features.columns if x!=min_colm]] # Remove chosen feature from the next tree building
        if feature.dtypes == 'object': # Discrete variale
            branches = feature.value_counts()
            for branch, count in branches.iteritems(): # Build a tree for each branch of the chosen feature
                partition = feature==branch
                DecisionTreeRegressor.build_tree(new_features[partition], target[partition], new_parent, f'=="{branch}"' )
        else: # Continous variale
            # Build 2 trees for the binary split of the chosen feature
            split_value = DecisionTreeRegressor.split_continuos(feature)
            split_a = feature < split_value
            split_b = feature >= split_value
            DecisionTreeRegressor.build_tree(new_features[split_a], target[split_a], new_parent, f'<{split_value}' )
            DecisionTreeRegressor.build_tree(new_features[split_b], target[split_b], new_parent, f'>={split_value}' )
        return parent