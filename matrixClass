# James Jasinski
# 2/12/19

import math
import string

# This class that represents a matrix that can perform "matrix calculator" functions 

class Matrix( object ) :

    # A matrix is represented by dimensions [rows x columns] and the elements in each row and column

    def __init__( self, numRows, numColumns, listOfRows ) :

        # Attributes

        self.numRows = numRows                                  
        self.numColumns = numColumns

        # Checks if numRows is not equals to the number of rows or if numColumns is not equal to the length of each row; 
        # otherwise the matrix creates itself

        if self.sameNumberRows(listOfRows) == False or self.sameLengthRows(listOfRows) == False :
            raise dimensionError("Oops! Something is wrong with the dimensions of your matrix or your dimension values!")
        else : 

            # Initialize matrix of zeroes with size ( numRows x numColumns )
     
            self.matrix = [ [0 for n in range( 0, self.numColumns )] for m in range( 0, self.numRows ) ]

            # Replaces zeroes with corresponding values from the listOfRows

            for i in range( 0, self.numRows ) :
                for j in range( 0, self.numColumns ) :
                    self.matrix[i][j] = listOfRows[i][j]


    # Utility function for the constructor: it checks if every row in the listOfRows is equal to the numColumns 

    def sameLengthRows( self, listOfRows ) :
        for i in range( 0, len(listOfRows) ) :
            if len(listOfRows[i]) != self.numColumns :
                return False
        return True


    # Utility function for the constructor: it checks if the numRows is equal to the size of the listOfRows 

    def sameNumberRows( self, listOfRows ) :
        if self.numRows == len(listOfRows) :
            return True
        else :
            return False


    # Returns ordered pair representing dimensions of the matrix (rows, columns)

    def size( self ) :   
        return ( self.numRows, self.numColumns )


    # Converts the matrix into a string for printing

    def __str__( self ) :

        mat = "["
        for i in range( 0, self.numRows ) :
            mat = mat + "["
            for j in range( 0, self.numColumns ) :
                mat = mat + str(self.matrix[i][j])
                if j != self.numColumns-1 :
                    mat = mat + ","
            if i == self.numRows-1 :
                mat = mat + "]"
            else :
                mat = mat + "], \n "
        mat = mat + "]"

        return mat


    # Returns nothing if you can't add the matrices, otherwise return a new matrix representing self+otherMatrix

    def add( self, otherMatrix ) :

        # Checks if matrices have the same dimensions

        if self.numRows == otherMatrix.numRows and self.numColumns == otherMatrix.numColumns :

            # Creates a new list of zeroes in the form of self's transpose (n x m)

            newList = [ [0 for m in range( 0, self.numColumns )] for n in range( 0, self.numRows ) ]

            # Creates new matrix for the sum

            total = Matrix( self.numRows, self.numColumns, newList )

            # Replaces values with the sum of corresponding values in self and the otherMatrix

            for i in range( 0, total.numRows ) :
                for j in range( 0, total.numColumns ) :
                    total.matrix[i][j] = self.matrix[i][j] + otherMatrix.matrix[i][j]

            return total

        else :
            return


    # Returns nothing if you can't add the matrices, otherwise return a new matrix representing the sum of 
    # the listOfMatrices and self 

    def sum( self, listofMatrices ) :
        numSame = 0
        for x in range( 0, len(listofMatrices) ) :
            if self.numRows == listofMatrices[x].numRows and self.numColumns == listofMatrices[x].numColumns :
                numSame = numSame+1
        
        if numSame == len(listofMatrices) :
            sum = Matrix( self.numRows, self.numColumns, self.matrix )
            for i in range( 0, len(listofMatrices) ) :
                sum = sum.add(listofMatrices[i])   
            return sum
        else :
            return 

    
    # Returns nothing if you can't subtract the matrices, otherwise return a new matrix representing self-otherMatrix 

    def subtract( self, otherMatrix ) :

        # Checks if matrices have the same dimensions

        if self.numRows == otherMatrix.numRows and self.numColumns == otherMatrix.numColumns :
            
            # Creates a new list of zeroes in the form of self's transpose (n x m)

            newList = [ [0 for m in range( 0, self.numColumns )] for n in range( 0, self.numRows ) ]

            # Creates new matrix for the difference

            difference = Matrix( self.numRows, self.numColumns, newList )

            # Replaces values with the difference of corresponding values in self and the otherMatrix

            for i in range( 0, difference.numRows ) :
                for j in range( 0, difference.numColumns ) :
                    difference.matrix[i][j] = self.matrix[i][j] - otherMatrix.matrix[i][j]

            return difference

        else :
            return


    # Returns nothing if you can't multiply the matrices, otherwise return a new matrix representing self*otherMatrix 

    def multiply( self, otherMatrix ) : 

        # Checks if self's numColumns is equal to otherMatrix's numRows

        if self.numColumns == otherMatrix.numRows :

            # Saves the number of columns in self/the number of rows in the otherMatrix

            selfNumColumns = self.numColumns

            # Creates a new list of zeroes in the form of the product (n x m)

            newList = [ [0 for m in range( 0, otherMatrix.numColumns )] for n in range( 0, self.numRows ) ]

            # Creates new matrix of their product

            product = Matrix( self.numRows, otherMatrix.numColumns, newList )

            # Replaces zeroes with the corresponding values in self (using listSum)

            for i in range( 0, product.numRows ) :
                for j in range( 0, product.numColumns ) :
                    product.matrix[i][j] = self.listSum(otherMatrix, i, j, selfNumColumns)

            return product 


    # Utility function for function multiply: calculates the term "product.matrix[i][j]" by multiplying term-by-term 
    # the entries of the ith row of self and the jth column of the otherMatrix

    def listSum( self, otherMatrix, row, column, numberOfTerms ):

        # Creates list and fills it with the corresponding terms 

        list1 = []
        for x in range( 0, numberOfTerms ) :
            list1.append( self.matrix[row][x]*otherMatrix.matrix[x][column] )

        # Takes the sum of the terms

        sum = 0
        for z in list1:
            sum = sum + z

        return sum


    # Returns a new matrix representing x*self for some integer x 

    def scalarMultiply( self, scalar ) :

        # Creates a new list of zeroes in the form of self's transpose (n x m)

        newList = [ [0 for m in range( 0, self.numColumns )] for n in range( 0, self.numRows ) ]

        # New matrix

        product = Matrix( self.numRows, self.numColumns, newList )

        # Replaces values with the product of the corresponding values in self and some scalar x 

        for i in range( 0, product.numRows ) :
            for j in range( 0, product.numColumns ) :
                product.matrix[i][j] = self.matrix[i][j] * scalar

        return product


    # Returns nothing if you can't take the transpose, otherwise return a new matrix representing self's transpose

    def transpose( self ) :

        # Creates a new list of zeroes in the form of self's transpose (n x m)

        newList = [ [0 for m in range( 0, self.numRows )] for n in range( 0, self.numColumns ) ]

        # Creates new matrix for transpose

        transpose = Matrix( self.numColumns, self.numRows, newList )

        # Replaces zeroes with the corresponding values in self

        for i in range( 0, transpose.numRows ) :
            for j in range( 0, transpose.numColumns ) :
                transpose.matrix[i][j] = self.matrix[j][i] 
        return transpose

    
    # Row Operations :

    # The scale method multiplies all elements in row i by a and returns the resulting matrix

    def scale( self, i, a ) :

        # Creates new matrix to return

        scaledMatrix = Matrix(self.numRows, self.numColumns, self.matrix)

        # Scales row i by a

        for x in range(0, len(self.matrix[i])) :
            scaledMatrix.matrix[i][x] = self.matrix[i][x]*a

        return scaledMatrix


    # The swap method interchanges rows i and j and returns the resulting matrix

    def swap( self, i, j ) :

        # Creates new matrix to return

        swappedMatrix = Matrix(self.numRows, self.numColumns, self.matrix)

        # Keeps track of row i

        rowI = self.matrix[i]

        # Swaps row i with row j, then swaps row j with the saved row i

        for x in range(0, len(self.matrix[i])) :
            swappedMatrix.matrix[i][x] = self.matrix[j][x]
        for y in range(0, len(self.matrix[i])) :
            swappedMatrix.matrix[j][y] = rowI[y]

        return swappedMatrix


    # The combine method adds a times row j to row i and returns the resulting matrix.

    def combine( self, i, j, a ) :

        # Creates new matrix to return

        combinedMatrix = Matrix(self.numRows, self.numColumns, self.matrix)

        # Creates row to combine (row j * a)

        rowToCombine = [0 for n in range( 0, len(self.matrix[j]) )]
        for x in range(0, len(self.matrix[j])) :
            rowToCombine[x] = self.matrix[j][x]*a

        # Adds created combined row to row i

        for y in range(0, len(self.matrix[i])) :
            combinedMatrix.matrix[i][y] = combinedMatrix.matrix[i][y] + rowToCombine[y]

        return combinedMatrix


# Class representing a square matrix 

class SquareMatrix( Matrix ) :

    # Initializes a matrix with n x n dimensions

    def __init__( self, n, listOfRows ) :
        Matrix.__init__( self, n, n, listOfRows)

    
    # Calculates the determinant of the matrix by using elementary row operations to get the matrix in upper
    # triangular form, then multiply the values of the diagonal

    def determinant( self ) :
        i = 0
        j = 0
        determinant = 1

        # Loops through matrix until the last row or the last column of the matrix is processed

        while i < self.numRows and j < self.numColumns :

            count = 0

            # If self.matrix[i][j] is equal to zero, the row is swapped with some other row below to guarantee that
            # it is not equal to zero

            if self.matrix[i][j] == 0 :

                # Finds the row below with the greatest value in the column

                maxNonZeroRow = i+1
                for x in range( i+1, self.numRows ) :
                    if self.matrix[x][j] != 0 :
                        if abs(self.matrix[x][j]) >= abs(self.matrix[maxNonZeroRow][j]) :
                            maxNonZeroRow = x
                        else :
                            pass
                    else :
                        count = count + 1

                # If they are all zero, move onto the next column

                if count == self.numRows-(i+1) :
                    j = j + 1

                # Otherwise, we swap the rows, negate the determinant, and eliminate all other values in 
                # the j column, below the current row, by subtracting the proper multiple of the i row from 
                # the other rows

                else :
                    self = self.swap( i, maxNonZeroRow )
                    determinant = determinant*-1
                    for y in range( i+1, self.numRows ) :
                        a = -self.matrix[y][j]/self.matrix[i][j]
                        self = self.combine(y, i, a)
                    i = i + 1
                    j = j + 1

            # If self.matrix[i][j] is not equal to zero, we just eliminate all other values in the j column 
            # by subtracting the proper multiple of the i row from the other rows

            else :
                for z in range( i+1, self.numRows ) :
                    a = -self.matrix[z][j]/self.matrix[i][j]
                    self = self.combine(z, i, a)
                i = i + 1
                j = j + 1

        # To calculate the determinant, we multiply the entries on the diagonal since the matrix is in 
        # upper triangular form
        
        for z in range(0, self.numRows) :
            determinant = determinant * self.matrix[z][z]

        return determinant 
            

    # Returns true if the matrix is invertible; otherwise it returns false

    def isInvertible( self ) :
        det = self.determinant()
        if det != 0 :          # Invertible if the determinant is non-zero
            return True
        else :
            return False


    # Calculates the inverse of the matrix

    def inverse( self ) :

        # If not invertible, we raise an error

        if self.isInvertible() == False :
            raise notInvertibleError("Sorry, the matrix you have is not invertible.")

        # Otherwise, we concatenate the matrix and its identity, then use elementary row operations to get
        # the matrix into reduced row-echelon form. If the matrix can be reduced to the identity matrix, 
        # then the inverse is the matrix on the right of the transformed, concatenated matrix
        
        else :

            # Create the identity matrix

            identity = SquareMatrix(self.numRows, self.matrix)
            for a in range( 0, identity.numRows ) :
                for b in range( 0, identity.numRows ) :
                    if a == b :
                        identity.matrix[a][b] = 1
                    else : 
                        identity.matrix[a][b] = 0

            # Create the concatenated matrix, with the identity on the right of the matrix

            rows = [ [0 for m in range(0, self.numColumns*2)] for n in range(0, self.numRows) ]
            concatMatrix = Matrix(self.numRows, self.numColumns*2, rows )
            for c in range( 0, self.numRows ) :
                for d in range( 0, self.numColumns) :
                    concatMatrix.matrix[c][d] = self.matrix[c][d]
            for e in range( 0, identity.numRows ) :
                for f in range( 0, identity.numColumns) :
                    concatMatrix.matrix[e][f + self.numColumns] = identity.matrix[e][f]

            i = 0
            j = 0

            # Loops through matrix until the last row or the last column of the matrix is processed

            while i < self.numRows and j < self.numColumns :

                count = 0

                # If concatMatrix.matrix[i][j] is equal to zero, the row is swapped with some other row below 
                # to guarantee that it is not equal to zero

                if concatMatrix.matrix[i][j] == 0 :

                    # Finds the row below with the greatest value in the column

                    maxNonZeroRow = i+1
                    for x in range( i+1, concatMatrix.numRows ) :
                        if concatMatrix.matrix[x][j] != 0 :
                            if abs(concatMatrix.matrix[x][j]) >= abs(concatMatrix.matrix[maxNonZeroRow][j]) :
                                maxNonZeroRow = x
                            else :
                                pass
                        else :
                            count = count + 1

                    # If they are all zero move onto the next column

                    if count == concatMatrix.numRows-(i+1) :
                        j = j + 1

                    # Otherwise, swap the rows, scale the row so the leading value is 1, then eliminate all other 
                    # values in the j column, above and below the current row, by subtracting the proper multiples 
                    # of the i row from the other rows

                    else :
                        concatMatrix = concatMatrix.swap( i, maxNonZeroRow )
                        concatMatrix = concatMatrix.scale( i, 1/concatMatrix.matrix[i][j] )

                        for y in range( 0, i ) :
                            if concatMatrix.matrix[y][j] == 0 :
                                pass
                            else :
                                a = -concatMatrix.matrix[y][j]/concatMatrix.matrix[i][j]
                                concatMatrix = concatMatrix.combine(y, i, a)

                        for z in range( i+1, concatMatrix.numRows ) :
                            if concatMatrix.matrix[z][j] == 0 :
                                pass
                            else :
                                b = -concatMatrix.matrix[z][j]/concatMatrix.matrix[i][j]
                                concatMatrix = concatMatrix.combine(z, i, b)

                        i = i + 1
                        j = j + 1

                # If concatMatrix.matrix[i][j] is not equal to zero, we just eliminate all other values in the 
                # j column by subtracting the proper multiple of the i row from the other rows

                else :
                    concatMatrix = concatMatrix.scale( i, 1/concatMatrix.matrix[i][j] )

                    for y in range( 0, i ) :
                        if concatMatrix.matrix[y][j] == 0 :
                            pass
                        else :
                            a = -concatMatrix.matrix[y][j]/concatMatrix.matrix[i][j]
                            concatMatrix = concatMatrix.combine(y, i, a)

                    for z in range( i+1, concatMatrix.numRows ) :
                        if concatMatrix.matrix[z][j] == 0 :
                            pass
                        else :
                            b = -concatMatrix.matrix[z][j]/concatMatrix.matrix[i][j]
                            concatMatrix = concatMatrix.combine(z, i, b)

                    i = i + 1
                    j = j + 1

            # Then, create a matrix for the inverse, and fill it with the values from the second part of the
            # concatenated matrix

            rows2 = [ [0 for m in range(0, identity.numColumns)] for n in range(0, identity.numRows) ]
            inverse = SquareMatrix(identity.numRows, rows2 )
            for n in range(0, identity.numRows) :
                for m in range(0, identity.numColumns) :
                    inverse.matrix[n][m] = concatMatrix.matrix[n][m + self.numColumns]

            return inverse


# Class representing a column vector

class ColumnVector( Matrix ) :

    # Initializes a matrix with n rows and only one column 

    def __init__( self, n, listOfElements ) :

        listOfRows = []
        for i in range(0, len(listOfElements)):
            listOfRows.append( [listOfElements[i]] )

        Matrix.__init__( self, n, 1, listOfRows )


    # Calculates the dot product of two column vectors by multiplying corresponding entries in the vectors,
    # and then adding them together.

    def dotProduct( self, otherVector ) :
        product = 0
        if self.numRows == otherVector.numRows :
            for i in range(0, self.numRows) :
                product = product + self.matrix[i][0]*otherVector.matrix[i][0]
            return product
        else :
            return 


# Class representing a system of linear equations

class LinearSystem( object ) :

    # Initializes linear system to have a matrix representing it

    def __init__( self, variables, equations, coefficients ) :

        self.vars = variables
        self.eqns = equations

        # One extra column for what the equations equal

        listOfRows = [ [0 for m in range( 0, self.vars+1 )] for n in range( 0, self.eqns ) ]

        for i in range(0, self.eqns) :
            for j in range(0, self.vars+1) :
                listOfRows[i][j] = coefficients[i][j]

        self.linearSys = Matrix( self.eqns, self.vars+1, listOfRows )


    # Converts the matrix representing the linear system into a string for printing
    
    def __str__( self ) :

        alphabet = list( string.ascii_lowercase )
        sys = ''
        for i in range( 0, self.eqns ) :
            for j in range( 0, self.vars+1 ) :
                if j == 0 :
                    sys = sys + str(self.linearSys.matrix[i][j]) + str(alphabet[j])
                elif j == self.vars :
                    sys = sys + ' = ' + str(self.linearSys.matrix[i][j])
                elif self.linearSys.matrix[i][j] >= 0 :
                    sys = sys + ' + ' + str(self.linearSys.matrix[i][j]) + str(alphabet[j])
                else :
                    sys = sys + ' - ' + str(abs(self.linearSys.matrix[i][j])) + str(alphabet[j])
            if i < self.eqns :
                sys = sys + '\n'

        return sys 


    # Solves a linear system (by getting it into reduced row echelon form, similar to inverse) and 
    # returns a column vector of the solutions

    def solve( self ) :
        i = 0
        j = 0

        while i < self.linearSys.numRows and j < self.linearSys.numColumns-1 :

            count = 0
            if self.linearSys.matrix[i][j] == 0 :

                maxNonZeroRow = i+1
                for x in range( i+1, self.linearSys.numRows ) :
                    if self.linearSys.matrix[x][j] != 0 :
                        if abs(self.linearSys.matrix[x][j]) >= abs(self.linearSys.matrix[maxNonZeroRow][j]) :
                            maxNonZeroRow = x
                        else :
                            pass
                    else :
                        count = count + 1

                if count == self.linearSys.numRows-(i+1) :
                    j = j + 1
                else :
                    self.linearSys = self.linearSys.swap( i, maxNonZeroRow )

                    self.linearSys = self.linearSys.scale( i, 1/self.linearSys.matrix[i][j] )

                    for y in range( 0, i ) :
                        if self.linearSys.matrix[y][j] == 0 :
                            pass
                        else :
                            a = -self.linearSys.matrix[y][j]/self.linearSys.matrix[i][j]
                            self.linearSys = self.linearSys.combine(y, i, a)

                    for z in range( i+1, self.linearSys.numRows ) :
                        if self.linearSys.matrix[z][j] == 0 :
                            pass
                        else :
                            a = -self.linearSys.matrix[z][j]/self.linearSys.matrix[i][j]
                            self.linearSys = self.linearSys.combine(z, i, a)

                    i = i + 1
                    j = j + 1

            else :
                self.linearSys = self.linearSys.scale( i, 1/self.linearSys.matrix[i][j] )

                for y in range( 0, i ) :
                        if self.linearSys.matrix[y][j] == 0 :
                            pass
                        else :
                            a = -self.linearSys.matrix[y][j]/self.linearSys.matrix[i][j]
                            self.linearSys = self.linearSys.combine(y, i, a)

                for z in range( i+1, self.linearSys.numRows ) :
                    if self.linearSys.matrix[z][j] == 0 :
                        pass
                    else :
                        a = -self.linearSys.matrix[z][j]/self.linearSys.matrix[i][j]
                        self.linearSys = self.linearSys.combine(z, i, a)

                i = i + 1
                j = j + 1

        # Check for cases where the linear system may have no solution or infinite solutions
        # First check if the bottom row is all equal to zero

        count = 0
        for a in range(0, self.linearSys.numColumns-1) :
            if self.linearSys.matrix[self.linearSys.numRows-1][a] == 0 :
                count = count + 1

        # If so, check if it equal to zero or a non-zero value

        if count == self.linearSys.numColumns-1 :
            if self.linearSys.matrix[self.linearSys.numRows-1][self.linearSys.numColumns-1] == 0 :
                raise infiniteSolutionError("Sorry, the linear system you are trying to solve has infinite solutions.")
            else :
                raise noSolutionError("Sorry, the linear system you are trying to solve has no solution.")

        # Otherwise, create a column vector for the solution
        
        listOfElements = []
        for z in range(0, self.linearSys.numRows) :
            listOfElements.append( self.linearSys.matrix[z][self.linearSys.numColumns-1] )

        soln = ColumnVector( self.linearSys.numRows, listOfElements)

        return soln


# Error Classes :

# Error class for dealing with wrong dimensions

class dimensionError( Exception ) :
    pass

# Error class for dealing with a linear system with no solution

class noSolutionError( Exception ) :
    pass

# Error class for dealing with a linear system with infinite solutions

class infiniteSolutionError( Exception ) :
    pass

# Error class for dealing with a square matrix that's not invertible

class notInvertibleError( Exception ) :
    pass



# Tests :

a = Matrix(3,2,[[1,2], [3,4], [5,6]])
print(a)
print()
print(a.transpose())
print()
print(a.scale(0,2))
print()
print(a.swap(0,1))
print()
print(a.combine(0,1,2))
print()

# b = Matrix(3,2,[[1,1], [1,1], [1,1]])
# print(b)
# print(b.size())
# print()
# print(a.add(b))
# print()
# print(a.subtract(b))
# print()
# print(b.scalarMultiply(6))
# print()

# c = Matrix(3,3,[[1,2,3], [4,5,6], [7,8,9]])
# print(c)
# print(c.size())
# print()
# print(c.transpose())
# print()
# print(c.multiply(c))
# print()

# d = Matrix(3,2,[[1,1], [1,1], [1,1]])
# e = Matrix(3,2,[[1,1], [1,1], [1,1]])

# list2 = [ d, e ]
# print(a.sum(list2))
# print()

# Both these matrices have errors so they stop running and print an error msg
# Put try and except here

# f = Matrix(3,3,[[1,2,3], [4,5], [7,8,9]])
# g = Matrix(3,2,[[1,2,3], [4,5,6], [7,8,9]])

h = SquareMatrix(3, [[1,2,3],[0,5,6],[0,8,9]])
print(h)
print()
print(h.determinant())
print()
# print(h.scale(0,2))
# print()

i = SquareMatrix(2, [[1,2],[1,1]])
print(i)
print()
# print(i.matrix)
# print()
print(i.determinant())
print()
print(i.inverse())
print()

k = ColumnVector(4, [1,2,3,4])
print(k)
print()
print(k.dotProduct(k))
print()

l = LinearSystem(2,2, [[1,2,3],[4,5,6]])
print(l)
print()
print(l.solve())
print()

j = SquareMatrix(4, [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(j)
print()
print(j.matrix)
print()
print(j.determinant())
print()
print(j.inverse())
print()
