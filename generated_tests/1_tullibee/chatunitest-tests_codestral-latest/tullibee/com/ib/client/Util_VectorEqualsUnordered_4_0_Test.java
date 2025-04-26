package com.ib.client;

import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    public void testVectorEqualsUnordered() {
        // Test case 1: Both vectors are null
        assertTrue(Util.VectorEqualsUnordered(null, null));
        // Test case 2: One vector is null, the other is not
        Vector<Integer> lhs = new Vector<>();
        assertFalse(Util.VectorEqualsUnordered(lhs, null));
        assertFalse(Util.VectorEqualsUnordered(null, lhs));
        // Test case 3: Both vectors are empty
        assertTrue(Util.VectorEqualsUnordered(lhs, new Vector<>()));
        // Test case 4: Vectors have different sizes
        lhs.add(1);
        assertFalse(Util.VectorEqualsUnordered(lhs, new Vector<>()));
        // Test case 5: Vectors have the same elements in the same order
        Vector<Integer> rhs = new Vector<>();
        rhs.add(1);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
        // Test case 6: Vectors have the same elements in different orders
        lhs.add(2);
        rhs.add(0, 2);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
        // Test case 7: Vectors have different elements
        rhs.set(1, 3);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
        // Test case 8: Vectors have the same elements but one has duplicates
        lhs.add(2);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
        // Test case 9: Vectors have the same elements but different types
        Vector<Object> objLhs = new Vector<>();
        objLhs.add(1);
        Vector<Object> objRhs = new Vector<>();
        objRhs.add("1");
        assertFalse(Util.VectorEqualsUnordered(objLhs, objRhs));
    }
}
