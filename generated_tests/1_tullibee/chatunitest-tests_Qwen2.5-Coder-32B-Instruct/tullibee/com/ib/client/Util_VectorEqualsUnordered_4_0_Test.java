package com.ib.client;

import java.lang.reflect.Method;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    public void testVectorEqualsUnordered() throws Exception {
        // Test case 1: Both vectors are null
        assertTrue(Util.VectorEqualsUnordered(null, null));
        // Test case 2: One vector is null
        assertFalse(Util.VectorEqualsUnordered(new Vector<>(), null));
        assertFalse(Util.VectorEqualsUnordered(null, new Vector<>()));
        // Test case 3: Both vectors are empty
        assertTrue(Util.VectorEqualsUnordered(new Vector<>(), new Vector<>()));
        // Test case 4: Vectors have the same elements in the same order
        Vector<String> vector1 = new Vector<>();
        vector1.add("a");
        vector1.add("b");
        Vector<String> vector2 = new Vector<>();
        vector2.add("a");
        vector2.add("b");
        assertTrue(Util.VectorEqualsUnordered(vector1, vector2));
        // Test case 5: Vectors have the same elements in different order
        Vector<String> vector3 = new Vector<>();
        vector3.add("b");
        vector3.add("a");
        assertTrue(Util.VectorEqualsUnordered(vector1, vector3));
        // Test case 6: Vectors have different elements
        Vector<String> vector4 = new Vector<>();
        vector4.add("a");
        vector4.add("c");
        assertFalse(Util.VectorEqualsUnordered(vector1, vector4));
        // Test case 7: Vectors have different sizes
        Vector<String> vector5 = new Vector<>();
        vector5.add("a");
        assertFalse(Util.VectorEqualsUnordered(vector1, vector5));
        // Test case 8: Vectors with duplicate elements in the same order
        Vector<String> vector6 = new Vector<>();
        vector6.add("a");
        vector6.add("a");
        Vector<String> vector7 = new Vector<>();
        vector7.add("a");
        vector7.add("a");
        assertTrue(Util.VectorEqualsUnordered(vector6, vector7));
        // Test case 9: Vectors with duplicate elements in different order
        Vector<String> vector8 = new Vector<>();
        vector8.add("a");
        vector8.add("b");
        vector8.add("a");
        Vector<String> vector9 = new Vector<>();
        vector9.add("a");
        vector9.add("a");
        vector9.add("b");
        assertTrue(Util.VectorEqualsUnordered(vector8, vector9));
        // Test case 10: Vectors with duplicate elements but different counts
        Vector<String> vector10 = new Vector<>();
        vector10.add("a");
        vector10.add("a");
        Vector<String> vector11 = new Vector<>();
        vector11.add("a");
        assertFalse(Util.VectorEqualsUnordered(vector10, vector11));
    }
}
