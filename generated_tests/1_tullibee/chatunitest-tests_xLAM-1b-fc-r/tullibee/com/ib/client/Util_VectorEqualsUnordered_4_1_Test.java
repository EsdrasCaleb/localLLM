package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_VectorEqualsUnordered_4_1_Test {

    @Test
    public void testVectorEqualsUnordered() {
        // Test with equal vectors
        Vector vector1 = new Vector();
        Vector vector2 = new Vector();
        vector1.add(1);
        vector1.add(2);
        vector2.add(1);
        vector2.add(2);
        assertEquals(true, Util.VectorEqualsUnordered(vector1, vector2));
        // Test with unequal vectors
        Vector vector3 = new Vector();
        Vector vector4 = new Vector();
        vector3.add(1);
        vector3.add(2);
        vector4.add(1);
        vector4.add(3);
        assertEquals(false, Util.VectorEqualsUnordered(vector3, vector4));
        // Test with null vectors
        Vector vector5 = null;
        Vector vector6 = new Vector();
        assertEquals(true, Util.VectorEqualsUnordered(vector5, vector6));
        Vector vector7 = new Vector();
        Vector vector8 = null;
        assertEquals(true, Util.VectorEqualsUnordered(vector7, vector8));
        // Test with empty vectors
        Vector vector9 = new Vector();
        Vector vector10 = new Vector();
        assertEquals(true, Util.VectorEqualsUnordered(vector9, vector10));
    }
}
