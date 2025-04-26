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
        // Test with two null vectors
        assertEquals(true, Util.VectorEqualsUnordered(null, null));
        // Test with two empty vectors
        assertEquals(true, Util.VectorEqualsUnordered(new Vector(), new Vector()));
        // Test with two vectors with different sizes
        Vector v1 = new Vector();
        v1.addElement(new Integer(1));
        v1.addElement(new Integer(2));
        Vector v2 = new Vector();
        v2.addElement(new Integer(1));
        v2.addElement(new Integer(2));
        v2.addElement(new Integer(3));
        assertEquals(false, Util.VectorEqualsUnordered(v1, v2));
        // Test with two vectors with same elements but different order
        Vector v3 = new Vector();
        v3.addElement(new Integer(1));
        v3.addElement(new Integer(2));
        Vector v4 = new Vector();
        v4.addElement(new Integer(2));
        v4.addElement(new Integer(1));
        assertEquals(false, Util.VectorEqualsUnordered(v3, v4));
        // Test with two vectors with same elements
        Vector v5 = new Vector();
        v5.addElement(new Integer(1));
        v5.addElement(new Integer(2));
        Vector v6 = new Vector();
        v6.addElement(new Integer(1));
        v6.addElement(new Integer(2));
        assertEquals(true, Util.VectorEqualsUnordered(v5, v6));
    }
}
