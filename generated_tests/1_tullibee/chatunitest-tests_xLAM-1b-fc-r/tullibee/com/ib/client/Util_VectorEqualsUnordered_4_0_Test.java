package com.ib.client;

import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    public void testVectorEqualsUnordered() {
        // Test with null vectors
        assertTrue(Util.VectorEqualsUnordered(null, null));
        assertFalse(Util.VectorEqualsUnordered(new Vector(), null));
        assertFalse(Util.VectorEqualsUnordered(null, new Vector()));
        // Test with equal vectors
        Vector v1 = new Vector(Arrays.asList(1, 2, 3));
        Vector v2 = new Vector(Arrays.asList(1, 2, 3));
        assertTrue(Util.VectorEqualsUnordered(v1, v2));
        // Test with unequal vectors
        Vector v3 = new Vector(Arrays.asList(1, 2, 3));
        Vector v4 = new Vector(Arrays.asList(1, 2, 4));
        assertFalse(Util.VectorEqualsUnordered(v3, v4));
        // Test with empty vectors
        Vector v5 = new Vector();
        Vector v6 = new Vector();
        assertTrue(Util.VectorEqualsUnordered(v5, v6));
        // Test with vectors with different sizes
        Vector v7 = new Vector(Arrays.asList(1, 2, 3));
        Vector v8 = new Vector(Arrays.asList(1, 2));
        assertFalse(Util.VectorEqualsUnordered(v7, v8));
    }
}
