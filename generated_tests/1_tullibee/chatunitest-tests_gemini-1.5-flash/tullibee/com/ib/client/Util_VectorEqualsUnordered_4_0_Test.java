package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    void testVectorEqualsUnordered_sameVectors() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        v1.add(3);
        assertTrue(Util.VectorEqualsUnordered(v1, v1));
    }

    @Test
    void testVectorEqualsUnordered_nullVectors() {
        assertTrue(Util.VectorEqualsUnordered(null, null));
    }

    @Test
    void testVectorEqualsUnordered_oneNullVector() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        assertFalse(Util.VectorEqualsUnordered(v1, null));
        assertFalse(Util.VectorEqualsUnordered(null, v1));
    }

    @Test
    void testVectorEqualsUnordered_emptyVectors() {
        Vector<Integer> v1 = new Vector<>();
        Vector<Integer> v2 = new Vector<>();
        assertTrue(Util.VectorEqualsUnordered(v1, v2));
    }

    @Test
    void testVectorEqualsUnordered_differentSizes() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        Vector<Integer> v2 = new Vector<>();
        v2.add(1);
        v2.add(2);
        v2.add(3);
        assertFalse(Util.VectorEqualsUnordered(v1, v2));
    }

    @Test
    void testVectorEqualsUnordered_sameElementsDifferentOrder() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        v1.add(3);
        Vector<Integer> v2 = new Vector<>();
        v2.add(3);
        v2.add(1);
        v2.add(2);
        assertTrue(Util.VectorEqualsUnordered(v1, v2));
    }

    @Test
    void testVectorEqualsUnordered_differentElements() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        v1.add(3);
        Vector<Integer> v2 = new Vector<>();
        v2.add(3);
        v2.add(1);
        v2.add(4);
        assertFalse(Util.VectorEqualsUnordered(v1, v2));
    }

    @Test
    void testVectorEqualsUnordered_duplicates() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        v1.add(2);
        Vector<Integer> v2 = new Vector<>();
        v2.add(2);
        v2.add(1);
        v2.add(2);
        assertTrue(Util.VectorEqualsUnordered(v1, v2));
    }

    @Test
    void testVectorEqualsUnordered_differentDuplicates() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        v1.add(2);
        Vector<Integer> v2 = new Vector<>();
        v2.add(2);
        v2.add(1);
        v2.add(3);
        assertFalse(Util.VectorEqualsUnordered(v1, v2));
    }
}
