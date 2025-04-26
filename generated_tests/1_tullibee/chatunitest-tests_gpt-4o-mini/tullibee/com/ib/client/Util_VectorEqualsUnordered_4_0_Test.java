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
    public void testVectorEqualsUnordered_SameReference() {
        Vector<Integer> vector = new Vector<>();
        vector.add(1);
        vector.add(2);
        assertTrue(Util.VectorEqualsUnordered(vector, vector));
    }

    @Test
    public void testVectorEqualsUnordered_NullVectors() {
        assertTrue(Util.VectorEqualsUnordered(null, null));
    }

    @Test
    public void testVectorEqualsUnordered_OneNullVector() {
        Vector<Integer> vector = new Vector<>();
        vector.add(1);
        assertFalse(Util.VectorEqualsUnordered(vector, null));
        assertFalse(Util.VectorEqualsUnordered(null, vector));
    }

    @Test
    public void testVectorEqualsUnordered_DifferentSizes() {
        Vector<Integer> lhs = new Vector<>();
        lhs.add(1);
        Vector<Integer> rhs = new Vector<>();
        rhs.add(1);
        rhs.add(2);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_EmptyVectors() {
        Vector<Integer> lhs = new Vector<>();
        Vector<Integer> rhs = new Vector<>();
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_SameElementsDifferentOrder() {
        Vector<Integer> lhs = new Vector<>();
        lhs.add(1);
        lhs.add(2);
        Vector<Integer> rhs = new Vector<>();
        rhs.add(2);
        rhs.add(1);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_DifferentElements() {
        Vector<Integer> lhs = new Vector<>();
        lhs.add(1);
        lhs.add(2);
        Vector<Integer> rhs = new Vector<>();
        rhs.add(2);
        rhs.add(3);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_MatchingWithDuplicates() {
        Vector<Integer> lhs = new Vector<>();
        lhs.add(1);
        lhs.add(1);
        lhs.add(2);
        Vector<Integer> rhs = new Vector<>();
        rhs.add(2);
        rhs.add(1);
        rhs.add(1);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_NoMatchingElement() {
        Vector<Integer> lhs = new Vector<>();
        lhs.add(1);
        lhs.add(2);
        Vector<Integer> rhs = new Vector<>();
        rhs.add(3);
        rhs.add(4);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }
}
