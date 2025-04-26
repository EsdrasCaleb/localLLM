package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_VectorEqualsUnordered_4_0_Test {

    @Mock
    private Vector lhs;

    @Mock
    private Vector rhs;

    @Test
    public void testVectorEqualsUnordered_whenBothVectorsAreNull_thenReturnTrue() {
        when(lhs.size()).thenReturn(0);
        when(rhs.size()).thenReturn(0);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_whenVectorsHaveDifferentSize_thenReturnFalse() {
        when(lhs.size()).thenReturn(2);
        when(rhs.size()).thenReturn(3);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_whenVectorsHaveSameElements_thenReturnTrue() {
        when(lhs.size()).thenReturn(3);
        when(rhs.size()).thenReturn(3);
        when(lhs.get(0)).thenReturn("A");
        when(lhs.get(1)).thenReturn("B");
        when(lhs.get(2)).thenReturn("C");
        when(rhs.get(0)).thenReturn("C");
        when(rhs.get(1)).thenReturn("A");
        when(rhs.get(2)).thenReturn("B");
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_whenVectorsHaveDifferentElements_thenReturnFalse() {
        when(lhs.size()).thenReturn(3);
        when(rhs.size()).thenReturn(3);
        when(lhs.get(0)).thenReturn("A");
        when(lhs.get(1)).thenReturn("B");
        when(lhs.get(2)).thenReturn("C");
        when(rhs.get(0)).thenReturn("D");
        when(rhs.get(1)).thenReturn("E");
        when(rhs.get(2)).thenReturn("F");
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }
}
