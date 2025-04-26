package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    void testVectorEqualsUnordered() {
        Vector<Integer> vector1 = Mockito.mock(Vector.class);
        Mockito.when(vector1.size()).thenReturn(2);
        Mockito.when(vector1.get(0)).thenReturn(1);
        Mockito.when(vector1.get(1)).thenReturn(2);
        Vector<Integer> vector2 = Mockito.mock(Vector.class);
        Mockito.when(vector2.size()).thenReturn(2);
        Mockito.when(vector2.get(0)).thenReturn(1);
        Mockito.when(vector2.get(1)).thenReturn(2);
        boolean result = Util.VectorEqualsUnordered(vector1, vector2);
        assertTrue(result);
    }
}
