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
        // Create mock objects for the vectors
        Vector<Object> lhsMock = Mockito.mock(Vector.class);
        Vector<Object> rhsMock = Mockito.mock(Vector.class);
        // Set up expectations for the mock objects
        when(lhsMock.contains(Mockito.any())).thenReturn(true);
        when(rhsMock.contains(Mockito.any())).thenReturn(true);
        when(lhsMock.indexOf(Mockito.any())).thenReturn(1);
        when(rhsMock.indexOf(Mockito.any())).thenReturn(2);
        // Call the method to be tested
        boolean result = Util.VectorEqualsUnordered(lhsMock, rhsMock);
        // Verify the result
        assertEquals(true, result);
    }
}
