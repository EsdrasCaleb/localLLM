// Test method
package com.densebrain.rif.server.transport;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class WebServiceDescriptor_hashCode_6_0_Test {

    @Test
    public void testHashCode() {
        // Create a mock object of WebServiceDescriptor
        WebServiceDescriptor mockDescriptor = mock(WebServiceDescriptor.class);
        // Set the return value of the hashCode method to a known value
        when(mockDescriptor.hashCode()).thenReturn(12345);
        // Call the hashCode method on the mock object
        int result = mockDescriptor.hashCode();
        // Verify that the hashCode method returned the expected value
        assertEquals(12345, result);
    }

    @Test
    public void testEquals() {
        // Create a mock object of WebServiceDescriptor
        WebServiceDescriptor mockDescriptor1 = mock(WebServiceDescriptor.class);
        WebServiceDescriptor mockDescriptor2 = mock(WebServiceDescriptor.class);
        // Set the return value of the equals method to a known value
        when(mockDescriptor1.equals(mockDescriptor2)).thenReturn(true);
        // Call the equals method on the mock object
        boolean result = mockDescriptor1.equals(mockDescriptor2);
        // Verify that the equals method returned the expected value
        assertTrue(result);
    }

    @Test
    public void testEqualsWithNull() {
        // Create a mock object of WebServiceDescriptor
        WebServiceDescriptor mockDescriptor1 = mock(WebServiceDescriptor.class);
        // Call the equals method on the mock object with null
        boolean result = mockDescriptor1.equals(null);
        // Verify that the equals method returned the expected value
        assertFalse(result);
    }
}
