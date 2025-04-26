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
    public void testEqualsWithNull() {
        // Create a mock object of WebServiceDescriptor
        WebServiceDescriptor mockDescriptor1 = mock(WebServiceDescriptor.class);
        // Call the equals method on the mock object with null
        boolean result = mockDescriptor1.equals(null);
        // Verify that the equals method returned the expected value
        assertFalse(result);
    }
}
