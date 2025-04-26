package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_equals_7_3_Test {

    private WebServiceDescriptor webServiceDescriptor;

    @Test
    public void testEquals() {
        // Create an instance of WebServiceDescriptor
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(MyClass.class, "http://example.com", "http://example.com");
        // Create another instance of WebServiceDescriptor
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(MyClass.class, "http://example.com", "http://example.com");
        // Test the equals method
        boolean result1 = descriptor1.equals(descriptor2);
        boolean result2 = descriptor2.equals(descriptor1);
        // Verify the results
        assertTrue(result1);
        assertTrue(result2);
    }
}

class MyClass {
    // Implementation details
}
