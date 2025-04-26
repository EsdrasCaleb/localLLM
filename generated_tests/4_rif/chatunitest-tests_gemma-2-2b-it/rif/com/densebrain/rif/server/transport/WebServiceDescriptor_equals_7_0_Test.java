package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_equals_7_0_Test {

    @Test
    void testEquals() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(WebServiceDescriptor.class, "http://example.com/namespace1", "http://example.com/namespace2");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(WebServiceDescriptor.class, "http://example.com/namespace1", "http://example.com/namespace2");
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(WebServiceDescriptor.class, "http://example.com/namespace3", "http://example.com/namespace2");
        assertTrue(descriptor1.equals(descriptor2));
        assertFalse(descriptor1.equals(descriptor3));
    }
}
