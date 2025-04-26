package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_equals_7_0_Test {

    @Test
    public void testEquals() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "http://example.com", "http://example.com/types");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "http://example.com", "http://example.com/types");
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(Integer.class, "http://example.com", "http://example.com/types");
        assertTrue(descriptor1.equals(descriptor2));
        assertFalse(descriptor1.equals(descriptor3));
    }
}
