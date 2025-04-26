package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_hashCode_6_0_Test {

    @Test
    public void testHashCode() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(WebServiceDescriptor.class, "http://example.com", "http://example.com");
        assertEquals(1, descriptor.hashCode());
    }
}
