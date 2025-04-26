package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class WebServiceDescriptor_hashCode_6_3_Test {

    @Test
    void testHashCode() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "http://example.com", "http://example.com");
        assertEquals(descriptor.hashCode(), descriptor.getServiceClazz().hashCode());
    }
}
