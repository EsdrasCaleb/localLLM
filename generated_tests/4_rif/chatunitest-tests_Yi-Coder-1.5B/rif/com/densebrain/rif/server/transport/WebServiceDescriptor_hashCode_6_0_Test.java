package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class WebServiceDescriptor_hashCode_6_0_Test {

    @Test
    public void testHashCode() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(WebServiceDescriptor.class, "http://www.example.com", "http://www.example.com/types");
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(WebServiceDescriptor.class, "http://www.example.com", "http://www.example.com/types");
        assertEquals(descriptor.hashCode(), descriptor1.hashCode());
    }
}
