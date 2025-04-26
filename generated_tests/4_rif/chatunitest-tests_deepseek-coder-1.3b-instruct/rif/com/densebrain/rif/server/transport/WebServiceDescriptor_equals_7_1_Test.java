package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class WebServiceDescriptor_equals_7_1_Test {

    @Test
    void testEquals() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "http://target1", "http://types1");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "http://target1", "http://types1");
        assertTrue(descriptor1.equals(descriptor2));
        descriptor2.setServiceClazz(Integer.class);
        assertFalse(descriptor1.equals(descriptor2));
        descriptor2.setServiceClazz(null);
        assertFalse(descriptor1.equals(descriptor2));
        descriptor2 = new WebServiceDescriptor(Integer.class, "http://target2", "http://types2");
        assertFalse(descriptor1.equals(descriptor2));
        descriptor2.setTargetNamespace("http://target2");
        assertFalse(descriptor1.equals(descriptor2));
        descriptor2.setTypesNamespace("http://types2");
        assertFalse(descriptor1.equals(descriptor2));
    }
}
