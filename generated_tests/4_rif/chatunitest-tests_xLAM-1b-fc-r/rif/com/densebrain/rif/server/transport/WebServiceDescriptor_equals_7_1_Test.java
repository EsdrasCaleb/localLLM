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
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(Class.class, "targetNamespace", "typesNamespace");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(Class.class, "targetNamespace", "typesNamespace");
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(Class.class, "differentNamespace", "typesNamespace");
        WebServiceDescriptor descriptor4 = new WebServiceDescriptor(Class.class, "targetNamespace", "differentNamespace");
        assertTrue(descriptor1.equals(descriptor2));
        assertFalse(descriptor1.equals(descriptor3));
        assertFalse(descriptor1.equals(descriptor4));
    }
}
