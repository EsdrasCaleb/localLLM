package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class WebServiceDescriptor_equals_7_0_Test {

    private WebServiceDescriptor descriptor1;

    private WebServiceDescriptor descriptor2;

    @BeforeEach
    void setUp() {
        descriptor1 = new WebServiceDescriptor(String.class, "namespace1", "typesNamespace1");
        descriptor2 = new WebServiceDescriptor(String.class, "namespace2", "typesNamespace2");
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(descriptor1.equals(descriptor1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(descriptor1.equals(null));
    }

    @Test
    void testEquals_DifferentClass() {
        assertFalse(descriptor1.equals("DifferentClass"));
    }

    @Test
    void testEquals_SameServiceClazz() {
        assertTrue(descriptor1.equals(descriptor2));
    }

    @Test
    void testEquals_DifferentServiceClazz() {
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(Integer.class, "namespace1", "typesNamespace1");
        assertFalse(descriptor1.equals(descriptor3));
    }

    @Test
    void testEquals_ServiceClazzNull() {
        descriptor1.setServiceClazz(null);
        descriptor2.setServiceClazz(null);
        assertTrue(descriptor1.equals(descriptor2));
    }

    @Test
    void testEquals_ServiceClazzNullOtherNotNull() {
        descriptor1.setServiceClazz(null);
        assertFalse(descriptor1.equals(descriptor2));
    }

    @Test
    void testEquals_ServiceClazzNotNullOtherNull() {
        descriptor2.setServiceClazz(null);
        assertFalse(descriptor1.equals(descriptor2));
    }
}
