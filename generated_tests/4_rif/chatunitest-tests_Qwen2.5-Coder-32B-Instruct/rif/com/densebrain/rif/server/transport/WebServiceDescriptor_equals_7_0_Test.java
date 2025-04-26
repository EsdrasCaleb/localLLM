package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_equals_7_0_Test {

    private WebServiceDescriptor descriptor1;

    private WebServiceDescriptor descriptor2;

    private WebServiceDescriptor descriptor3;

    private WebServiceDescriptor descriptor4;

    @BeforeEach
    public void setUp() {
        descriptor1 = new WebServiceDescriptor(String.class, "namespace1", "typesNamespace1");
        descriptor2 = new WebServiceDescriptor(String.class, "namespace2", "typesNamespace2");
        descriptor3 = new WebServiceDescriptor(Integer.class, "namespace1", "typesNamespace1");
        descriptor4 = new WebServiceDescriptor(String.class, "namespace1", "typesNamespace1");
    }

    @Test
    public void testEquals_SameInstance_ReturnsTrue() {
        assertTrue(descriptor1.equals(descriptor1));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        assertFalse(descriptor1.equals(null));
    }

    @Test
    public void testEquals_DifferentClass_ReturnsFalse() {
        assertFalse(descriptor1.equals(new Object()));
    }

    @Test
    public void testEquals_SameServiceClass_ReturnsTrue() {
        assertTrue(descriptor1.equals(descriptor4));
    }

    @Test
    public void testEquals_DifferentServiceClass_ReturnsFalse() {
        assertFalse(descriptor1.equals(descriptor3));
    }

    @Test
    public void testEquals_ServiceClassNullInBoth_ReturnsTrue() {
        WebServiceDescriptor nullDescriptor1 = new WebServiceDescriptor(null, "namespace1", "typesNamespace1");
        WebServiceDescriptor nullDescriptor2 = new WebServiceDescriptor(null, "namespace2", "typesNamespace2");
        assertTrue(nullDescriptor1.equals(nullDescriptor2));
    }

    @Test
    public void testEquals_ServiceClassNullInOne_ReturnsFalse() {
        WebServiceDescriptor nullDescriptor = new WebServiceDescriptor(null, "namespace1", "typesNamespace1");
        assertFalse(nullDescriptor.equals(descriptor1));
        assertFalse(descriptor1.equals(nullDescriptor));
    }
}
